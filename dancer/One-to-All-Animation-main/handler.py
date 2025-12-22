import runpod
from runpod.serverless.utils import rp_upload
import os
import base64
import json
import uuid
import logging
import subprocess
import tempfile
import shutil
from pathlib import Path

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 导入推理模块
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
video_gen_path = os.path.join(BASE_DIR, 'video-generation')
sys.path.insert(0, video_gen_path)
# 保存原始工作目录
original_cwd = os.getcwd()

import torch
import torch.multiprocessing as mp
from PIL import Image
import imageio
import numpy as np
from functools import partial
import decord
from infer_utils import load_poses_whole_video, resizecrop
from safetensors.torch import load_file as safe_load
from opensora.sample.pipeline_wanx_vhuman_tokenreplace import WanPipeline
from opensora.model_variants.wanx_diffusers_src import WanTransformer3DModel_Refextractor_2D_Controlnet_prefix
from opensora.encoder_variants import get_text_enc
from opensora.vae_variants import get_vae
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

# ===== 配置 =====
# BASE_DIR 已在上面定义
model_path = os.path.join(BASE_DIR, "pretrained_models/Wan2.1-T2V-1.3B-Diffusers")
vae_path = os.path.join(BASE_DIR, "pretrained_models/Wan2.1-T2V-1.3B-Diffusers/vae")
config_path = os.path.join(video_gen_path, "configs/wan2.1_t2v_1.3b.json")
model_dtype = torch.bfloat16
ckpt_path = os.path.join(BASE_DIR, "checkpoints/One-to-All-1.3b_2")
max_short = 384
MAIN_CHUNK = 81
OVERLAP_FRAMES = 5
FINAL_CHUNK_CANDIDATES = [65, 69, 73, 77, 81]

negative_prompt = [
    "black background, Aerial view, aerial view, overexposed, low quality, deformation, a poor composition, bad hands, bad teeth, bad eyes, bad limbs, distortion"
]
black_pose_cfg = True
black_image_cfg = True
controlnet_conditioning_scale = 1.0
case1 = False

# ===== 核心函数 =====

def build_pipe(device, ckpt_path):
    """构建推理管道"""
    # 确保在正确的目录下执行（某些模块可能需要相对路径）
    current_dir = os.getcwd()
    try:
        if os.getcwd() != video_gen_path:
            os.chdir(video_gen_path)
        
        scheduler = FlowMatchEulerDiscreteScheduler(
            shift=7.0,
            num_train_timesteps=1000,
            use_dynamic_shifting=False
        )

        vae = get_vae('wanx', vae_path, model_dtype)
        encoders = get_text_enc('wanx-t2v', model_path, model_dtype)
        text_encoder = encoders.text_encoder
        tokenizer = encoders.tokenizer

        model = WanTransformer3DModel_Refextractor_2D_Controlnet_prefix.from_config(config_path).to(model_dtype)
        model.set_up_controlnet(os.path.join(video_gen_path, "configs/wan2.1_t2v_1.3b_controlnet_2.json"), model_dtype)
        model.set_up_refextractor(os.path.join(video_gen_path, "configs/wan2.1_t2v_1.3b_refextractor_2d_withmask2.json"), model_dtype)
        model.eval()
        model.requires_grad_(False)
        
        # 加载checkpoint
        checkpoint = {}
        shard_files = [f for f in os.listdir(ckpt_path) if f.endswith(".safetensors")]
        for shard_file in shard_files:
            sd = safe_load(os.path.join(ckpt_path, shard_file), device='cpu')
            checkpoint.update(sd)
        model.load_state_dict(checkpoint, strict=True)

        pipe = WanPipeline(
            transformer=model,
            vae=vae.vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            scheduler=scheduler
        )
        pipe.to(device, dtype=model_dtype)
        return pipe
    finally:
        # 恢复原始目录
        if os.getcwd() != current_dir:
            os.chdir(current_dir)

def build_split_plan(total_len: int):
    """构建分块计划"""
    ranges = []
    start = 0
    while True:
        current_chunk_end = start + MAIN_CHUNK
        next_chunk_start = start + (MAIN_CHUNK - OVERLAP_FRAMES)
        if next_chunk_start + MAIN_CHUNK >= total_len:
            ranges.append((start, current_chunk_end))
            final_chunk_start = -1
            for length in FINAL_CHUNK_CANDIDATES:
                potential_start = total_len - length
                if potential_start < current_chunk_end - OVERLAP_FRAMES:
                    final_chunk_start = potential_start
                    break
            if final_chunk_start == -1:
                final_chunk_start = next_chunk_start
            ranges.append((final_chunk_start, total_len))
            break
        else:
            ranges.append((start, current_chunk_end))
            start = next_chunk_start
    return ranges

def preprocess_one(reference_path, video_path, frame_interval, do_align, alignmode, h=None, w=None, face_change=True, head_change=True, without_face=False):
    """预处理单个任务，返回预处理结果"""
    # 确保在正确的目录下执行
    current_dir = os.getcwd()
    try:
        if os.getcwd() != video_gen_path:
            os.chdir(video_gen_path)
        
        if not h or not w:
            ref_img_tmp = Image.open(reference_path).convert("RGB")
            w, h = ref_img_tmp.size
        else:
            h = int(h)
            w = int(w)

        max_short_preprocess = 768
        if min(h, w) > max_short_preprocess:
            if h < w:
                scale = max_short_preprocess / h
                h, w = max_short_preprocess, int(w * scale)
            else:
                scale = max_short_preprocess / w
                w, h = max_short_preprocess, int(h * scale)
        new_h = (h // 16) * 16
        new_w = (w // 16) * 16
        transform = partial(resizecrop, th=new_h, tw=new_w)
        anchor_idx = 0

        pose_tensor, image_input, pose_input, mask_input = load_poses_whole_video(
            video_path=video_path,
            reference=reference_path,
            frame_interval=frame_interval,
            do_align=do_align,
            transform=transform,
            alignmode=alignmode,
            anchor_idx=anchor_idx,
            face_change=face_change,
            head_change=head_change,
            without_face=without_face,
        )

        if os.path.isdir(video_path):
            fps = 30
            logger.info(f"⚠ {video_path} isdir, use default fps={fps}")
        else:
            vr = decord.VideoReader(video_path)
            fps = vr.get_avg_fps()

        output_fps = fps / frame_interval

        return {
            'pose_tensor': pose_tensor,  # (T,C,H,W) 0-255 uint8格式，推理时再归一化
            'image_input': image_input,
            'pose_input': pose_input,
            'mask_input': mask_input,
            'fps': output_fps,
            'preprocess_h': new_h,
            'preprocess_w': new_w,
        }
    finally:
        # 恢复原始目录
        if os.getcwd() != current_dir:
            os.chdir(current_dir)

# ===== 工具函数 =====

def process_input(input_data, temp_dir, output_filename, input_type):
    """处理输入数据并返回文件路径"""
    if input_type == "path":
        logger.info(f"📁 路径输入: {input_data}")
        if not os.path.exists(input_data):
            raise Exception(f"文件不存在: {input_data}")
        return input_data
    elif input_type == "url":
        logger.info(f"🌐 URL输入: {input_data}")
        os.makedirs(temp_dir, exist_ok=True)
        file_path = os.path.abspath(os.path.join(temp_dir, output_filename))
        return download_file_from_url(input_data, file_path)
    elif input_type == "base64":
        logger.info(f"🔢 Base64输入")
        return save_base64_to_file(input_data, temp_dir, output_filename)
    else:
        raise Exception(f"不支持的输入类型: {input_type}")

def download_file_from_url(url, output_path):
    """从URL下载文件"""
    try:
        result = subprocess.run(
            ['wget', '-O', output_path, '--no-verbose', url],
            capture_output=True, text=True, timeout=300
        )
        if result.returncode == 0:
            logger.info(f"✅ 下载成功: {url} -> {output_path}")
            return output_path
        else:
            raise Exception(f"URL下载失败: {result.stderr}")
    except Exception as e:
        logger.error(f"❌ 下载错误: {e}")
        raise

def is_base64_string(s):
    """检测字符串是否是 base64 编码"""
    if not isinstance(s, str):
        return False
    # base64 字符串通常很长（至少几百字符），且只包含 base64 字符
    if len(s) < 100:
        return False
    # 检查是否包含 base64 字符集（A-Z, a-z, 0-9, +, /, =）
    base64_chars = set('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=')
    # 如果字符串中超过 90% 的字符是 base64 字符，很可能是 base64
    if len(s) > 0:
        base64_ratio = sum(1 for c in s if c in base64_chars) / len(s)
        return base64_ratio > 0.9
    return False

def save_base64_to_file(base64_data, temp_dir, output_filename):
    """将Base64数据保存为文件"""
    try:
        # 处理 data URL 格式
        if ',' in base64_data:
            base64_data = base64_data.split(',')[1]
        
        decoded_data = base64.b64decode(base64_data)
        os.makedirs(temp_dir, exist_ok=True)
        file_path = os.path.abspath(os.path.join(temp_dir, output_filename))
        with open(file_path, 'wb') as f:
            f.write(decoded_data)
        logger.info(f"✅ Base64已保存: {file_path}")
        return file_path
    except Exception as e:
        logger.error(f"❌ Base64解码失败: {e}")
        raise Exception(f"Base64解码失败: {e}")

def generate_single_video(
    reference_path,
    video_path,
    output_path,
    frame_interval=1,
    do_align=True,
    alignmode="ref",
    h=None,
    w=None,
    face_change=True,
    head_change=False,
    without_face=False,
    new_h=None,
    new_w=None,
    ref_cfg=2.5,
    pose_cfg=1.5,
    prompt=""
):
    """生成单个视频"""
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # 构建模型
    logger.info("构建模型...")
    pipe = build_pipe(device, ckpt_path)
    
    # 预处理
    logger.info("预处理中...")
    preprocess_data = preprocess_one(
        reference_path=reference_path,
        video_path=video_path,
        frame_interval=frame_interval,
        do_align=do_align,
        alignmode=alignmode,
        h=h,
        w=w,
        face_change=face_change,
        head_change=head_change,
        without_face=without_face
    )
    
    # 准备输入
    image_input = preprocess_data['image_input']
    pose_input_img = preprocess_data['pose_input']
    mask_input = preprocess_data['mask_input']
    pose_tensor = preprocess_data['pose_tensor']
    pose_fps = preprocess_data['fps']
    preprocess_h = preprocess_data['preprocess_h']
    preprocess_w = preprocess_data['preprocess_w']
    
    # 确定推理分辨率
    if new_h and new_w:
        new_h = float(new_h)
        new_w = float(new_w)
    else:
        new_h, new_w = preprocess_h, preprocess_w
    
    if min(new_h, new_w) > max_short:
        if new_h < new_w:
            scale = max_short / new_h
            new_h, new_w = max_short, int(new_w * scale)
        else:
            scale = max_short / new_w
            new_w, new_h = max_short, int(new_h * scale)
            
    new_h, new_w = int(new_h//16*16), int(new_w//16*16)
    transform_fn = partial(resizecrop, th=new_h, tw=new_w)
    image_input = transform_fn(image_input)
    
    # 转换pose_tensor并应用transform
    pose_tensor_np = pose_tensor.permute(0, 2, 3, 1).numpy().astype(np.uint8)  # (T,H,W,C)
    pose_frames_list = []
    for frame_np in pose_tensor_np:
        pil_img = Image.fromarray(frame_np)
        pil_img = transform_fn(pil_img)
        frame_np = np.array(pil_img)
        pose_frames_list.append(frame_np)
    pose_frames_np = np.stack(pose_frames_list)
    pose_tensor = torch.from_numpy(pose_frames_np).float().permute(3, 0, 1, 2) / 255.0 * 2 - 1
    pose_tensor = pose_tensor.unsqueeze(0)
    
    pose_input_img = transform_fn(pose_input_img)
    mask_input = transform_fn(mask_input)
    
    # 转换为灰度图以确保是单通道
    if mask_input.mode != "L":
        mask_input = mask_input.convert("L")
    mask_np = np.array(mask_input, dtype=np.float32) / 255.0
    mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).unsqueeze(2)
    
    src_pose_tensor = torch.from_numpy(np.array(pose_input_img)).unsqueeze(0).float().permute(0, 3, 1, 2) / 255.0 * 2 - 1
    src_pose_tensor = src_pose_tensor.unsqueeze(2)
    
    # 分块处理
    split_plan = build_split_plan(pose_tensor.shape[2])
    all_generated_frames_np = {}
    
    logger.info(f"开始推理，共 {len(split_plan)} 个分块...")
    for idx, (start, end) in enumerate(split_plan):
        logger.info(f"处理分块 {idx+1}/{len(split_plan)}: 帧 {start}-{end}")
        sub_video = pose_tensor[:, :, start:end]
        prev_frames = None
        if start > 0:
            needed_idx = range(start, start + OVERLAP_FRAMES)
            if all(k in all_generated_frames_np for k in needed_idx):
                prev_frames = [
                    Image.fromarray(all_generated_frames_np[k]) for k in needed_idx
                ]
        
        output_chunk = pipe(
            image=image_input,
            image_mask=mask_tensor,
            control_video=sub_video,
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=new_h,
            width=new_w,
            num_frames=end - start,
            image_guidance_scale=ref_cfg,
            pose_guidance_scale=pose_cfg,
            num_inference_steps=30,
            generator=torch.Generator(device=device).manual_seed(42),
            black_image_cfg=black_image_cfg,
            black_pose_cfg=black_pose_cfg,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            return_tensor=True,
            case1=case1,
            token_replace=(prev_frames is not None),
            prev_frames=prev_frames,
            image_pose=src_pose_tensor
        ).frames
        
        output_chunk = (
            output_chunk[0].detach().cpu() / 2 + 0.5
        ).float().clamp(0, 1).permute(1, 2, 3, 0).numpy()
        output_chunk = (output_chunk * 255).astype("uint8")
        
        for j in range(end - start):
            gidx = start + j
            all_generated_frames_np[gidx] = output_chunk[j]
    
    # 合成最终视频
    logger.info("合成最终视频...")
    alpha = 0.6
    frames_combined = []
    
    src_uint8 = ((pose_tensor / 2 + 0.5).clamp(0, 1) * 255
                )[0].byte().permute(1, 2, 3, 0).numpy()
    sorted_idx = sorted(all_generated_frames_np.keys())
    
    for t in sorted_idx:
        src  = src_uint8[t].astype(np.float32)
        pred = all_generated_frames_np[t].astype(np.float32)
        blended = (alpha * src + (1 - alpha) * pred).round().astype(np.uint8)
        concat  = np.concatenate([blended, pred.astype(np.uint8)], axis=1)
        frames_combined.append(concat)
    
    imageio.mimwrite(
        output_path,
        frames_combined,
        fps=pose_fps,
        quality=5
    )
    
    logger.info(f"✅ 视频生成完成: {output_path}")
    del pipe
    torch.cuda.empty_cache()
    
    return output_path

def handler(job):
    """
    RunPod handler for One-to-All-Animation
    
    支持的输入参数:
    - reference_image: 参考图片 (路径/URL/base64) (必需)
    - driving_video: 驱动视频 (路径/URL/base64) (必需)
    - frame_interval: 帧间隔 (默认: 1)
    - do_align: 是否对齐 (默认: True)
    - alignmode: 对齐模式 "ref" 或 "pose" (默认: "ref")
    - height: 输出高度 (可选)
    - width: 输出宽度 (可选)
    - face_change: 是否淡化面部关键点 (默认: True)
    - head_change: 是否淡化头部关键点 (默认: False)
    - without_face: 是否跳过面部关键点 (默认: False)
    - inference_height: 推理高度 (可选)
    - inference_width: 推理宽度 (可选)
    - ref_cfg: 参考图像引导强度 (默认: 2.5)
    - pose_cfg: 姿态引导强度 (默认: 1.5)
    - prompt: 文本提示词 (可选)
    """
    job_input = job.get("input", {})
    
    # 记录输入（排除base64数据，但显示长度信息）
    log_input = {}
    for k, v in job_input.items():
        if k in ["reference_image", "driving_video"]:
            if isinstance(v, str):
                log_input[k] = f"<base64 data, length: {len(v)}>"
            else:
                log_input[k] = f"<{type(v).__name__}>"
        else:
            log_input[k] = v
    logger.info(f"收到任务输入: {log_input}")
    logger.info(f"输入参数键: {list(job_input.keys())}")
    
    task_id = f"task_{uuid.uuid4()}"
    temp_dir = os.path.join("/tmp", task_id)
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        # 处理参考图片
        reference_path = None
        if "reference_image_path" in job_input:
            reference_path = process_input(job_input["reference_image_path"], temp_dir, "reference.jpg", "path")
        elif "reference_image_url" in job_input:
            reference_path = process_input(job_input["reference_image_url"], temp_dir, "reference.jpg", "url")
        elif "reference_image" in job_input:
            # 自动检测是base64、URL还是路径
            ref_input = job_input["reference_image"]
            if isinstance(ref_input, str):
                if ref_input.startswith("http://") or ref_input.startswith("https://"):
                    reference_path = process_input(ref_input, temp_dir, "reference.jpg", "url")
                elif is_base64_string(ref_input):
                    # 优先检测 base64（因为 base64 字符串通常很长）
                    reference_path = process_input(ref_input, temp_dir, "reference.jpg", "base64")
                else:
                    # 尝试作为路径检查
                    try:
                        if os.path.exists(ref_input):
                            reference_path = process_input(ref_input, temp_dir, "reference.jpg", "path")
                        else:
                            # 如果路径不存在，尝试作为 base64 处理
                            reference_path = process_input(ref_input, temp_dir, "reference.jpg", "base64")
                    except Exception:
                        # 如果路径检查出错，尝试作为 base64 处理
                        reference_path = process_input(ref_input, temp_dir, "reference.jpg", "base64")
            else:
                raise Exception(f"reference_image 参数类型错误: {type(ref_input)}")
        else:
            raise Exception("缺少必需参数: reference_image")
        
        # 处理驱动视频
        video_path = None
        if "driving_video_path" in job_input:
            video_path = process_input(job_input["driving_video_path"], temp_dir, "driving.mp4", "path")
        elif "driving_video_url" in job_input:
            video_path = process_input(job_input["driving_video_url"], temp_dir, "driving.mp4", "url")
        elif "driving_video" in job_input:
            # 自动检测是base64、URL还是路径
            vid_input = job_input["driving_video"]
            if isinstance(vid_input, str):
                if vid_input.startswith("http://") or vid_input.startswith("https://"):
                    video_path = process_input(vid_input, temp_dir, "driving.mp4", "url")
                elif is_base64_string(vid_input):
                    # 优先检测 base64（因为 base64 字符串通常很长）
                    video_path = process_input(vid_input, temp_dir, "driving.mp4", "base64")
                else:
                    # 尝试作为路径检查
                    try:
                        if os.path.exists(vid_input):
                            video_path = process_input(vid_input, temp_dir, "driving.mp4", "path")
                        else:
                            # 如果路径不存在，尝试作为 base64 处理
                            video_path = process_input(vid_input, temp_dir, "driving.mp4", "base64")
                    except Exception:
                        # 如果路径检查出错，尝试作为 base64 处理
                        video_path = process_input(vid_input, temp_dir, "driving.mp4", "base64")
            else:
                raise Exception(f"driving_video 参数类型错误: {type(vid_input)}")
        else:
            raise Exception("缺少必需参数: driving_video")
        
        # 获取参数
        frame_interval = job_input.get("frame_interval", 1)
        do_align = job_input.get("do_align", True)
        alignmode = job_input.get("alignmode", "ref")
        h = job_input.get("height")
        w = job_input.get("width")
        face_change = job_input.get("face_change", True)
        head_change = job_input.get("head_change", False)
        without_face = job_input.get("without_face", False)
        new_h = job_input.get("inference_height")
        new_w = job_input.get("inference_width")
        ref_cfg = float(job_input.get("ref_cfg", 2.5))
        pose_cfg = float(job_input.get("pose_cfg", 1.5))
        prompt = job_input.get("prompt", "")
        
        # 生成视频
        output_path = os.path.join(temp_dir, "output.mp4")
        generate_single_video(
            reference_path=reference_path,
            video_path=video_path,
            output_path=output_path,
            frame_interval=frame_interval,
            do_align=do_align,
            alignmode=alignmode,
            h=h,
            w=w,
            face_change=face_change,
            head_change=head_change,
            without_face=without_face,
            new_h=new_h,
            new_w=new_w,
            ref_cfg=ref_cfg,
            pose_cfg=pose_cfg,
            prompt=prompt
        )
        
        # 上传结果
        logger.info("上传结果...")
        uploaded_url = rp_upload.upload_file(job["id"], output_path)
        
        # 清理临时文件
        shutil.rmtree(temp_dir, ignore_errors=True)
        
        return {
            "status": "success",
            "video_url": uploaded_url,
            "message": "视频生成成功"
        }
        
    except Exception as e:
        logger.error(f"❌ 处理失败: {e}", exc_info=True)
        # 清理临时文件
        shutil.rmtree(temp_dir, ignore_errors=True)
        return {
            "status": "error",
            "error": str(e)
        }

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})

