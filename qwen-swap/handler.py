import runpod
from runpod.serverless.utils import rp_upload
import os
import sys
import websocket
import base64
import json
import uuid
import logging
import urllib.request
import urllib.parse
import binascii # Base64 에러 처리를 위해 import
import subprocess
import time

# 로깅 설정（必须在导入 ComfyUI 之前）
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== 架构说明 ====================
# 
# 正确方案：使用 API Prompt 模板（ComfyUI 官方推荐）
# 
# Step 1: 在 UI 里调通 workflow
# Step 2: 点 Save → 导出 API 格式（Save (API) / Copy API）
# Step 3: 生产环境只做「参数注入」
# 
# 优势：
# - 转换逻辑：0（不需要转换）
# - GraphBuilder 依赖：不需要
# - Custom node 兼容：UI 已验证
# - Debug 成本：直观
# - 可维护性：高
# - 符合 ComfyUI 设计：顺着来
#

# CUDA 검사 및 설정
def check_cuda_availability():
    """CUDA 사용 가능 여부를 확인하고 환경 변수를 설정합니다."""
    try:
        import torch
        if torch.cuda.is_available():
            logger.info("✅ CUDA is available and working")
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
            return True
        else:
            logger.error("❌ CUDA is not available")
            raise RuntimeError("CUDA is required but not available")
    except Exception as e:
        logger.error(f"❌ CUDA check failed: {e}")
        raise RuntimeError(f"CUDA initialization failed: {e}")

# CUDA 검사 실행
try:
    cuda_available = check_cuda_availability()
    if not cuda_available:
        raise RuntimeError("CUDA is not available")
except Exception as e:
    logger.error(f"Fatal error: {e}")
    logger.error("Exiting due to CUDA requirements not met")
    exit(1)



server_address = os.getenv('SERVER_ADDRESS', '127.0.0.1')
client_id = str(uuid.uuid4())
def save_data_if_base64(data_input, temp_dir, output_filename):
    """
    입력 데이터가 Base64 문자열인지 확인하고, 맞다면 파일로 저장 후 경로를 반환합니다.
    만약 일반 경로 문자열이라면 그대로 반환합니다.
    """
    # 입력값이 문자열이 아니면 그대로 반환
    if not isinstance(data_input, str):
        return data_input

    try:
        # Base64 문자열은 디코딩을 시도하면 성공합니다.
        decoded_data = base64.b64decode(data_input)
        
        # 디렉토리가 존재하지 않으면 생성
        os.makedirs(temp_dir, exist_ok=True)
        
        # 디코딩에 성공하면, 임시 파일로 저장합니다.
        file_path = os.path.abspath(os.path.join(temp_dir, output_filename))
        with open(file_path, 'wb') as f: # 바이너리 쓰기 모드('wb')로 저장
            f.write(decoded_data)
        
        # 저장된 파일의 경로를 반환합니다.
        print(f"✅ Base64 입력을 '{file_path}' 파일로 저장했습니다.")
        return file_path

    except (binascii.Error, ValueError):
        # 디코딩에 실패하면, 일반 경로로 간주하고 원래 값을 그대로 반환합니다.
        print(f"➡️ '{data_input}'은(는) 파일 경로로 처리합니다.")
        return data_input
    
def queue_prompt_via_http(prompt):
    """
    通过 HTTP /prompt API 发送 prompt（回退方案）
    
    注意：虽然使用 HTTP API，但仍然遵循"方案一"原则：
    - 只做参数注入，不转换 workflow
    - 使用 API Prompt 模板
    """
    url = f"http://{server_address}:8188/prompt"
    logger.info(f"Queueing prompt via HTTP API: {url}")
    
    p = {"prompt": prompt, "client_id": client_id}
    data = json.dumps(p).encode('utf-8')
    req = urllib.request.Request(url, data=data)
    req.add_header('Content-Type', 'application/json')
    
    try:
        response = urllib.request.urlopen(req)
        result = json.loads(response.read())
        logger.info(f"✅ Prompt queued via HTTP API, prompt_id: {result.get('prompt_id')}")
        return result
    except urllib.error.HTTPError as e:
        error_body = ""
        try:
            error_body = e.read().decode('utf-8')
        except:
            error_body = str(e)
        
        logger.error(f"HTTP Error {e.code}: {e.reason}")
        logger.error(f"Error response body: {error_body}")
        
        # 尝试解析错误 JSON
        try:
            error_json = json.loads(error_body)
            logger.error(f"Error JSON: {json.dumps(error_json, indent=2)}")
        except:
            pass
        
        raise Exception(f"ComfyUI API error ({e.code}): {error_body[:500]}") from e

def validate_and_fix_prompt(prompt):
    """
    验证和修复 API Prompt（节点验证、无效节点移除、调度器修复等）
    
    重要说明：
    - prompt 必须是 API 格式：{node_id: {class_type, inputs}}
    - 不再支持 UI 格式转换
    - 使用方式：在 ComfyUI UI 中导出 API 格式，然后只做参数注入
    """
    # 调度器名称映射：将无效的调度器名称映射到有效的名称
    scheduler_name_mapping = {
        "beta57": "normal",  # beta57 可能不被支持，使用 normal 替代
    }
    # 验证 prompt 格式（必须是 API 格式）
    if not isinstance(prompt, dict):
        raise ValueError(f"Prompt must be a dict (API format), got {type(prompt)}")
    
    # 检查是否是 UI 格式（有 nodes 数组）
    if "nodes" in prompt:
        raise ValueError(
            "UI format workflow is not supported. "
            "Please export API format from ComfyUI UI (Save → API format), "
            "then use parameter injection only."
        )
    
    # 验证所有节点都有 class_type，并处理无效节点
    nodes_to_remove = []
    
    # 已知不支持的节点类型（可能是自定义节点未安装，或导出问题）
    # 这些节点类型在 ComfyUI 中可能不存在，需要移除
    unsupported_node_types = [
        "Sigmas Power",      # 可能是自定义节点，当前环境未安装
        "SigmasPreview",     # 预览节点，用于调试，输出未被使用，可以安全移除
    ]
    
    for node_id, node_data in prompt.items():
        # 检查节点 ID 是否有效
        if not node_id or node_id == "#id" or not isinstance(node_id, str):
            logger.warning(f"Invalid node ID found: {node_id}, removing from prompt")
            nodes_to_remove.append(node_id)
            continue
        
        if not isinstance(node_data, dict):
            logger.warning(f"Node {node_id} is not a dict, removing from prompt")
            nodes_to_remove.append(node_id)
            continue
        
        # 检查 class_type
        if "class_type" not in node_data:
            # 检查是否有 UNKNOWN 字段（导出问题）
            if "inputs" in node_data and "UNKNOWN" in node_data.get("inputs", {}):
                unknown_value = node_data["inputs"]["UNKNOWN"]
                logger.warning(f"Node {node_id} has UNKNOWN field: {unknown_value}")
                
                # 尝试修复：如果是 GGUF 模型文件，转换为 UNETLoader
                if isinstance(unknown_value, str) and unknown_value.endswith(".gguf"):
                    logger.info(f"Attempting to fix node {node_id}: converting UNKNOWN GGUF to UNETLoader")
                    # 修复节点：添加 class_type 和正确的 inputs
                    node_data["class_type"] = "UNETLoader"
                    # 将 GGUF 文件名转换为 safetensors 文件名
                    # Qwen-Image-Edit-2509-Q8_0.gguf -> qwen_image_edit_2509_fp8_e4m3fn.safetensors
                    if "Qwen-Image-Edit-2509" in unknown_value:
                        node_data["inputs"] = {
                            "unet_name": "qwen_image_edit_2509_fp8_e4m3fn.safetensors",
                            "weight_dtype": "default"
                        }
                        logger.info(f"Fixed node {node_id}: UNETLoader with unet_name=qwen_image_edit_2509_fp8_e4m3fn.safetensors")
                    else:
                        # 其他 GGUF 文件，尝试通用转换
                        safetensors_name = unknown_value.replace(".gguf", ".safetensors").replace("-", "_").lower()
                        node_data["inputs"] = {
                            "unet_name": safetensors_name,
                            "weight_dtype": "default"
                        }
                        logger.warning(f"Fixed node {node_id} with guessed safetensors name: {safetensors_name}")
                else:
                    # 无法修复，标记为删除
                    logger.warning(f"Node {node_id} has UNKNOWN field but cannot be auto-fixed - removing from prompt. "
                                 f"This node may not be needed or needs to be fixed in the exported workflow.")
                    nodes_to_remove.append(node_id)
            else:
                raise ValueError(f"Node {node_id} missing required 'class_type' property")
        
        if "inputs" not in node_data:
            logger.warning(f"Node {node_id} missing 'inputs' property, removing from prompt")
            nodes_to_remove.append(node_id)
            continue
        
        # 检查是否是不支持的节点类型（可能是自定义节点未安装）
        class_type = node_data.get("class_type", "")
        if class_type in unsupported_node_types:
            logger.warning(f"Node {node_id} has unsupported class_type '{class_type}' (may not be installed), removing from prompt")
            nodes_to_remove.append(node_id)
            continue
    
    # 在删除节点前，先修复引用（需要访问被删除节点的信息）
    # 对于某些节点类型（如 Sigmas Power），可以尝试将引用改为引用其输入节点
    nodes_to_fix = {}  # {node_id: {input_key: new_reference}}
    for node_id, node_data in prompt.items():
        if "inputs" in node_data:
            for input_key, input_value in node_data["inputs"].items():
                # 检查是否是节点引用 [node_id, output_index]
                if isinstance(input_value, list) and len(input_value) >= 1:
                    referenced_node_id = str(input_value[0])
                    if referenced_node_id in nodes_to_remove:
                        removed_node = prompt.get(referenced_node_id)  # 在删除前获取
                        if removed_node and removed_node.get("class_type") == "Sigmas Power":
                            # Sigmas Power 节点：将引用改为引用其 sigmas 输入
                            sigmas_input = removed_node.get("inputs", {}).get("sigmas")
                            if isinstance(sigmas_input, list) and len(sigmas_input) >= 1:
                                # sigmas 输入通常是 [source_node_id, output_index]
                                source_node_id = str(sigmas_input[0])
                                source_output_index = sigmas_input[1] if len(sigmas_input) > 1 else 0
                                # 如果源节点还在 prompt 中，则替换引用
                                if source_node_id in prompt and source_node_id not in nodes_to_remove:
                                    if node_id not in nodes_to_fix:
                                        nodes_to_fix[node_id] = {}
                                    nodes_to_fix[node_id][input_key] = [source_node_id, source_output_index]
                                    logger.info(f"Will fix node {node_id} input '{input_key}': "
                                              f"change reference from removed node {referenced_node_id} "
                                              f"to {source_node_id} (Sigmas Power bypass)")
                                    continue
                        logger.warning(f"Node {node_id} input '{input_key}' references removed node {referenced_node_id}. "
                                     f"This may cause execution errors.")
    
    # 应用修复
    for node_id, fixes in nodes_to_fix.items():
        if node_id in prompt:
            for input_key, new_reference in fixes.items():
                prompt[node_id]["inputs"][input_key] = new_reference
    
    # 移除无效节点
    for node_id in nodes_to_remove:
        del prompt[node_id]
        logger.info(f"Removed invalid node {node_id} from prompt")
    
    logger.info(f"✅ Validated API format prompt with {len(prompt)} nodes (removed {len(nodes_to_remove)} invalid nodes)")
    
    # 修复无效的调度器和采样器名称
    scheduler_name_mapping = {
        "beta57": "normal",  # beta57 可能不被支持，使用 normal 替代
    }
    sampler_name_mapping = {
        "res_2s": "euler",  # res_2s 可能不被支持，使用 euler 替代
    }
    
    for node_id, node_data in prompt.items():
        if isinstance(node_data, dict) and "inputs" in node_data:
            inputs = node_data["inputs"]
            
            # 修复调度器名称
            if "scheduler" in inputs:
                scheduler_name = inputs["scheduler"]
                if scheduler_name in scheduler_name_mapping:
                    new_scheduler = scheduler_name_mapping[scheduler_name]
                    inputs["scheduler"] = new_scheduler
                    logger.info(f"✅ Fixed scheduler name in node {node_id}: {scheduler_name} -> {new_scheduler}")
            
            # 修复采样器名称（KSamplerSelect 节点）
            if "sampler_name" in inputs:
                sampler_name = inputs["sampler_name"]
                if sampler_name in sampler_name_mapping:
                    new_sampler = sampler_name_mapping[sampler_name]
                    inputs["sampler_name"] = new_sampler
                    logger.info(f"✅ Fixed sampler name in node {node_id}: {sampler_name} -> {new_sampler}")

def get_image(filename, subfolder, folder_type):
    url = f"http://{server_address}:8188/view"
    logger.info(f"Getting image from: {url}")
    data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
    url_values = urllib.parse.urlencode(data)
    with urllib.request.urlopen(f"{url}?{url_values}") as response:
        return response.read()

def get_history(prompt_id):
    url = f"http://{server_address}:8188/history/{prompt_id}"
    logger.info(f"Getting history from: {url}")
    with urllib.request.urlopen(url) as response:
        return json.loads(response.read())

def get_images(ws, prompt):
    """
    执行 API Prompt 并获取生成的图片
    使用 HTTP API 发送 prompt，WebSocket 仅用于监听执行状态
    """
    # 先验证和修复 prompt
    validate_and_fix_prompt(prompt)
    
    # 使用 HTTP API 发送 prompt（WebSocket 发送不被支持）
    prompt_id = queue_prompt_via_http(prompt)['prompt_id']
    output_images = {}
    max_wait = 300  # 最多等待 5 分钟
    start_time = time.time()
    
    while time.time() - start_time < max_wait:
        try:
            ws.settimeout(1.0)
            out = ws.recv()
            if isinstance(out, str):
                message = json.loads(out)
                msg_type = message.get('type')
                
                if msg_type == 'executing':
                    data = message.get('data', {})
                    if data.get('node') is None and data.get('prompt_id') == prompt_id:
                        logger.info("✅ Workflow execution completed")
                        break
                elif msg_type == 'execution_error':
                    error_data = message.get('data', {})
                    error_msg = error_data.get('message', 'Unknown error')
                    error_node = error_data.get('node_id', 'unknown')
                    error_type = error_data.get('node_type', 'unknown')
                    logger.error(f"❌ Execution error at node {error_node} ({error_type}): {error_msg}")
                    raise Exception(f"ComfyUI execution error (node {error_node}): {error_msg}")
                elif msg_type == 'execution_cached':
                    # 执行被缓存，继续等待完成
                    logger.debug(f"Execution cached for prompt {prompt_id}")
                elif msg_type == 'status':
                    # 状态更新，继续等待
                    logger.debug(f"Status update: {message.get('data', {})}")
                elif msg_type == 'progress':
                    # 进度更新，继续等待
                    logger.debug(f"Progress update: {message.get('data', {})}")
        except websocket.WebSocketTimeoutException:
            # 超时继续等待
            elapsed = time.time() - start_time
            if int(elapsed) % 10 == 0:  # 每10秒记录一次
                logger.info(f"Still waiting for execution to complete... ({int(elapsed)}s elapsed)")
            continue
        except Exception as e:
            if "timeout" in str(e).lower():
                continue
            # 如果是执行错误，已经在上面的 execution_error 处理中抛出
            raise
    
    if time.time() - start_time >= max_wait:
        raise Exception(f"Workflow execution timeout after {max_wait}s")

    history = get_history(prompt_id)[prompt_id]
    for node_id in history['outputs']:
        node_output = history['outputs'][node_id]
        images_output = []
        if 'images' in node_output:
            for image in node_output['images']:
                image_data = get_image(image['filename'], image['subfolder'], image['type'])
                # bytes 객체를 base64로 인코딩하여 JSON 직렬화 가능하게 변환
                if isinstance(image_data, bytes):
                    import base64
                    image_data = base64.b64encode(image_data).decode('utf-8')
                images_output.append(image_data)
        output_images[node_id] = images_output

    return output_images

def load_workflow(workflow_path):
    """
    加载 API Prompt 模板文件
    
    重要：只支持 API 格式，不支持 UI 格式
    - 在 ComfyUI UI 中：Save → 导出 API 格式（Save (API) / Copy API）
    - 得到的是：{node_id: {class_type, inputs}} 格式
    
    如果遇到 UI 格式，会抛出错误提示用户导出 API 格式
    """
    if not os.path.exists(workflow_path):
        raise FileNotFoundError(f"Workflow file not found: {workflow_path}")
    logger.info(f"Loading API prompt template from: {workflow_path}")
    
    with open(workflow_path, 'r', encoding='utf-8') as file:
        prompt = json.load(file)
    
    # 验证格式：必须是 API 格式，不能是 UI 格式
    if "nodes" in prompt:
        raise ValueError(
            f"UI format workflow is not supported. "
            f"Please export API format from ComfyUI UI:\n"
            f"  1. Open workflow in ComfyUI UI\n"
            f"  2. Click 'Save' → Select 'Save (API)' or 'Copy API'\n"
            f"  3. Save the API format JSON file\n"
            f"  4. Use that file as the workflow template"
        )
    
    # 验证 API 格式
    if not isinstance(prompt, dict):
        raise ValueError(f"Prompt must be a dict (API format), got {type(prompt)}")
    
    # 验证所有节点都有必需的字段
    for node_id, node_data in prompt.items():
        if not isinstance(node_data, dict):
            raise ValueError(f"Node {node_id} must be a dict, got {type(node_data)}")
        if "class_type" not in node_data:
            # 某些节点可能没有 class_type（如未正确导出的节点）
            # 检查是否有 UNKNOWN 字段（这通常是导出问题）
            if "inputs" in node_data and "UNKNOWN" in node_data.get("inputs", {}):
                logger.warning(f"Node {node_id} has UNKNOWN field - this may be an export issue. "
                             f"Node data: {json.dumps(node_data, indent=2)}")
                # 不抛出错误，但记录警告，允许继续执行
            else:
                raise ValueError(f"Node {node_id} missing required 'class_type' property")
        if "inputs" not in node_data:
            raise ValueError(f"Node {node_id} missing required 'inputs' property")
    
    logger.info(f"✅ Loaded API prompt template with {len(prompt)} nodes")
    return prompt

# ------------------------------
# 입력 처리 유틸 (path/url/base64)
# ------------------------------
def process_input(input_data, temp_dir, output_filename, input_type):
    """입력 데이터를 처리하여 파일 경로를 반환하는 함수
    - input_type: "path" | "url" | "base64"
    """
    if input_type == "path":
        logger.info(f"📁 경로 입력 처리: {input_data}")
        return input_data
    elif input_type == "url":
        logger.info(f"🌐 URL 입력 처리: {input_data}")
        os.makedirs(temp_dir, exist_ok=True)
        file_path = os.path.abspath(os.path.join(temp_dir, output_filename))
        return download_file_from_url(input_data, file_path)
    elif input_type == "base64":
        logger.info("🔢 Base64 입력 처리")
        return save_base64_to_file(input_data, temp_dir, output_filename)
    else:
        raise Exception(f"지원하지 않는 입력 타입: {input_type}")

def download_file_from_url(url, output_path):
    """URL에서 파일을 다운로드하는 함수"""
    try:
        result = subprocess.run([
            'wget', '-O', output_path, '--no-verbose', url
        ], capture_output=True, text=True)
        if result.returncode == 0:
            logger.info(f"✅ URL에서 파일을 성공적으로 다운로드했습니다: {url} -> {output_path}")
            return output_path
        else:
            logger.error(f"❌ wget 다운로드 실패: {result.stderr}")
            raise Exception(f"URL 다운로드 실패: {result.stderr}")
    except subprocess.TimeoutExpired:
        logger.error("❌ 다운로드 시간 초과")
        raise Exception("다운로드 시간 초과")
    except Exception as e:
        logger.error(f"❌ 다운로드 중 오류 발생: {e}")
        raise Exception(f"다운로드 중 오류 발생: {e}")

def save_base64_to_file(base64_data, temp_dir, output_filename):
    """Base64 데이터를 파일로 저장하는 함수"""
    try:
        decoded_data = base64.b64decode(base64_data)
        os.makedirs(temp_dir, exist_ok=True)
        file_path = os.path.abspath(os.path.join(temp_dir, output_filename))
        with open(file_path, 'wb') as f:
            f.write(decoded_data)
        logger.info(f"✅ Base64 입력을 '{file_path}' 파일로 저장했습니다.")
        return file_path
    except (binascii.Error, ValueError) as e:
        logger.error(f"❌ Base64 디코딩 실패: {e}")
        raise Exception(f"Base64 디코딩 실패: {e}")

def handler(job):
    job_input = job.get("input", {})

    logger.info(f"Received job input: {job_input}")
    task_id = f"task_{uuid.uuid4()}"

    # ------------------------------
    # 이미지 입력 수집 (1개 또는 2개)
    # 지원 키: image_path | image_url | image_base64
    #         image_path_2 | image_url_2 | image_base64_2
    # ------------------------------
    image1_path = None
    image2_path = None

    if "image_path" in job_input:
        image1_path = process_input(job_input["image_path"], task_id, "input_image_1.jpg", "path")
    elif "image_url" in job_input:
        image1_path = process_input(job_input["image_url"], task_id, "input_image_1.jpg", "url")
    elif "image_base64" in job_input:
        image1_path = process_input(job_input["image_base64"], task_id, "input_image_1.jpg", "base64")

    if "image_path_2" in job_input:
        image2_path = process_input(job_input["image_path_2"], task_id, "input_image_2.jpg", "path")
    elif "image_url_2" in job_input:
        image2_path = process_input(job_input["image_url_2"], task_id, "input_image_2.jpg", "url")
    elif "image_base64_2" in job_input:
        image2_path = process_input(job_input["image_base64_2"], task_id, "input_image_2.jpg", "base64")

    # ------------------------------
    # Workflow 선택 및加载 API Prompt 模板
    # ------------------------------
    workflow_type = job_input.get("workflow_type", "default")
    
    if workflow_type == "head_swap_v3":
        # Head Swap V3 workflow 사용
        if not image2_path:
            return {"error": "Head Swap V3 workflow requires two images (body and face)"}
        # 使用 API 格式的 workflow 文件
        workflow_path = "/Head_Swap_V3__api.json"
    else:
        # 기본 workflow 사용
        if image2_path:
            workflow_path = "/qwen_image_edit_2.json"
        else:
            workflow_path = "/qwen_image_edit_1.json"
    
    # 加载 API Prompt 模板（必须是 API 格式）
    prompt = load_workflow(workflow_path)
    
    logger.info("Head_Swap_V3__api")
    # 使用 deepcopy 避免修改模板
    import copy
    prompt = copy.deepcopy(prompt)
    
    # ------------------------------
    # 修复模型路径和节点（处理 Dockerfile 中的实际文件路径 + UNKNOWN 节点）
    # ------------------------------
    def fix_model_paths_and_nodes(prompt_data):
        """
        修复模型路径和节点问题：
        1. 修复 LoRA 路径映射（workflow 路径 -> Dockerfile 实际路径）
        2. 修复 UNKNOWN 节点（GGUF -> UNETLoader）
        3. 修复无效的调度器名称（beta57 -> normal）
        """
        # 调度器名称映射：将无效的调度器名称映射到有效的名称
        scheduler_name_mapping = {
            "beta57": "normal",  # beta57 可能不被支持，使用 normal 替代
        }
        
        for node_id, node_data in prompt_data.items():
            if not isinstance(node_data, dict):
                continue
            
            # 1. 修复 UNKNOWN 节点（在 queue_prompt_via_websocket 之前修复）
            if "class_type" not in node_data:
                if "inputs" in node_data and "UNKNOWN" in node_data.get("inputs", {}):
                    unknown_value = node_data["inputs"]["UNKNOWN"]
                    logger.warning(f"Node {node_id} has UNKNOWN field: {unknown_value}")
                    
                    # 尝试修复：如果是 GGUF 模型文件，转换为 UNETLoader
                    if isinstance(unknown_value, str) and unknown_value.endswith(".gguf"):
                        logger.info(f"Fixing node {node_id}: converting UNKNOWN GGUF to UNETLoader")
                        node_data["class_type"] = "UNETLoader"
                        # 将 GGUF 文件名转换为 safetensors 文件名
                        if "Qwen-Image-Edit-2509" in unknown_value:
                            node_data["inputs"] = {
                                "unet_name": "qwen_image_edit_2509_fp8_e4m3fn.safetensors",
                                "weight_dtype": "default"
                            }
                            logger.info(f"✅ Fixed node {node_id}: UNETLoader with unet_name=qwen_image_edit_2509_fp8_e4m3fn.safetensors")
                        else:
                            # 其他 GGUF 文件，尝试通用转换
                            safetensors_name = unknown_value.replace(".gguf", ".safetensors").replace("-", "_").lower()
                            node_data["inputs"] = {
                                "unet_name": safetensors_name,
                                "weight_dtype": "default"
                            }
                            logger.warning(f"Fixed node {node_id} with guessed safetensors name: {safetensors_name}")
            
            # 2. 修复 LoRA 路径映射
            if "inputs" in node_data:
                inputs = node_data["inputs"]
                
                # 修复 LoRA 路径
                if "lora_name" in inputs:
                    lora_name = inputs["lora_name"]
                    # 映射：qwen/Qwen-Image-Edit-2509-Lightning-4steps-V1.0-fp32.safetensors
                    #   -> Qwen-Image-Lightning-4steps-V1.0.safetensors
                    if lora_name == "qwen/Qwen-Image-Edit-2509-Lightning-4steps-V1.0-fp32.safetensors":
                        inputs["lora_name"] = "Qwen-Image-Lightning-4steps-V1.0.safetensors"
                        logger.info(f"✅ Fixed LoRA path in node {node_id}: {lora_name} -> Qwen-Image-Lightning-4steps-V1.0.safetensors")
                
                # 3. 修复无效的调度器名称
                if "scheduler" in inputs:
                    scheduler_name = inputs["scheduler"]
                    if scheduler_name in scheduler_name_mapping:
                        new_scheduler = scheduler_name_mapping[scheduler_name]
                        inputs["scheduler"] = new_scheduler
                        logger.info(f"✅ Fixed scheduler name in node {node_id}: {scheduler_name} -> {new_scheduler}")
    
    # 修复模型路径和节点
    fix_model_paths_and_nodes(prompt)
    
    # ------------------------------
    # 参数注入（只修改 API Prompt 的 inputs）
    # ------------------------------
    if workflow_type == "head_swap_v3":
        # Head Swap V3 workflow 的参数注入
        # 根据导出的 API Prompt 模板进行参数注入
        
        # 343: LoadImage (Body Reference)
        if "343" in prompt and image1_path:
            prompt["343"]["inputs"]["image"] = image1_path
        
        # 349: LoadImage (Face Reference)
        if "349" in prompt and image2_path:
            prompt["349"]["inputs"]["image"] = image2_path
        
        # 348: TextEncodeQwenImageEditPlus (prompt)
        if "348" in prompt:
            prompt["348"]["inputs"]["prompt"] = job_input.get(
                "prompt", 
                "head_swap: start with Picture 1 as the base image, keeping its lighting, environment, and background. remove the head from Picture 1 completely and replace it with the head from Picture 2. ensure the head and body have correct anatomical proportions, and blend the skin tones, shadows, and lighting naturally so the final result appears as one coherent, realistic person."
            )
        
        # 395: SamplerCustom (seed)
        if "395" in prompt:
            prompt["395"]["inputs"]["noise_seed"] = job_input.get("seed", 43)
        
        # 406: ImageResizeKJv2 (Body image resize)
        # 注意：节点 345 (EmptySD3LatentImage) 的 width/height 是从节点 406 连接的
        # 所以只需要修改 406 的 width/height，345 会自动使用
        if "406" in prompt:
            width = job_input.get("width", 1328)
            height = job_input.get("height", 1328)
            prompt["406"]["inputs"]["width"] = width
            prompt["406"]["inputs"]["height"] = height
        
        # 405: ImageResizeKJv2 (Face image resize)
        # 如果需要调整 Face 图片的尺寸，也可以修改这个节点
        if "405" in prompt:
            # 默认使用和 Body 图片相同的尺寸
            width = job_input.get("width", 1328)
            height = job_input.get("height", 1328)
            prompt["405"]["inputs"]["width"] = width
            prompt["405"]["inputs"]["height"] = height
        
        # 注意：节点 345 (EmptySD3LatentImage) 的 width/height 是从节点 406 连接的
        # 格式：["406", 1] 和 ["406", 2]
        # 所以不需要直接修改 345 的 width/height，它们会自动从 406 获取
    else:
        # 默认 workflow 的参数注入
        # 注意：这些 node_id 需要根据实际导出的 API Prompt 调整
        if "78" in prompt and image1_path:
            prompt["78"]["inputs"]["image"] = image1_path
        if "123" in prompt and image2_path:
            prompt["123"]["inputs"]["image"] = image2_path
        if "111" in prompt:
            prompt["111"]["inputs"]["prompt"] = job_input.get("prompt", "")
        if "3" in prompt:
            prompt["3"]["inputs"]["seed"] = job_input.get("seed", 954812286882415)
        if "128" in prompt:
            prompt["128"]["inputs"]["value"] = job_input.get("width", 720)
        if "129" in prompt:
            prompt["129"]["inputs"]["value"] = job_input.get("height", 1280)

    ws_url = f"ws://{server_address}:8188/ws?clientId={client_id}"
    logger.info(f"Connecting to WebSocket: {ws_url}")
    
    # 먼저 HTTP 연결이 가능한지 확인
    http_url = f"http://{server_address}:8188/"
    logger.info(f"Checking HTTP connection to: {http_url}")
    
    # HTTP 연결 확인 (최대 1분)
    max_http_attempts = 180
    for http_attempt in range(max_http_attempts):
        try:
            import urllib.request
            response = urllib.request.urlopen(http_url, timeout=5)
            logger.info(f"HTTP 연결 성공 (시도 {http_attempt+1})")
            break
        except Exception as e:
            logger.warning(f"HTTP 연결 실패 (시도 {http_attempt+1}/{max_http_attempts}): {e}")
            if http_attempt == max_http_attempts - 1:
                raise Exception("ComfyUI 서버에 연결할 수 없습니다. 서버가 실행 중인지 확인하세요.")
            time.sleep(1)
    
    ws = websocket.WebSocket()
    # 웹소켓 연결 시도 (최대 3분)
    max_attempts = int(180/5)  # 3분 (1초에 한 번씩 시도)
    for attempt in range(max_attempts):
        try:
            ws.connect(ws_url)
            logger.info(f"웹소켓 연결 성공 (시도 {attempt+1})")
            break
        except Exception as e:
            logger.warning(f"웹소켓 연결 실패 (시도 {attempt+1}/{max_attempts}): {e}")
            if attempt == max_attempts - 1:
                raise Exception("웹소켓 연결 시간 초과 (3분)")
            time.sleep(5)
    images = get_images(ws, prompt)
    ws.close()

    # 이미지가 없는 경우 처리
    if not images:
        return {"error": "이미지를 생성할 수 없습니다."}
    
    # 첫 번째 이미지 반환
    for node_id in images:
        if images[node_id]:
            return {"image": images[node_id][0]}
    
    return {"error": "이미지를 찾을 수 없습니다."}

runpod.serverless.start({"handler": handler})