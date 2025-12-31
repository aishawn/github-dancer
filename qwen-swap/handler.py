import runpod
from runpod.serverless.utils import rp_upload
import os
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


# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    
def queue_prompt(prompt):
    url = f"http://{server_address}:8188/prompt"
    logger.info(f"Queueing prompt to: {url}")
    p = {"prompt": prompt, "client_id": client_id}
    data = json.dumps(p).encode('utf-8')
    req = urllib.request.Request(url, data=data)
    return json.loads(urllib.request.urlopen(req).read())

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
    prompt_id = queue_prompt(prompt)['prompt_id']
    output_images = {}
    while True:
        out = ws.recv()
        if isinstance(out, str):
            message = json.loads(out)
            if message['type'] == 'executing':
                data = message['data']
                if data['node'] is None and data['prompt_id'] == prompt_id:
                    break
        else:
            continue

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
    if not os.path.exists(workflow_path):
        raise FileNotFoundError(f"Workflow file not found: {workflow_path}")
    logger.info(f"Loading workflow from: {workflow_path}")
    with open(workflow_path, 'r') as file:
        workflow = json.load(file)
        # 新格式 (nodes 배열)을旧格式 (ComfyUI API 형식)으로 변환
        if "nodes" in workflow:
            # links 맵 생성: link_id -> [source_node_id, output_index]
            links_map = {}
            if "links" in workflow:
                for link in workflow["links"]:
                    link_id = link[0]
                    source_node_id = str(link[1])
                    source_output_index = link[2]
                    target_node_id = str(link[3])
                    target_input_index = link[4]
                    links_map[link_id] = [source_node_id, source_output_index]
            
            # nodes 배열을 API 형식으로 변환
            prompt = {}
            for node in workflow["nodes"]:
                node_id = str(node["id"])
                node_type = node.get("type", "")
                
                # 非执行节点跳过（这些节点不应该发送到 ComfyUI API）
                # 包括：注释节点（Note, MarkdownNote）、路由节点（Reroute）、逻辑节点、预览节点、原始值节点等
                non_executable_types = ["MarkdownNote", "Note", "Reroute", "GetNode", "SetNode", "PrimitiveNode", "PrimitiveStringMultiline", "SigmasPreview", "Sigmas Power"]
                if node_type in non_executable_types or (isinstance(node_type, str) and (any(node_type.startswith(t) for t in ["Note", "Markdown", "Primitive"]) or node_type.endswith("Preview") or "Power" in node_type)):
                    logger.info(f"Skipping non-executable node: {node_id} (type: {node_type})")
                    continue
                
                converted_node = {}
                widgets_values = node.get("widgets_values", [])
                
                # UnetLoaderGGUF를 UNETLoader로 변환
                if node_type == "UnetLoaderGGUF":
                    converted_node["class_type"] = "UNETLoader"
                    converted_inputs = {}
                    # GGUF 모델 파일명을 safetensors로 변경
                    if widgets_values and len(widgets_values) > 0:
                        # Qwen-Image-Edit-2509-Q8_0.gguf -> qwen_image_edit_2509_fp8_e4m3fn.safetensors
                        converted_inputs["unet_name"] = "qwen_image_edit_2509_fp8_e4m3fn.safetensors"
                        converted_inputs["weight_dtype"] = "default"
                    converted_node["inputs"] = converted_inputs
                else:
                    # inputs 배열을 inputs 객체로 변환
                    if "inputs" in node and isinstance(node["inputs"], list):
                        converted_inputs = {}
                        widget_index = 0
                        for input_item in node["inputs"]:
                            if isinstance(input_item, dict) and "name" in input_item:
                                input_name = input_item["name"]
                                if "link" in input_item and input_item["link"] is not None:
                                    # link가 있으면 links_map에서 찾아서 [node_id, output_index] 형식으로 변환
                                    link_id = input_item["link"]
                                    if link_id in links_map:
                                        converted_inputs[input_name] = links_map[link_id]
                                else:
                                    # link가 없으면 widgets_values에서 값을 가져옴
                                    # LoadImage의 image 입력은 widgets_values[0]에서 가져옴
                                    if node_type == "LoadImage" and input_name == "image":
                                        if widget_index < len(widgets_values):
                                            converted_inputs[input_name] = widgets_values[widget_index]
                                            widget_index += 1
                                    # 다른 노드 타입도 필요시 처리
                                    elif "widget" in input_item:
                                        if widget_index < len(widgets_values):
                                            converted_inputs[input_name] = widgets_values[widget_index]
                                            widget_index += 1
                        converted_node["inputs"] = converted_inputs
                    elif "inputs" in node:
                        converted_node["inputs"] = node["inputs"]
                    else:
                        converted_node["inputs"] = {}
                    
                    # type을 class_type으로 변환
                    if node_type:
                        converted_node["class_type"] = node_type
                
                # widgets_values는 API 형식에서도 필요할 수 있으므로 복사
                if widgets_values:
                    converted_node["widgets_values"] = widgets_values
                
                # 다른 필드 복사
                for key in ["properties", "title", "color", "bgcolor"]:
                    if key in node:
                        converted_node[key] = node[key]
                
                prompt[node_id] = converted_node
            
            logger.info(f"Loaded workflow with {len(prompt)} executable nodes")
            return prompt
        return workflow

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
    # Workflow 선택
    # ------------------------------
    workflow_type = job_input.get("workflow_type", "default")
    
    if workflow_type == "head_swap_v3":
        # Head Swap V3 workflow 사용
        if not image2_path:
            return {"error": "Head Swap V3 workflow requires two images (body and face)"}
        workflow_path = "/Head Swap V3 Simple Workflow (With Lightining LoRA) .json"
        prompt = load_workflow(workflow_path)
        
        # Head Swap V3 workflow 노드 설정
        # 验证必需的节点是否存在
        required_nodes = ["343", "349", "348", "395", "406", "345"]
        missing_nodes = [node_id for node_id in required_nodes if node_id not in prompt]
        if missing_nodes:
            raise ValueError(f"Required nodes not found in workflow: {missing_nodes}")
        
        # Node 343: Body Reference (image1)
        prompt["343"]["inputs"]["image"] = image1_path
        # Node 349: Face Reference (image2)
        prompt["349"]["inputs"]["image"] = image2_path
        # Node 348: TextEncodeQwenImageEditPlus (prompt)
        prompt["348"]["widgets_values"][0] = job_input.get("prompt", "head_swap: start with Picture 1 as the base image, keeping its lighting, environment, and background. remove the head from Picture 1 completely and replace it with the head from Picture 2. ensure the head and body have correct anatomical proportions, and blend the skin tones, shadows, and lighting naturally so the final result appears as one coherent, realistic person.")
        # Node 395: SamplerCustom (seed)
        prompt["395"]["widgets_values"][1] = job_input.get("seed", 43)
        # Node 406: ImageResizeKJv2 (width, height for body image)
        prompt["406"]["widgets_values"][0] = job_input.get("width", 1328)
        prompt["406"]["widgets_values"][1] = job_input.get("height", 1328)
        # Node 345: EmptySD3LatentImage (width, height)
        prompt["345"]["widgets_values"][0] = job_input.get("width", 1024)
        prompt["345"]["widgets_values"][1] = job_input.get("height", 1024)
    else:
        # 기본 workflow 사용
        if image2_path:
            workflow_path = "/qwen_image_edit_2.json"
        else:
            workflow_path = "/qwen_image_edit_1.json"

        prompt = load_workflow(workflow_path)

        prompt["78"]["inputs"]["image"] = image1_path
        if image2_path:
            prompt["123"]["inputs"]["image"] = image2_path

        prompt["111"]["inputs"]["prompt"] = job_input.get("prompt", "")

        prompt["3"]["inputs"]["seed"] = job_input.get("seed", 954812286882415)
        prompt["128"]["inputs"]["value"] = job_input.get("width", 720)
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