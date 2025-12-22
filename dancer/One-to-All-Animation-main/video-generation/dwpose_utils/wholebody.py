import cv2
import numpy as np
import os

import onnxruntime as ort
from .onnxdet import inference_detector
from .onnxpose import inference_pose
import torch

class Wholebody:
    def __init__(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # device = 'cuda'
        providers = ['CPUExecutionProvider'
                 ] if device == 'cpu' else ['CUDAExecutionProvider']
        # 获取基于文件位置的绝对路径
        # wholebody.py 在 video-generation/dwpose_utils/ 目录下
        # 需要向上两级到 BASE_DIR，然后进入 pretrained_models/DWPose/
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.dirname(os.path.dirname(current_file_dir))
        dwpose_dir = os.path.join(base_dir, 'pretrained_models', 'DWPose')
        onnx_det = os.path.join(dwpose_dir, 'yolox_l.onnx')
        onnx_pose = os.path.join(dwpose_dir, 'dw-ll_ucoco_384.onnx')

        self.session_det = ort.InferenceSession(path_or_bytes=onnx_det, providers=providers)
        self.session_pose = ort.InferenceSession(path_or_bytes=onnx_pose, providers=providers)
    
    def __call__(self, oriImg):
        det_result = inference_detector(self.session_det, oriImg)
        keypoints, scores = inference_pose(self.session_pose, det_result, oriImg)

        keypoints_info = np.concatenate(
            (keypoints, scores[..., None]), axis=-1)
        # compute neck joint
        neck = np.mean(keypoints_info[:, [5, 6]], axis=1)
        # neck score when visualizing pred
        neck[:, 2:4] = np.logical_and(
            keypoints_info[:, 5, 2:4] > 0.3,
            keypoints_info[:, 6, 2:4] > 0.3).astype(int)
        new_keypoints_info = np.insert(
            keypoints_info, 17, neck, axis=1)
        mmpose_idx = [
            17, 6, 8, 10, 7, 9, 12, 14, 16, 13, 15, 2, 1, 4, 3
        ]
        openpose_idx = [
            1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17
        ]
        new_keypoints_info[:, openpose_idx] = \
            new_keypoints_info[:, mmpose_idx]
        keypoints_info = new_keypoints_info

        keypoints, scores = keypoints_info[
            ..., :2], keypoints_info[..., 2]
        
        return keypoints, scores



