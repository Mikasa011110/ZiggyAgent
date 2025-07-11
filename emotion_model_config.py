#!/usr/bin/env python3
"""
表情识别模型配置文件
用于优化模型预测准确性和处理流程
"""

import torch
import torch.nn as nn
from torchvision import transforms
import cv2
import numpy as np
from PIL import Image
import os

class EmotionModelConfig:
    """表情识别模型配置类"""
    
    def __init__(self, model_path='ResNet50.pt'):
        self.model_path = model_path
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 情绪标签映射 - 根据RAF-DB数据集标准
        self.emotion_labels = {
            0: "surprise",
            1: "fear", 
            2: "disgust",
            4: "happiness",
            3: "sadness",
            5: "anger",
            6: "neutral"
        }

            # 原始情绪标签映射
            # 0: "surprise",
            # 1: "fear", 
            # 2: "disgust",
            # 3: "happiness",
            # 4: "sadness",
            # 5: "anger",
            # 6: "neutral
        
        # 优化的预处理流程
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 置信度阈值配置
        self.confidence_threshold = 0.4  # 提高置信度阈值
        self.min_face_size = 80  # 最小人脸尺寸
        
        # 加载模型
        self.load_model()
    
    def load_model(self):
        """加载和验证模型"""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
            
            # 加载模型
            self.model = torch.load(self.model_path, map_location=self.device)
            self.model.eval()
            
            # 验证模型
            self._validate_model()
            print(f"✅ 模型加载成功，使用设备: {self.device}")
            
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            self.model = None
    
    def _validate_model(self):
        """验证模型结构"""
        if self.model is None:
            raise ValueError("模型未加载")
        
        # 测试推理
        test_input = torch.randn(1, 3, 224, 224).to(self.device)
        with torch.no_grad():
            output = self.model(test_input)
        
        if output.shape[1] != 7:
            raise ValueError(f"模型输出类别数不正确: 期望7，实际{output.shape[1]}")
    
    def improve_face_detection(self, img):
        """改进的人脸检测"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 使用多个级联分类器
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
        
        # 检测正面人脸
        faces = face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(self.min_face_size, self.min_face_size)
        )
        
        # 如果没有检测到，尝试侧面人脸
        if len(faces) == 0:
            faces = profile_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=3, 
                minSize=(self.min_face_size, self.min_face_size)
            )
        
        # 如果仍然没有，使用更宽松的参数
        if len(faces) == 0:
            faces = face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.05, 
                minNeighbors=3, 
                minSize=(50, 50)
            )
        
        return faces
    
    def preprocess_face(self, face_img):
        """改进的人脸预处理"""
        # 确保人脸图像足够大
        h, w = face_img.shape[:2]
        if h < 80 or w < 80:
            # 放大到最小尺寸
            scale = max(80/h, 80/w)
            new_h, new_w = int(h * scale), int(w * scale)
            face_img = cv2.resize(face_img, (new_w, new_h))
        
        # 转换为PIL图像
        face_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
        
        # 应用预处理
        input_tensor = self.transform(face_pil).unsqueeze(0).to(self.device)
        return input_tensor
    
    def predict_emotion(self, img):
        """预测表情"""
        if self.model is None:
            return {
                'error': '模型未正确加载',
                'success': False
            }
        
        try:
            # 人脸检测
            faces = self.improve_face_detection(img)
            
            if len(faces) == 0:
                return {
                    'error': '未检测到人脸，请确保人脸清晰可见',
                    'success': False
                }
            
            # 选择最大的人脸
            faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
            x, y, w, h = faces[0]
            
            # 预处理
            face_img = img[y:y+h, x:x+w]
            input_tensor = self.preprocess_face(face_img)
            
            # 推理
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probs = torch.nn.functional.softmax(outputs, dim=1)[0]
                confidence, pred_idx = torch.max(probs, 0)
            
            # 获取所有情绪概率
            all_probs = probs.cpu().tolist()
            emotion_probs = {
                self.emotion_labels[i]: round(all_probs[i] * 100, 2) 
                for i in range(len(self.emotion_labels))
            }
            
            # 检查置信度
            if confidence.item() < self.confidence_threshold:
                return {
                    'emotion': 'uncertain',
                    'confidence': round(confidence.item() * 100, 2),
                    'message': f'置信度过低 ({confidence.item()*100:.1f}%)，无法准确识别',
                    'face_rect': {'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h)},
                    'all_emotions': emotion_probs,
                    'success': True
                }
            
            return {
                'emotion': self.emotion_labels[pred_idx.item()],
                'confidence': round(confidence.item() * 100, 2),
                'face_rect': {'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h)},
                'all_emotions': emotion_probs,
                'success': True
            }
            
        except Exception as e:
            return {
                'error': f'预测失败: {str(e)}',
                'success': False
            }
    
    def get_model_info(self):
        """获取模型信息"""
        if self.model is None:
            return {'error': '模型未加载'}
        
        total_params = sum(p.numel() for p in self.model.parameters())
        return {
            'model_path': self.model_path,
            'device': str(self.device),
            'total_params': total_params,
            'emotion_labels': self.emotion_labels,
            'confidence_threshold': self.confidence_threshold
        }

# 全局模型实例
emotion_model = None

def get_emotion_model():
    """获取全局模型实例"""
    global emotion_model
    if emotion_model is None:
        emotion_model = EmotionModelConfig()
    return emotion_model 