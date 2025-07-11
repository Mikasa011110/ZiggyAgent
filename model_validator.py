#!/usr/bin/env python3
"""
模型验证脚本 - 用于检查表情识别模型是否正确加载和配置
"""

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import os

def validate_model():
    """验证模型是否正确加载和配置"""
    
    print("=== 模型验证开始 ===\n")
    
    # 1. 检查模型文件
    model_path = 'ResNet50.pt'
    if not os.path.exists(model_path):
        print(f"❌ 错误: 模型文件 {model_path} 不存在!")
        return False
    
    print(f"✅ 模型文件存在: {model_path}")
    print(f"📁 文件大小: {os.path.getsize(model_path) / (1024*1024):.1f} MB\n")
    
    # 2. 尝试加载模型
    try:
        model = torch.load(model_path, map_location='cpu')
        print("✅ 模型加载成功")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return False
    
    # 3. 检查模型类型
    if isinstance(model, nn.Module):
        print("✅ 模型是有效的PyTorch模块")
    else:
        print(f"⚠️  警告: 模型类型为 {type(model)}, 可能不是标准的PyTorch模型")
    
    # 4. 设置模型为评估模式
    model.eval()
    print("✅ 模型设置为评估模式")
    
    # 5. 测试模型输入输出
    try:
        # 创建测试输入
        test_input = torch.randn(1, 3, 224, 224)
        print(f"📊 测试输入形状: {test_input.shape}")
        
        # 前向传播
        with torch.no_grad():
            test_output = model(test_input)
        
        print(f"📊 模型输出形状: {test_output.shape}")
        print(f"📊 输出类别数: {test_output.shape[1]}")
        
        # 检查输出是否合理
        if test_output.shape[1] == 7:
            print("✅ 输出类别数正确 (7种情绪)")
        else:
            print(f"⚠️  警告: 期望7个类别，实际得到{test_output.shape[1]}个类别")
        
        # 测试softmax
        probs = torch.nn.functional.softmax(test_output, dim=1)
        prob_sum = torch.sum(probs, dim=1).item()
        print(f"📊 Softmax概率和: {prob_sum:.6f} (应该接近1.0)")
        
        if 0.99 < prob_sum < 1.01:
            print("✅ Softmax输出正常")
        else:
            print(f"⚠️  警告: Softmax概率和异常: {prob_sum}")
            
    except Exception as e:
        print(f"❌ 模型推理测试失败: {e}")
        return False
    
    # 6. 检查模型参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n📊 模型参数统计:")
    print(f"   总参数数: {total_params:,}")
    print(f"   可训练参数: {trainable_params:,}")
    
    # 7. 测试预处理流程
    print(f"\n=== 预处理流程测试 ===")
    
    # 创建测试图像
    test_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
    
    # 预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    try:
        pil_img = Image.fromarray(test_img)
        tensor_img = transform(pil_img).unsqueeze(0)
        print(f"✅ 预处理成功")
        print(f"📊 预处理后张量形状: {tensor_img.shape}")
        
        # 测试完整流程
        with torch.no_grad():
            output = model(tensor_img)
            probs = torch.nn.functional.softmax(output, dim=1)
            confidence, pred_idx = torch.max(probs, 1)
            
        print(f"✅ 完整推理流程成功")
        print(f"📊 预测类别: {pred_idx.item()}")
        print(f"📊 置信度: {confidence.item():.4f}")
        
    except Exception as e:
        print(f"❌ 预处理或推理失败: {e}")
        return False
    
    # 8. 情绪标签映射检查
    print(f"\n=== 情绪标签映射 ===")
    emotion_labels = {
        0: "surprise",
        1: "fear", 
        2: "disgust",
        3: "happiness",
        4: "sadness",
        5: "anger",
        6: "neutral"
    }
    
    for idx, emotion in emotion_labels.items():
        print(f"   {idx}: {emotion}")
    
    print(f"\n=== 模型验证完成 ===")
    print(f"✅ 模型验证通过!")
    return True

def test_with_sample_image():
    """使用样本图像测试模型"""
    print(f"\n=== 样本图像测试 ===")
    
    # 创建一个人脸样式的测试图像
    test_img = np.ones((200, 200, 3), dtype=np.uint8) * 128  # 灰色背景
    
    # 添加简单的"人脸"特征
    # 眼睛
    cv2.circle(test_img, (70, 80), 10, (255, 255, 255), -1)
    cv2.circle(test_img, (130, 80), 10, (255, 255, 255), -1)
    # 嘴巴 - 微笑
    cv2.ellipse(test_img, (100, 140), (30, 15), 0, 0, 180, (255, 255, 255), 3)
    
    # 保存测试图像
    cv2.imwrite('test_face.jpg', cv2.cvtColor(test_img, cv2.COLOR_RGB2BGR))
    print("📸 创建测试图像: test_face.jpg")
    
    # 加载模型
    model = torch.load('ResNet50.pt', map_location='cpu')
    model.eval()
    
    # 预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    pil_img = Image.fromarray(test_img)
    tensor_img = transform(pil_img).unsqueeze(0)
    
    # 推理
    with torch.no_grad():
        output = model(tensor_img)
        probs = torch.nn.functional.softmax(output, dim=1)[0]
        
    # 显示结果
    emotion_labels = {
        0: "surprise", 1: "fear", 2: "disgust", 3: "happiness",
        4: "sadness", 5: "anger", 6: "neutral"
    }
    
    print(f"\n📊 测试图像预测结果:")
    for i, prob in enumerate(probs):
        print(f"   {emotion_labels[i]}: {prob.item():.4f} ({prob.item()*100:.2f}%)")
    
    confidence, pred_idx = torch.max(probs, 0)
    print(f"\n🎯 最高置信度: {emotion_labels[pred_idx.item()]} ({confidence.item()*100:.2f}%)")

if __name__ == "__main__":
    if validate_model():
        test_with_sample_image()
    else:
        print("❌ 模型验证失败，请检查模型文件") 