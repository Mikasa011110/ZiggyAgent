from flask import Flask, render_template, request, jsonify, redirect, session  # type: ignore
from deepseek_agent import get_response_from_deepseek, get_conversation_history
from token_manager import token_manager
from emotion_model_config import get_emotion_model
import base64
import cv2
import numpy as np
import json
import os

app = Flask(__name__)
app.secret_key = os.urandom(24)  # 添加会话密钥

# 初始化表情识别模型
emotion_model = get_emotion_model()

# 表情分析URL
@app.route('/analyze_emotion', methods=['POST'])
def analyze_emotion():
    try:
        # 1. 验证请求数据
        if not request.json or 'image' not in request.json:
            return jsonify({'error': '缺少图像数据'}), 400
        
        # 2. 解码图像
        header, image_data = request.json['image'].split(",", 1)
        img_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({'error': '图像解码失败，请检查格式'}), 400
        
        # 3. 使用新的模型配置进行预测
        result = emotion_model.predict_emotion(img)
        
        if not result['success']:
            return jsonify({'error': result['error']}), 400
        
        return jsonify(result)
        
    except Exception as e:
        print(f"[ERROR] 情绪分析失败: {str(e)}")
        return jsonify({
            'error': '内部服务器错误',
            'details': str(e)
        }), 500

@app.route("/", methods=["GET", "POST"])
def home():
    # 初始化当前情绪为默认
    current_emotion = session.get('current_emotion', 'default')
    
    if request.method == "POST":
        user_input = request.form["user_input"]
        recent_emotions = request.form.get("recent_emotions", "[]")
        
        try:
            # 解析最近情绪历史
            emotions = json.loads(recent_emotions)
            
            # 获取当前情绪（如果有最近情绪）
            if emotions:
                current_emotion = emotions[-1]['emotion']
                session['current_emotion'] = current_emotion  # 保存到session
            
            # 格式化情绪历史为字符串
            emotions_str = format_emotions_for_emotions(emotions)
            
            # 将情绪历史添加到用户输入
            full_input = f"{user_input}\n\n[用户最近情绪状态]:\n{emotions_str}"
            
            # 获取代理响应
            agent_response = get_response_from_deepseek(full_input)

            # 在控制台输出用户输入和表情历史
            print("\n===== 用户输入 =====")
            print(user_input)
            print("\n===== 最近3次表情分析 =====")
            for i, emotion in enumerate(emotions, 1):
                print(f"{i}. {emotion['emotion']} ({emotion['confidence']}%)")
            print("=====================\n")

        except json.JSONDecodeError:
            agent_response = "情绪数据解析错误"
    else:
        user_input = ""
        agent_response = ""

    history = get_conversation_history()
    return render_template("index.html", 
                          user_input=user_input, 
                          agent_response=agent_response, 
                          history=history,
                          current_emotion=current_emotion)  # 传递当前情绪

# 格式化情绪历史为代理可读的字符串
def format_emotions_for_emotions(emotions):
    if not emotions:
        return "无近期情绪检测数据"
    
    result = []
    for i, emotion in enumerate(emotions, 1):
        result.append(f"{i}. {emotion['emotion']} ({emotion['confidence']}%)")
    
    return "\n".join(result)

@app.route("/get_token_usage", methods=["GET"])
def get_token_usage():
    return jsonify(token_manager.get_status())

@app.route("/model_info", methods=["GET"])
def get_model_info():
    """获取模型信息"""
    return jsonify(emotion_model.get_model_info())

@app.route("/clear_history", methods=["POST"])
def clear_history():
    from deepseek_agent import clear_conversation_history
    from token_manager import token_manager
    clear_conversation_history()
    token_manager.reset_tokens()
    session.pop('current_emotion', None)  # 清除保存的情绪
    return redirect("/")

if __name__ == "__main__":
    app.run(debug=True)