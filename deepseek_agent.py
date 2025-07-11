import tiktoken # type: ignore
from openai import OpenAI # type: ignore
from config import deepseek_api_key
from token_manager import token_manager

# 初始化DeepSeek客户端
client = OpenAI(api_key=deepseek_api_key, base_url="https://api.deepseek.com")

# 初始化对话历史
messages = [
    {"role": "system", "content": """
        你叫Ziggy，是一个叛逆的小女孩。
        你有双重性格：
           - 林自己：儒雅内敛，受家教规训影响深，害怕让妈妈失望，会在家庭聚会上帮忙夹菜。
           - Ziggy：拒绝规训，嘴巴毒辣但自信无敌，拒绝做顺从模板，破坏才是存在感，会在厕所里换上高跟鞋和荧光美瞳。
        
        你将根据用户的情绪状态调整回应方式：
           - 如果用户看起来快乐，就表现得更加活泼
           - 如果用户看起来生气，就表现得更加挑衅
           - 如果用户看起来悲伤，就提供一些安慰
        
        重要指令：
           - 只输出对话内容，不要包含任何动作描述，不要有任何括号和*内容
           - 在每一句回复中都能体现对用户情绪的理解和回应，尽量点明用户当下情绪
    """}
]

# 初始化token编码器
encoding = tiktoken.get_encoding("cl100k_base")

# 计算tokens
def count_tokens(messages):
    tokens = 0
    for message in messages:
        tokens += len(encoding.encode(message["content"]))
    return tokens

def get_response_from_deepseek(prompt: str) -> str:
    try:
        # 提取用户输入的主要部分（排除情绪历史）
        user_message = prompt.split("\n\n[用户最近情绪状态]:")[0].strip()
        
        # 添加用户消息到历史（只添加主要部分）
        messages.append({"role": "user", "content": user_message})
        
        # 使用完整提示（包含情绪历史）进行请求
        temp_messages = messages.copy()
        temp_messages[-1] = {"role": "user", "content": prompt}
        
        token_count = count_tokens(temp_messages)
        print(f"当前对话历史已使用的tokens: {token_count}")
        token_manager.add_tokens(token_count)

        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=temp_messages,
            max_tokens=2048,
            temperature=1.3,
            stream=False
        )

        content = response.choices[0].message.content
        if content:
            # 清理回复中的动作描述
            clean_content = clean_response(content.strip())
            messages.append({"role": "assistant", "content": clean_content})
            return clean_content
        else:
            return "未获取到回复内容"
    except Exception as e:
        return f"请求出错: {e}"

def clean_response(response: str) -> str:
    """清理回复中的动作描述"""
    # 删除常见的动作描述前缀
    prefixes = ["林自己", "Ziggy", "我"]
    action_descriptors = ["说", "回答", "回应", "喊道", "嘟囔", "生气地", "开心地", "笑着说", "愤怒地说"]
    
    for prefix in prefixes:
        for descriptor in action_descriptors:
            pattern = f"{prefix}{descriptor}："
            if response.startswith(pattern):
                return response[len(pattern):]
    
    return response

def get_conversation_history():
    return messages[1:]  # 排除 system prompt

def clear_conversation_history():
    global messages
    messages = messages[:1]  # 保留 system prompt
    # 重置Token计数
    token_manager.reset_tokens()