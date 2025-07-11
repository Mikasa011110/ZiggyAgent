from threading import Lock

class TokenManager:
    def __init__(self):
        self.total_tokens = 0
        self.lock = Lock()

    def add_tokens(self, tokens):
        with self.lock:
            self.total_tokens = tokens

    def get_status(self):
        return {
            "used_tokens": self.total_tokens,
            "limit": 4096
        }
    def reset_tokens(self):
        self.used_tokens = 0

# 全局单例
token_manager = TokenManager()