from ollama import Client
from typing import List, Dict, Optional, Union
import json
import os
import openai  # 新增
import anthropic  # Claude API

class ChatBot:
    def __init__(self, default_model_id: Optional[str] = None, setting_path: str = "setting.json"):
        """初始化聊天機器人
        
        Args:
            default_model_id: 預設的 model id
            setting_path: 設定檔路徑
        """
        self.setting_path = setting_path
        self.model_configs = self._load_settings()
        self.current_model_id = default_model_id
        self.model = None
        self.host = None
        self.client = None
        self.platform = None
        if default_model_id:
            self.set_model(default_model_id)

    def _load_settings(self):
        if not os.path.exists(self.setting_path):
            return {}
        with open(self.setting_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                if isinstance(data, list):
                    configs = {item['id']: item for item in data}
                else:
                    return {}
            except Exception as e:
                print(f"載入 setting.json 失敗: {e}")
                return {}
        # 新增：讀取 api_key.json 並補齊 api_key
        api_key_path = os.path.join(os.path.dirname(self.setting_path), 'api_key.json')
        if os.path.exists(api_key_path):
            try:
                with open(api_key_path, 'r', encoding='utf-8') as f:
                    api_keys = json.load(f)
            except Exception as e:
                print(f"載入 api_key.json 失敗: {e}")
                api_keys = {}
        else:
            api_keys = {}
        for config in configs.values():
            # 若有 openai_api_key 或 claude_api_key 欄位，則從 api_key.json 補上 api_key
            if 'openai_api_key' in config and not config.get('api_key'):
                key_name = config['openai_api_key']
                config['api_key'] = api_keys.get(key_name, '')
            elif 'claude_api_key' in config and not config.get('api_key'):
                key_name = config['claude_api_key']
                config['api_key'] = api_keys.get(key_name, '')
        return configs

    def set_model(self, model_id: str):
        self.current_model_id = model_id
        config = self.model_configs.get(model_id)
        if not config:
            raise ValueError(f"找不到 model_id: {model_id} 的設定")
        platform = config.get('platform')
        if platform == 'ollama':
            self.model = config.get('model')
            self.host = config.get('url')
            self.client = Client(host=self.host)
            self.platform = 'ollama'
        elif platform == 'openai':
            self.model = config.get('model')
            self.api_key = config.get('api_key')
            self.platform = 'openai'
        elif platform == 'claude':
            self.model = config.get('model')
            self.api_key = config.get('api_key')
            self.client = anthropic.Anthropic(api_key=self.api_key)
            self.platform = 'claude'
        else:
            raise ValueError(f"不支援的平台: {platform}")

    def chat(self, messages: List[Dict[str, str]], 
             temperature: float = 0.0, stream: bool = False, model_id: Optional[str] = None):
        """與AI進行對話
        
        Args:
            messages: 消息列表，包含對話歷史
            temperature: 溫度參數，控制回答的隨機性
            stream: 是否使用流式輸出
            model_id: 指定的模型 id（可選，若有則臨時切換）
            
        Returns:
            AI的回應文本或 generator（若 stream=True）
        """
        if stream:
            return self._chat_stream(messages, temperature, model_id)
        else:
            return self._chat_non_stream(messages, temperature, model_id)
    
    def _chat_stream(self, messages: List[Dict[str, str]], temperature: float, model_id: Optional[str] = None):
        """流式輸出版本"""
        try:
            if model_id and model_id != self.current_model_id:
                self.set_model(model_id)

            formatted_messages = self._format_messages(messages)

            if self.platform == 'claude':
                claude_messages, user_messages, system_message = self._prepare_claude_messages(formatted_messages)
                api_params = {
                    "model": self.model,
                    "max_tokens": 20000,  # Claude-3.5-Sonnet 的最大限制
                    "messages": user_messages,
                    "temperature": temperature
                }
                if system_message:
                    api_params["system"] = system_message

                with self.client.messages.stream(**api_params) as stream_resp:
                    for text in stream_resp.text_stream:
                        if text:
                            yield text

            elif self.platform == 'openai':
                openai_messages = self._prepare_openai_messages(formatted_messages)
                openai.api_key = self.api_key
                response = openai.chat.completions.create(
                    model=self.model,
                    messages=openai_messages,
                    temperature=temperature,
                    stream=True
                )
                for chunk in response:
                    if hasattr(chunk, 'choices') and chunk.choices:
                        delta = chunk.choices[0].delta
                        if hasattr(delta, 'content') and delta.content:
                            yield delta.content

            elif self.platform == 'ollama':
                ollama_messages = self._prepare_ollama_messages(formatted_messages)
                response = self.client.chat(
                    model=self.model,
                    messages=ollama_messages,
                    stream=True,
                    options={"temperature": temperature}
                )
                for chunk in response:
                    if isinstance(chunk, dict) and 'message' in chunk:
                        content = chunk['message'].get('content', '')
                        if content:
                            yield content

        except Exception as e:
            yield f"發生錯誤: {str(e)}"

    def _chat_non_stream(self, messages: List[Dict[str, str]], temperature: float, model_id: Optional[str] = None):
        """非流式輸出版本"""
        try:
            if model_id and model_id != self.current_model_id:
                self.set_model(model_id)

            formatted_messages = self._format_messages(messages)

            if self.platform == 'claude':
                claude_messages, user_messages, system_message = self._prepare_claude_messages(formatted_messages)
                api_params = {
                    "model": self.model,
                    "max_tokens": 20000,  # Claude-3.5-Sonnet 的最大限制
                    "messages": user_messages,
                    "temperature": temperature
                }
                if system_message:
                    api_params["system"] = system_message

                response = self.client.messages.create(**api_params)
                return response.content[0].text

            elif self.platform == 'openai':
                openai_messages = self._prepare_openai_messages(formatted_messages)
                openai.api_key = self.api_key
                response = openai.chat.completions.create(
                    model=self.model,
                    messages=openai_messages,
                    temperature=temperature,
                    stream=False
                )
                return response.choices[0].message.content

            elif self.platform == 'ollama':
                ollama_messages = self._prepare_ollama_messages(formatted_messages)
                response = self.client.chat(
                    model=self.model,
                    messages=ollama_messages,
                    stream=False,
                    options={"temperature": temperature}
                )
                return response['message']['content']

            else:
                return "不支援的平台"
        except Exception as e:
            return f"發生錯誤: {str(e)}"

    def _format_messages(self, messages):
        """統一格式化消息"""
        formatted_messages = []
        for msg in messages:
            if isinstance(msg, dict):
                if 'role' in msg and 'content' in msg:
                    formatted_messages.append(msg)
                elif 'template' in msg:
                    formatted_messages.append({'role': 'user', 'content': msg['template']})
            elif isinstance(msg, str):
                formatted_messages.append({'role': 'user', 'content': msg})
            else:
                formatted_messages.append({'role': 'user', 'content': str(msg)})
        return formatted_messages

    def _prepare_claude_messages(self, formatted_messages):
        """準備 Claude 格式的消息"""
        claude_messages = []
        for msg in formatted_messages:
            content = msg['content']
            if isinstance(content, str):
                content_list = [{"type": "text", "text": content}]
            elif isinstance(content, list):
                content_list = content
            else:
                content_list = [{"type": "text", "text": str(content)}]
            claude_messages.append({
                "role": msg["role"],
                "content": content_list
            })

        system_message = None
        user_messages = []
        for msg in claude_messages:
            if msg["role"] == "system":
                system_message = "".join([c["text"] for c in msg["content"] if c["type"] == "text"])
            else:
                user_messages.append(msg)
        
        return claude_messages, user_messages, system_message

    def _prepare_openai_messages(self, formatted_messages):
        """準備 OpenAI 格式的消息"""
        openai_messages = []
        for msg in formatted_messages:
            content = msg['content']
            if isinstance(content, list):
                content_str = "".join([c["text"] for c in content if c.get("type") == "text"])
            else:
                content_str = str(content)
            openai_messages.append({
                "role": msg["role"],
                "content": content_str
            })
        return openai_messages

    def _prepare_ollama_messages(self, formatted_messages):
        """準備 Ollama 格式的消息"""
        ollama_messages = []
        for msg in formatted_messages:
            content = msg['content']
            if isinstance(content, list):
                content_str = "".join([c["text"] for c in content if c.get("type") == "text"])
            else:
                content_str = str(content)
            ollama_messages.append({
                "role": msg["role"],
                "content": content_str
            })
        return ollama_messages
