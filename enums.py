from enum import Enum

class ModelId(Enum):
    # doubao: https://www.volcengine.com/docs/82379/1330310
    DOUBAO = "doubao-seed-1-6-250615"
    DEEPSEEK_R1 = "deepseek-r1-250120"
    KIMI = "kimi-k2-250711"
    # ali：https://help.aliyun.com/zh/model-studio/getting-started/models
    QWEN3 = "qwen3-0.6b"


class Role(Enum):
    USER = "user"
    SYSTEM = "system"
    ASSISTANT = "assistant"

class ApiKey(Enum):
    DOUBAO = "DOUBAO_API_KEY"
    ALI = "ALI_API_KEY"

class BaseUrl(Enum):
    DOUBAO = "https://ark.cn-beijing.volces.com/api/v3"
    ALI = "https://dashscope.aliyuncs.com/compatible-mode/v1"