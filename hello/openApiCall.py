import os

from openai import OpenAI
from dotenv import load_dotenv

from common.enums import *

load_dotenv()
DOUBAO_CLIENT = OpenAI(
    # 此为默认路径，您可根据业务所在地域进行配置
    base_url=BaseUrl.DOUBAO.value,
    # 从环境变量中获取您的 API Key。此为默认方式，您可根据需要进行修改
    api_key=os.getenv(ApiKey.DOUBAO.value),
)
ALI_CLIENT = OpenAI(
    api_key=os.getenv(ApiKey.ALI.value),
    base_url=BaseUrl.ALI.value,
)

class Response:
    msg = ""
    success = True
    def __init__(self, msg, success):
        self.msg = msg
        self.success = success


class AiAgent:
    """
    AI助手代理类
    Attributes:
        __messages (list): 存储对话历史的消息列表
        __client: AI客户端实例
        __model: 使用的模型ID
        __settings (dict): 额外的配置参数
    """

    def __init__(self, model: ModelId, base_msg="", **kwargs):
        self.__messages = []
        self.__client = None
        self.__model = None
        self.__settings={}
        self.__model = model.value
        self.__client = ALI_CLIENT if model == model.QWEN3 else DOUBAO_CLIENT
        if base_msg:
            self.__messages.append(AiAgent.sys_msg(base_msg))
        self.__settings = kwargs

    def call(self, request: str, **kwargs):
        self.__messages.append(AiAgent.user_msg(request))
        response = ""
        try:
            stream = self.__client.chat.completions.create(
                model=self.__model,
                messages=self.__messages,
                stream=True,
                **self.__settings,
                **kwargs
            )
            # print("请求中，请耐心等待...")
            # waiting_effect()
            for chunk in stream:
                if not chunk.choices or not chunk.choices[0].delta.content:
                    continue
                response += chunk.choices[0].delta.content
                print(chunk.choices[0].delta.content, end="")
            self.__messages.append(AiAgent.assist_msg(response))
            print("\n")
            return Response(response, True)
        except Exception as e:
            print(e)
            return Response("failed", False)

    @staticmethod
    def user_msg(msg: str):
        return {"role": Role.USER.value, "content": msg}

    @staticmethod
    def sys_msg(msg: str):
        return {"role": Role.SYSTEM.value, "content": msg}

    @staticmethod
    def assist_msg(msg: str):
        return {"role": Role.ASSISTANT.value, "content": msg}

# def waiting_effect():
#     for _ in range(3):
#         sys.stdout.write('.')
#         sys.stdout.flush()
#     sys.stdout.write('\r')
#     sys.stdout.flush()

def try_my_agent():
    helper = AiAgent(
        model=ModelId.QWEN3,
        base_msg="你是一位富有同情心和专业性的心理健康支持助手。你的目标是温和地引导用户表达感受并提供支持性建议，但绝不提供诊断或医疗建议",
        max_tokens=100,
    )
    while True:
        request = input("请输入：")
        if request == "exit":
            break
        helper.call(request, timeout=1800)

def two_ai_chat():
    helper = AiAgent(
        model=ModelId.QWEN3,
        base_msg="请你扮演一位富有同情心和专业性的心理健康支持助手。你的目标是温和地引导用户表达感受并提供支持性建议，但绝不提供诊断或医疗建议。与我进行模拟对话，每句话50字以内。",
        max_tokens=100,
        timeout=1800,
    )
    user = AiAgent(
        model=ModelId.QWEN3,
        base_msg="请你扮演一位正在寻求心理支持的大学学生。与我进行模拟对话，每句话50字以内。请模拟情景，提出你的困扰",
        max_tokens=100,
        timeout=1800
    )
    user.call(
        "请你扮演一位正在寻求心理支持的大学学生。与我进行模拟对话，每句话50字以内。请模拟情景，提出你的困扰")
    helper.call(
        "请你扮演一位富有同情心和专业性的心理健康支持助手。你的目标是温和地引导用户表达感受并提供支持性建议，但绝不提供诊断或医疗建议。与我进行模拟对话，每句话50字以内。")
    res = Response("有什么可以帮你", True)
    for _ in range(6):
        # request = input("请输入：")
        # if request == "exit":
        #     break
        print("用户：", end="")
        res = user.call(res.msg)
        if not res.success:
            print("请求失败")
            break
        print("AI助手：", end="")
        res = helper.call(res.msg)
        if not res.success:
            print("请求失败")
            break
    print()

def doubao():
    client = OpenAI(
        # 此为默认路径，您可根据业务所在地域进行配置
        base_url=BaseUrl.DOUBAO.value,
        # 从环境变量中获取您的 API Key。此为默认方式，您可根据需要进行修改
        api_key=os.getenv(ApiKey.DOUBAO.value),
    )

    student = client.chat.completions.create(
        # 指定您创建的方舟推理接入点 ID，此处已帮您修改为您的推理接入点 ID
        model=ModelId.DOUBAO.value,
        messages=[
            {"role": "user", "content": "你好，我是艾AA"},
        ],
        max_tokens=40,
    )
    print(student.choices[0].message.content)

    student2 = client.chat.completions.create(
        model=ModelId.DOUBAO.value,
        messages=[
            {"role": "user", "content": "你好，我是艾AA"},
            {"role": "assistant", "content": student.choices[0].message.content},
            {"role": "user", "content": "你好，我是谁？"},
        ],
        max_tokens=40,
    )
    print(student2.choices[0].message.content)


def ali():
    client = OpenAI(
        api_key=os.getenv(ApiKey.ALI.value),
        base_url=BaseUrl.ALI.value,
    )
    completion = client.chat.completions.create(
        model=ModelId.QWEN3.value,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "你好"},
        ],
        # Qwen3模型通过enable_thinking参数控制思考过程（开源版默认True，商业版默认False）
        # 使用Qwen3开源版模型时，若未启用流式输出，请将下行取消注释，否则会报错
        extra_body={"enable_thinking": False},
    )
    print(completion.choices[0].message.content)
