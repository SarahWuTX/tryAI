from camel.agents import ChatAgent
from camel.messages import BaseMessage
from camel.models import ModelFactory
from camel.types import RoleType, ModelPlatformType
import os
from dotenv import load_dotenv
import time


def agent_test():
    # 加载环境变量 (例如你的API Key)
    load_dotenv()
    # 假设你已经设置了 MODELSCOPE_SDK_TOKEN 或 OPENAI_API_KEY
    # 确保你的API Key是有效的，这里以ModelScope的Qwen为例
    # 如果使用OpenAI，请将 ModelPlatformType.OPENAI 和对应的 model_type 替换
    # api_key = os.getenv('MODELSCOPE_SDK_TOKEN')

    # 创建模型实例 (这里使用ModelScope的Qwen2.5-72B-Instruct作为示例)
    # 实际使用时，请替换为你的模型配置
    model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,  # 兼容OpenAI API的平台
        model_type="doubao-seed-1-6-250615",  # 你的模型类型
        url='https://ark.cn-beijing.volces.com/api/v3',  # 你的模型API地址
        api_key="b4e0fbd6-b235-4827-8124-22677f98e823"
    )

    # 1. 创建系统提示词 (定义角色职责)
    # 助手Agent：富有同情心的心理健康支持者
    assistant_sys_msg = BaseMessage(
        role_name="心理健康助手",
        role_type=RoleType.ASSISTANT,  # Camel内置角色类型
        content="你是一位富有同情心和专业性的心理健康支持助手。你的目标是温和地引导用户表达感受并提供支持性建议，但绝不提供诊断或医疗建议。",
        meta_dict=None
    )

    # 用户Agent：正在寻求心理支持的大学学生
    user_sys_msg = BaseMessage(
        role_name="用户",
        role_type=RoleType.USER,
        content="你是一位正在寻求心理支持的大学学生，最近感到压力很大。",
        meta_dict=None
    )

    # 2. 创建Agent实例
    assistant_agent = ChatAgent(assistant_sys_msg, model=model)
    user_agent = ChatAgent(user_sys_msg, model=model)

    print("--- 两个AI角色的首次对话 ---")

    # 3. 用户Agent发起对话 (模拟用户输入)
    user_msg = BaseMessage(
        role_name="用户",
        role_type=RoleType.USER,
        content="你好，我最近感觉压力很大，睡不好觉。",
        meta_dict=None
    )

    # 4. 助手Agent回应
    # 注意：这里为了演示简化，我们直接让助手Agent对用户消息进行一步回应。
    # 实际多轮对话中，会有一个循环来管理消息的传递和上下文。
    assistant_response = assistant_agent.step(user_msg)
    print(f"用户: {user_msg.content}")
    print(f"心理健康助手: {assistant_response.msg.content}")

    # 5. 用户Agent再次回应，模拟多轮对话
    user_msg_2 = BaseMessage(
        role_name="用户",
        role_type=RoleType.USER,
        content="是的，主要是因为期末考试临近，感觉时间不够用。",
        meta_dict=None
    )
    assistant_response_2 = assistant_agent.step(user_msg_2)
    print(f"用户: {user_msg_2.content}")
    print(f"心理健康助手: {assistant_response_2.msg.content}")


if __name__ == '__main__':
    agent_test()