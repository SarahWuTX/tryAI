import os

from dotenv import load_dotenv
from langchain_community.llms import Tongyi
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

from common.enums import ApiKey


def create_memory_chatbot():
    load_dotenv()
    # 初始化通义千问模型
    llm = Tongyi(
        api_key=os.getenv(ApiKey.ALI.value),
    )
    # 创建记忆对象
    memory = ConversationBufferMemory()
    # 创建对话链
    conversation = ConversationChain(
        llm=llm,
        memory=memory
    )
    return conversation


if __name__ == "__main__":
    chatbot = create_memory_chatbot()
    while True:
        user_input = input("你: ")
        if user_input.lower() == 'exit':
            break
        response = chatbot.predict(input=user_input)
        print("机器人: ", response)
