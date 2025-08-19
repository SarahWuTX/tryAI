# 导入所需的库
import streamlit as st

from common.llm_util import *
from common.utils import ProjPaths

MODEL_ID = "Qwen/Qwen3-1.7B"
MAX_LENGTH = 512

# 定义一个函数，用于获取模型和tokenizer
@st.cache_resource
def get_model(model_id: str):
    save_dir = ProjPaths.get_model_dir()
    model_dir = download_model(model_id=model_id, save_dir=save_dir)
    tokenizer, model = load_model(f"{save_dir}/{model_id}")
    return tokenizer, model

tokenizer, model = get_model(MODEL_ID)

# 如果用户在聊天输入框中输入了内容，则执行以下操作
if prompt := st.chat_input():
    # 将用户的输入添加到session_state中的messages列表中
    st.session_state.messages.append({"role": "user", "content": prompt})
    # 在聊天界面上显示用户的输入
    st.chat_message("user").write(prompt)

    # 构建输入
    input_ids = tokenizer.apply_chat_template(st.session_state.messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([input_ids], return_tensors="pt").to("cuda")
    generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=MAX_LENGTH)
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    # 将模型的输出添加到session_state中的messages列表中
    st.session_state.messages.append({"role": "assistant", "content": response})
    # 在聊天界面上显示模型的输出
    st.chat_message("assistant").write(response)
    # print(st.session_state)

if __name__ == '__streamlit__':
    # 在侧边栏中创建一个标题和一个链接
    with st.sidebar:
        st.markdown("## LLM")
        "[开源大模型食用指南 self-llm](https://github.com/datawhalechina/self-llm.git)"
        # 创建一个滑块，用于选择最大长度，范围在0到1024之间，默认值为512
        st.slider("max_length", 0, 1024, MAX_LENGTH, step=1)

    # 创建一个标题和一个副标题
    st.title("💬 DeepSeek-Coder-V2-Lite-Instruct")
    st.caption("🚀 A streamlit chatbot powered by Self-LLM")

    # 如果session_state中没有"messages"，则创建一个包含默认消息的列表
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "有什么可以帮您的？"}]

    # 遍历session_state中的所有消息，并显示在聊天界面上
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

"""
启动 streamlit 服务

streamlit run /root/autodl-tmp/chatBot.py --server.address 127.0.0.1 --server.port 6006 --server.enableCORS false

"""
