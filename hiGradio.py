import gradio as gr
import random


# 共享的心理安慰响应函数
def respond(message):
    """根据用户输入提供心理安慰"""
    if "开心" in message or "高兴" in message:
        return random.choice([
            "真为你感到开心！继续保持这种正能量~",
            "听到好消息真棒！正能量会传染哦！"
        ])
    elif "难过" in message or "伤心" in message:
        return random.choice([
            "给你一个虚拟拥抱🤗，我在这里陪着你",
            "感受到你的情绪了，要不要试试深呼吸三次？"
        ])
    elif "焦虑" in message or "压力" in message:
        return random.choice([
            "这种感受是暂时的，我们一起面对好吗？",
            "你已经做得很好了，给自己一点喘息空间"
        ])
    return "谢谢你愿意分享，我在认真倾听..."


# 创建对比界面
with gr.Blocks() as demo:
    gr.Markdown("# 🧠 心理助手设计对比实验")
    gr.Markdown("体验两种设计风格，感受微妙的心理差异")

    with gr.Row():
        # ❌ 冷漠版设计
        with gr.Column():
            gr.Markdown("## ❄️ 冷漠版体验")
            input_cold = gr.Textbox(label="输入问题", placeholder="在此输入你的问题")
            gr.Examples(
                examples=["我睡不着", "没人懂我", "生活没意思"],
                inputs=input_cold,
                label="常见问题示例"
            )
            btn_cold = gr.Button("获取回应", variant="secondary")
            output_cold = gr.Textbox(label="助手回复", interactive=False)
            btn_cold.click(respond, inputs=input_cold, outputs=output_cold)

        # ✅ 温暖版设计
        with gr.Column():
            gr.Markdown("## 🔥 温暖版体验")
            input_warm = gr.Textbox(
                label="悄悄告诉我你的心事...",
                placeholder="今天发生的开心/烦恼的事都行~",
                lines=2
            )
            gr.Examples(
                examples=["今天被同事夸奖了！", "养的花开了一朵🌼", "完成了一个小目标🎯"],
                inputs=input_warm,
                label="分享小确幸"
            )
            btn_warm = gr.Button("需要安慰", variant="primary")
            output_warm = gr.Textbox(label="心灵回应", interactive=False)
            btn_warm.click(respond, inputs=input_warm, outputs=output_warm)

    # 投票功能
    gr.Markdown("## 📊 你的体验反馈")
    gr.Markdown("### 哪个设计让你更愿意敞开心扉？")

    # 使用gr.State存储投票数据
    vote_data = gr.State(value={"❄️ 冷漠版": 0, "🔥 温暖版": 0, "🤔 差不多": 0})

    with gr.Row():
        vote = gr.Radio(
            choices=["❄️ 冷漠版", "🔥 温暖版", "🤔 差不多"],
            label="投票选择"
        )
        submit_btn = gr.Button("提交投票")

    # 显示格式化统计结果
    stats = gr.Textbox(label="当前投票统计", interactive=False)


    def update_vote(choice, current_data):
        # 创建数据副本避免直接修改
        new_data = current_data.copy()
        new_data[choice] += 1

        # 生成格式化统计文本
        total = sum(new_data.values())
        stats_text = f"总票数: {total}\n\n"
        for option, count in new_data.items():
            stats_text += f"{option}: {count}票 ({count / total * 100:.1f}%)\n"

        return new_data, stats_text


    submit_btn.click(
        fn=update_vote,
        inputs=[vote, vote_data],
        outputs=[vote_data, stats]
    )

# 启动应用
if __name__ == "__main__":
    demo.launch()
