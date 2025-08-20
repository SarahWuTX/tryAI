from peft import LoraConfig
from transformers import TrainingArguments

from common.llm_util import *

MODEL_ID = "Qwen/Qwen3-1.7B"
DATASET = "krisfu/delicate_medical_r1_data"
SWANLAB_PROJECT_NAME = "qwen3-sft-medical"
SWANLAB_RUN_NAME = "qwen3-1.7B"
MODEL_OUTPUT_DIR = f"{ProjPaths.get_project_root()}/output/{SWANLAB_RUN_NAME}"
CONFIG = {
    "prompt": "你是一个医学专家，你需要根据用户的问题，给出带有思考的回答。",
    "max_length": 2048,
}


def run_train_exp(lora: bool):
    # 模型训练准备
    helper = TrainHelper(MODEL_ID, CONFIG)
    helper.use_swanlab(SWANLAB_PROJECT_NAME)
    helper.init_model()
    # 准备数据集
    ds_processor = DatasetProcessor(DATASET, CONFIG, file_name="data")
    ds_processor.prepare_dataset(helper.tokenizer)
    if lora:
        # 定义LoRA配置
        lora_config = LoraConfig(
            r=8, # LoRA矩阵的秩，控制低秩矩阵的维度
            lora_alpha=32, # 缩放因子，调节LoRA更新的幅度
            target_modules=["q_proj", "v_proj"], # 指定需要应用LoRA的模型层（注意力机制中的查询和值投影层）
            lora_dropout=0.05, # 随机失活概率，控制模型参数的稀疏程度，防止过拟合
            bias="none", # 是否应用偏置
            task_type="CAUSAL_LM" # 任务类型为因果语言建模
        )
        helper.set_lora(lora_config)
    # 训练参数
    args = TrainingArguments(
        output_dir=MODEL_OUTPUT_DIR,
        per_device_train_batch_size=1, # 训练批次大小
        per_device_eval_batch_size=1, # 评估批次大小
        gradient_accumulation_steps=4, # 梯度累积步数
        eval_strategy="steps", # 评估间隔
        eval_steps=100, # 评估间隔
        logging_steps=10, # 训练日志间隔
        num_train_epochs=2, # 训练轮数
        save_steps=400, # 保存间隔
        learning_rate=1e-4, # 学习率
        save_on_each_node=True, # 是否在每个节点上保存模型
        gradient_checkpointing=True, # 启用梯度检查点技术以节省显存，但会增加训练时间
        report_to="swanlab",
        run_name=SWANLAB_RUN_NAME,
    )
    helper.train(args, ds_processor)
    # 模型测试
    helper.check_model(ds_processor)
    swanlab.finish()


if __name__ == "__main__":
    run_train_exp(lora=True)
