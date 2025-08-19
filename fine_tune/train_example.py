from peft import LoraConfig
from transformers import TrainingArguments

from common.llm_util import *

MODEL_ID = "Qwen/Qwen3-1.7B"
DATASET = "krisfu/delicate_medical_r1_data"
PROMPT = "你是一个医学专家，你需要根据用户的问题，给出带有思考的回答。"
SWANLAB_PROJECT_NAME = "qwen3-sft-medical"
SWANLAB_RUN_NAME = "qwen3-1.7B"
MODEL_OUTPUT_DIR = f"{ProjPaths.get_project_root()}/output/{SWANLAB_RUN_NAME}"
MAX_LENGTH = 2048
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
            r=8,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        helper.set_lora(lora_config)
    # 训练参数
    args = TrainingArguments(
        output_dir=MODEL_OUTPUT_DIR,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        eval_strategy="steps",
        eval_steps=100,
        logging_steps=10,
        num_train_epochs=2,
        save_steps=400,
        learning_rate=1e-4,
        save_on_each_node=True,
        gradient_checkpointing=True,
        report_to="swanlab",
        run_name=SWANLAB_RUN_NAME,
    )
    helper.train(args, ds_processor)
    # 模型测试
    helper.check_model(ds_processor)
    swanlab.finish()


if __name__ == "__main__":
    run_train_exp(lora=True)
