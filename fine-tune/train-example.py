from common.llm_util import *

MODEL_ID = "Qwen/Qwen3-1.7B"
DATASET = "krisfu/delicate_medical_r1_data"
PROMPT = "你是一个医学专家，你需要根据用户的问题，给出带有思考的回答。"
MAX_LENGTH = 2048
RUN_NAME = "qwen3-1.7B"
SWANLAB_PROJECT_NAME = "qwen3-sft-medical"
MODEL_OUTPUT_DIR = f"{ProjPaths.get_project_root()}/output/{RUN_NAME}"
TEST_FILEPATH = f"{ProjPaths.get_project_root()}/data/{DATASET}/test.jsonl"


def main(lora: bool):
    # 准备数据集
    ds_processor = DatasetProcessor(DATASET, PROMPT, file_name="data", max_length=MAX_LENGTH)
    ds_processor.prepare_dataset()
    # 模型训练准备
    helper = TrainHelper(MODEL_ID, ds_processor)
    helper.use_swanlab(SWANLAB_PROJECT_NAME)
    helper.init_model()
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
        run_name=RUN_NAME,
    )
    helper.train(args)
    # 模型测试
    helper.check_model(pandas.read_json(TEST_FILEPATH, lines=True)[:3])
    swanlab.finish()


if __name__ == "__main__":
    main(lora=True)
