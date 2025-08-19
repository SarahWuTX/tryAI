import json
import os
import random

import pandas
import swanlab
import torch
from datasets import Dataset
from pandas import Series
from peft import get_peft_model
from transformers import (
    AutoModelForCausalLM,
    Trainer,
    DataCollatorForSeq2Seq
)
from modelscope import snapshot_download, AutoTokenizer
from modelscope.msdatasets import MsDataset

from common.utils import ProjPaths


def download_model(model_id: str, save_dir=None):
    """
    从modelscope下载模型到本地目录
    Args:
        model_id: ModelScope上的模型标识符
        save_dir: 本地保存路径，默认为当前目录

    Returns:
        str: 下载的模型目录路径
    """
    print(f"下载模型{model_id}到{save_dir}...")
    if not save_dir:
        save_dir = ProjPaths.get_model_dir()
    model_dir = snapshot_download(model_id, cache_dir=save_dir, revision="master")
    return f"{save_dir}/{model_id}"


def load_model(model_local_dir: str):
    """
    transformers加载模型权重
    Args:
        model_local_dir:
    Returns:
        tokenizer, model
    """
    print("加载模型...")
    tokenizer = AutoTokenizer.from_pretrained(model_local_dir, use_fast=False, trust_remote_code=True)
    device_maop = {
        "": "cuda:0"
    }
    model = AutoModelForCausalLM.from_pretrained(model_local_dir, device_map=device_maop, torch_dtype=torch.bfloat16)
    return tokenizer, model


class DatasetProcessor:
    def __init__(self, ds_name, config: dict, subset_name="default", train_ratio=0.9, file_name="data"):
        self.eval_dataset = None
        self.train_dataset = None
        self.format_train_fp = None
        self.format_val_fp = None
        self.val_filepath = None
        self.train_filepath = None
        self.tokenizer = None
        self.ds_name = ds_name
        self.subset_name = subset_name
        self.train_ratio = train_ratio
        self.file_name = file_name
        for key, value in config.items():
            setattr(self, key, value)

    def download_dataset(self):
        """
            下载并处理数据集，将其划分为训练集和验证集，并保存为JSONL格式文件
        参数:
            ds_name (str): 数据集名称，用于加载对应的数据集
            subset_name (str, optional): 数据集子集名称，默认为"default"
            train_ratio (float, optional): 训练集占比，默认为0.9
            file_name (str, optional): 保存数据集的文件夹名称，默认为"dataset"
        """
        print("下载数据集...")
        # 加载数据集并打乱顺序
        ds = MsDataset.load(self.ds_name, subset_name=self.subset_name, split='train')
        data_list = list(ds)
        random.seed(42)
        random.shuffle(data_list)

        # 划分训练集和验证集
        split_idx = int(len(data_list) * self.train_ratio)
        train_data = data_list[:split_idx]
        val_data = data_list[split_idx:]

        # 写入文件
        ds_dir = "./" + self.file_name
        os.mkdir(ds_dir)
        self.train_filepath = ds_dir + "/train.jsonl"
        self.val_filepath = ds_dir + "/val.jsonl"
        with open(self.train_filepath, 'w', encoding='utf-8') as f:
            for item in train_data:
                json.dump(item, f, ensure_ascii=False)
                f.write('\n')
        with open(self.val_filepath, 'w', encoding='utf-8') as f:
            for item in val_data:
                json.dump(item, f, ensure_ascii=False)
                f.write('\n')
        print(f"The dataset has been split successfully.")
        print(f"Train Set Size：{len(train_data)}")
        print(f"Val Set Size：{len(val_data)}")
        return self.train_filepath, self.val_filepath

    def format_jsonl(self, origin_path: str):
        """
        将原始数据集转换为大模型微调所需数据格式的新数据集
        """
        print("标准化数据集...")
        new_path = origin_path.replace(".jsonl", "_format.jsonl")
        if not os.path.exists(new_path):
            messages = []
            # 读取旧的JSONL文件
            with open(origin_path, "r") as file:
                for line in file:
                    # 解析每一行的json数据
                    data = json.loads(line)
                    inputs = data["question"]
                    output = f"<think>{data["think"]}</think> \n {data["answer"]}"
                    message = {
                        "instruction": self.prompt,
                        "input": f"{inputs}",
                        "output": output,
                    }
                    messages.append(message)
            # 保存重构后的JSONL文件
            with open(new_path, "w", encoding="utf-8") as file:
                for message in messages:
                    file.write(json.dumps(message, ensure_ascii=False) + "\n")
        return new_path

    def process_jsonfile(self, json_file_path) -> Dataset:
        print("转化数据集至模型可用...")
        df = pandas.read_json(json_file_path, lines=True)
        ds = Dataset.from_pandas(df)
        dataset = ds.map(self.dataset_process_func, remove_columns=ds.column_names)
        return dataset

    def dataset_process_func(self, example):
        """
        将数据集进行预处理
        """
        input_ids, attention_mask, labels = [], [], []
        instruction = self.tokenizer(
            f"<|im_start|>system\n{self.prompt}<|im_end|>\n<|im_start|>user\n{example['input']}<|im_end|>\n<|im_start|>assistant\n",
            add_special_tokens=False,
        )
        response = self.tokenizer(f"{example['output']}", add_special_tokens=False)
        input_ids = instruction["input_ids"] + response["input_ids"] + [self.tokenizer.pad_token_id]
        attention_mask = (
                instruction["attention_mask"] + response["attention_mask"] + [1]
        )
        labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [self.tokenizer.pad_token_id]
        if len(input_ids) > self.max_length:  # 做一个截断
            input_ids = input_ids[:self.max_length]
            attention_mask = attention_mask[:self.max_length]
            labels = labels[:self.max_length]
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    def prepare_dataset(self, tokenizer):
        self.tokenizer = tokenizer
        # 获取数据集
        train_fp, val_fp = self.download_dataset()
        # 格式化数据集
        self.format_train_fp = self.format_jsonl(train_fp)
        self.format_val_fp = self.format_jsonl(val_fp)
        # 得到训练集、验证集
        self.train_dataset = self.process_jsonfile(self.format_train_fp)
        self.eval_dataset = self.process_jsonfile(self.format_val_fp)
        return self.train_dataset, self.eval_dataset


class TrainHelper:
    def __init__(self, model_id: str, config: dict):
        self.model = None
        self.tokenizer = None
        # self.dataset_processor = None
        # 初始化
        self.model_id = model_id
        self._use_swanlab = False
        for key, value in config.items():
            setattr(self, key, value)

    def init_model(self):
        # 在modelscope上下载Qwen模型到本地目录下
        model_dir = download_model(self.model_id)
        # Transformers加载模型权重
        self.tokenizer, self.model = load_model(model_local_dir=model_dir)
        self.model.enable_input_require_grads()  # 开启梯度检查点时，要执行该方法

    def use_swanlab(self, project_name):
        self._use_swanlab = True
        os.environ["SWANLAB_PROJECT"] = project_name
        swanlab.config.update({
            "model": self.model_id,
            "prompt": self.prompt,
            "data_max_length": self.max_length,
        })
        print("[done]swanlab update")

    def set_lora(self, lora_config):
        print("设置lora...")
        self.lora_config = lora_config
        self.model = get_peft_model(self.model, lora_config)

    def train(self, train_args, dataset_processor: DatasetProcessor):
        print("开始训练...")
        if dataset_processor.train_dataset is None or dataset_processor.eval_dataset is None:
            dataset_processor.prepare_dataset(self.tokenizer)
        self.trainer = Trainer(
            model=self.model,
            args=train_args,
            train_dataset=dataset_processor.train_dataset,
            eval_dataset=dataset_processor.eval_dataset,
            data_collator=DataCollatorForSeq2Seq(tokenizer=self.tokenizer, padding=True),
        )
        self.trainer.train()

    def predict(self, messages):
        device = "cuda"
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(device)
        generated_ids = self.model.generate(
            model_inputs.input_ids,
            max_new_tokens=self.max_length,
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response

    def check_model(self, ds_processor: DatasetProcessor, n=3):
        # 用测试集的前3条，主观看模型
        test_series = pandas.read_json(ds_processor.format_val_fp, lines=True)[:n]
        test_text_list = []
        for index, row in test_series.iterrows():
            instruction = row['instruction']
            input_value = row['input']
            messages = [
                {"role": "system", "content": f"{instruction}"},
                {"role": "user", "content": f"{input_value}"}
            ]
            response = self.predict(messages)
            response_text = f"""
            Question: {input_value}

            LLM:{response}
            """
            print(response_text)
            test_text_list.append(response_text)
        if self._use_swanlab:
            swanlab.log({"Prediction": [swanlab.Text(x)] for x in test_text_list})
        return test_text_list


def dow1():
    download_model("Qwen/Qwen3-1.7B")
    tokenizers, model = load_model("./Qwen/Qwen3-1.7B")
