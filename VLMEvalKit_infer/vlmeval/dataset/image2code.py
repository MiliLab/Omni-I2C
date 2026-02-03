import os
import sys
import json
import logging
import base64
import mimetypes
from ..smp import *
from .image_base import ImageBaseDataset

# logger = get_logger("image2code") # 如果有需要可以取消注释

class Image2Code(ImageBaseDataset):

    TYPE = "VQA"
    
    # 你的数据路径配置
    DATASET_URL = {
        'Image2Code_Full': '/home/zhoujiawei/LMUData/Image2Code_Full.tsv',
        'Image2Code_Html_Svg': '/home/zhoujiawei/LMUData/image2code_html_svg.tsv'
    }

    def build_prompt(self, line):
        # 调用父类方法构建 Prompt
        msgs = super().build_prompt(line)
        return msgs
    
    # 核心修改：Evaluate 只负责保存推理结果，不进行代码执行
    def evaluate(self, eval_file, **judge_kwargs):
        logger = get_logger("image2code_eval")
        
        # 1. 加载推理结果
        # VLMEvalKit 通常会把推理结果保存在 eval_file 中
        logger.info(f"Loading inference results from {eval_file}")
        infer_data_all = load(eval_file).to_dict(orient="records")
        
        # 2. 确定结果保存路径
        # 格式通常是: model_name_dataset_name_gen_results.jsonl
        infer_model = judge_kwargs.get("model", "unknown_model")
        result_file_path = os.path.abspath(
            get_intermediate_file_path(eval_file, f'_gen_results', 'jsonl')
        )

        # 3. 直接保存结果 (纯搬运)
        # 这里不做任何代码提取或执行，直接把 prediction 存下来
        logger.info(f"Saving {len(infer_data_all)} predictions to {result_file_path}...")
        
        with open(result_file_path, "w", encoding='utf-8') as f:
            for item in infer_data_all:
                # 确保每个 item 都有必要的字段，这里仅做最基础的清洗
                # 如果需要，可以在这里把 execution_success 设为 null，表示待评测
                item["execution_success"] = None 
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        logger.info("Save successful. No local execution performed.")
        logger.info(f"Please run 'eval_local_all.py' on this file to get execution stats.")

        # 4. 返回一个占位结果
        # 返回 result_file 路径给框架，框架可能需要这个路径
        return {
            "exec_rate": 0.0,   # 占位，表示尚未评测
            "success_count": 0,
            "total_count": len(infer_data_all),
            "result_file": result_file_path
        }