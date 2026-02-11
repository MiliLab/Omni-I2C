# pipeline_config.py
import os

class Config:
    # ==================================================
    # 1. 路径配置 (全局通用)
    # ==================================================
    # Step 1 代码执行的工作目录 (存放生成的图片和代码文件)
    WORK_ROOT = "./eval_workspace"
    
    # Step 2 需要的标准答案分析文件路径 (请修改为实际路径)
    GT_ANALYSIS_FILE = "./gt_data/gt_func_para_full_list.jsonl"

    # 逻辑：程序会去这个目录下寻找 {index}.smi 文件
    GT_SMILES_DIR = "./gt_data/Image2Code_Full"

    # Step 3 需要的标准答案图片文件夹路径 (请修改为实际路径)
    GT_IMAGE_DIR = "./gt_data/Image2Code_Full"


    # ==================================================
    # 2. Step 2 API 配置 (代码逻辑评测 - Logic & Para)
    # ==================================================
    STEP2_CONFIG = {
        "api_key": "",
        "base_url": "",
        "model": "claude-sonnet-4-5-20250929", # Step 2 专用模型
    }

    # ==================================================
    # 3. Step 3 API 配置 (视觉评测 - Image Score)
    # ==================================================
    STEP3_CONFIG = {
        "api_key": "", 
        "base_url": "1",
        "model": "gpt-4.1-2025-04-14", # Step 3 专用模型
        "workers": 8
    }
    
    # 全局并发数
    WORKERS = 8