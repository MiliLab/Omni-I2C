#!/bin/bash

# 初始化 Conda（这句很关键）
eval "$(conda shell.bash hook)"

cd /home/zhoujiawei/VLMEvalKit/VLMEvalKit

conda activate image2code

# python run.py --data Image2Code_Html --model Claude4.5_Sonnet --verbose --reuse 
python run.py --data Image2Code_Full --model Claude4.5_Sonnet --verbose --reuse 

# 打印一下环境信息，方便排查
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
nvidia-smi

# ------------------------------------------------------------------
# 【核心修改】
# 1. 去掉了 MASTER_PORT 和 torchrun
# 2. 直接使用 python 运行
# 3. 确保你的 config.py 里已经配置了 use_lmdeploy=True 和 tp=4
# ------------------------------------------------------------------

python run.py \
    --data Image2Code_Full_2_1 \
    --model InternVL3_5-38B-Instruct \
    --verbose