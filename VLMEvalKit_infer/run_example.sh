#!/bin/bash

# 初始化 Conda（这句很关键）
eval "$(conda shell.bash hook)"

# 指定运行文件夹
# cd <Your folder>/VLMEvalKit_infer

conda activate omni_i2c

python run.py --data Image2Code_Full --model Claude4.5_Sonnet --verbose --reuse 

python run.py --data Image2Code_Full --model Gemini2_5_Pro --verbose --reuse 

python run.py --data Image2Code_Full --model Gemini3_Pro --verbose --reuse 

python run.py --data Image2Code_Full --model GPT_5_1 --verbose --reuse 

python run.py --data Image2Code_Full --model InternVL3_5_241B_A28B--verbose --reuse 

python run.py --data Image2Code_Full --model InternVL3_5-38B-Instruct --verbose --reuse 

python run.py --data Image2Code_Full --model InternVL3_5-8B-Instruct --verbose --reuse 

python run.py --data Image2Code_Full --model Qwen3-VL-235b-a22b-Instruct --verbose --reuse 

python run.py --data Image2Code_Full --model Qwen3-VL-32B-Instruct --verbose --reuse 

python run.py --data Image2Code_Full --model Qwen3-VL-8B-Instruct --verbose --reuse 

python run.py --data Image2Code_Full --model Qwen2.5-VL-72B-Instruct --verbose --reuse 

python run.py --data Image2Code_Full --model Qwen2.5-VL-7B-Instruct --verbose --reuse 

python run.py --data Image2Code_Full --model Gemma3-27B-Instruct --verbose --reuse 