import argparse
import json
import os
import pandas as pd
from pathlib import Path
from openai import OpenAI

# 导入配置
from pipeline_config import Config

# 导入三个步骤的执行函数
from step1_execute import run_step1
from step2_evaluate import run_step2
from step3_evaluate import run_step3

# -----------------------------------------------------------------------------
# 1. 基础数据加载与工具函数
# -----------------------------------------------------------------------------

def load_initial_data(input_path):
    data_list = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                data_list.append(item)
    return data_list

def load_checkpoint_indices(checkpoint_path):
    indices = set()
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        return indices
    
    try:
        with open(checkpoint_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        item = json.loads(line)
                        if "index" in item:
                            indices.add(str(item["index"]))
                    except: pass
    except Exception as e:
        print(f"⚠️ Error reading checkpoint {checkpoint_path}: {e}")
    
    return indices

def merge_checkpoint_to_data(data_list, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        return data_list

    chk_map = {}
    with open(checkpoint_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    item = json.loads(line)
                    chk_map[str(item['index'])] = item
                except: pass
    
    count = 0
    for item in data_list:
        idx = str(item['index'])
        if idx in chk_map:
            item.update(chk_map[idx])
            count += 1
    
    print(f"   -> Merged {count} items from checkpoint: {os.path.basename(checkpoint_path)}")
    return data_list

def save_final_results(data, output_path):
    print(f"💾 Saving final detailed results to: {output_path}")
    temp_path = str(output_path) + ".tmp"
    with open(temp_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    if os.path.exists(output_path): os.remove(output_path)
    os.rename(temp_path, output_path)

def generate_final_report(data, output_md_path, model_name):
    print(f"📊 Generating final statistics report...")
    df = pd.DataFrame(data)
    
    # [Modify] 增加 image_evaluation 检查
    required_cols = ['execution_success', 'logic_evaluation', 'parameter_evaluation', 'rdkit_evaluation', 'image_evaluation']
    for c in required_cols:
        if c not in df.columns: df[c] = None

    def get_score(row, source, keys):
        val = row.get(source)
        if isinstance(val, dict):
            for k in keys:
                val = val.get(k)
                if val is None: return 0.0
            try: return float(val)
            except: return 0.0
        return 0.0

    def norm_type(t):
        t = str(t).lower()
        if "smiles" in t: return "SMILES"
        if "python" in t: return "Python"
        if "latex" in t or "tikz" in t or "tex" in t: return "LaTeX"
        if "html" in t: return "HTML"
        if "svg" in t: return "SVG"
        return "Other"
    
    df['type_category'] = df['code_type'].apply(norm_type)
    df['exec_success'] = df['execution_success'].fillna(False).astype(bool)
    
    df['logic_score'] = df.apply(lambda r: get_score(r, 'logic_evaluation', ['logic_score']), axis=1)
    df['param_score'] = df.apply(lambda r: get_score(r, 'parameter_evaluation', ['parameter_score']), axis=1)
    
    # [Modify] 从 image_evaluation 获取分数
    df['image_score'] = df.apply(lambda r: get_score(r, 'image_evaluation', ['score']), axis=1)
    df['rdkit_score'] = df.apply(lambda r: get_score(r, 'rdkit_evaluation', ['score']), axis=1)

    summary_md = f"# Evaluation Report: {model_name}\n\n"
    
    total = len(df)
    total_exec = df['exec_success'].sum()
    summary_md += f"## 1. Overall Overview\n"
    summary_md += f"- **Total Samples**: {total}\n"
    summary_md += f"- **Overall Execution Success**: {total_exec}/{total} ({total_exec/total*100:.2f}%)\n\n"

    summary_md += f"## 2. Detailed Statistics by Code Type\n\n"
    
    # Part A: Non-SMILES
    non_smiles_df = df[df['type_category'] != 'SMILES'].copy()
    if not non_smiles_df.empty:
        summary_md += "### A. General Code Types (Python, LaTeX, HTML, SVG)\n"
        stats = non_smiles_df.groupby('type_category').agg({
            'exec_success': ['count', 'mean'],
            'logic_score': 'mean',
            'param_score': 'mean',
            'image_score': 'mean'
        })
        stats.columns = ['Count', 'Success Rate', 'Avg Logic', 'Avg Param', 'Avg Image']
        stats['Success Rate'] = stats['Success Rate'] * 100
        
        total_ns = len(non_smiles_df)
        avg_ns = pd.DataFrame({
            'Count': [total_ns],
            'Success Rate': [non_smiles_df['exec_success'].mean() * 100],
            'Avg Logic': [non_smiles_df['logic_score'].mean()],
            'Avg Param': [non_smiles_df['param_score'].mean()],
            'Avg Image': [non_smiles_df['image_score'].mean()]
        }, index=['AVERAGE (Non-SMILES)'])
        
        final_table = pd.concat([stats, avg_ns])
        summary_md += final_table.round(2).to_markdown() + "\n\n"
        
    # Part B: SMILES
    smiles_df = df[df['type_category'] == 'SMILES'].copy()
    if not smiles_df.empty:
        summary_md += "### B. Chemical Molecules (SMILES)\n"
        s_count = len(smiles_df)
        s_exec_rate = smiles_df['exec_success'].mean() * 100
        s_tanimoto = smiles_df['rdkit_score'].mean()
        
        smiles_stats = pd.DataFrame({
            'Metric': ['Count', 'Execution Success Rate', 'Avg Tanimoto Similarity'],
            'Value': [f"{s_count}", f"{s_exec_rate:.2f}%", f"{s_tanimoto:.2f}"]
        })
        summary_md += smiles_stats.to_markdown(index=False) + "\n\n"

    with open(output_md_path, 'w', encoding='utf-8') as f:
        f.write(summary_md)
    print(f"✅ Report saved to: {output_md_path}")

# -----------------------------------------------------------------------------
# 2. 主流程
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", "-i", required=True)
    parser.add_argument("--model_name", "-m", required=True)
    args = parser.parse_args()

    Config.INPUT_FILE_PATH = args.input_file
    output_dir = Path("output") / args.model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    result_jsonl = output_dir / f"{args.model_name}_eval_details.jsonl"
    report_md = output_dir / f"{args.model_name}_report.md"
    
    step1_work_dir = Path(Config.WORK_ROOT) / args.model_name
    
    chk1 = step1_work_dir / "step1_checkpoint.jsonl"
    chk2 = step1_work_dir / "step2_checkpoint.jsonl" 
    chk3 = step1_work_dir / "step3_checkpoint.jsonl" 

    print(f"🚀 Pipeline Start: {args.model_name}")

    # 1. 加载原始数据
    data = load_initial_data(args.input_file)
    print(f"📚 Loaded {len(data)} initial items.")

    # Step 1
    step1_finished_indices = load_checkpoint_indices(str(chk1))
    todo_step1 = []
    for item in data:
        idx = str(item['index'])
        if idx not in step1_finished_indices:
            todo_step1.append(item)
    
    if todo_step1:
        print(f"⚡ Step 1: Executing {len(todo_step1)} items...")
        run_step1(todo_step1, step1_work_dir, str(chk1), workers=Config.WORKERS)
    else:
        print("⏩ Step 1 Skipped.")

    data = merge_checkpoint_to_data(data, str(chk1))

    # Step 2
    step2_finished_indices = load_checkpoint_indices(str(chk2))
    todo_step2 = []
    for item in data:
        idx = str(item['index'])
        if idx in step2_finished_indices: continue
        todo_step2.append(item)

    if todo_step2:
        print(f"🧠 Step 2: Evaluating {len(todo_step2)} items...")
        client_s2 = OpenAI(api_key=Config.STEP2_CONFIG['api_key'], base_url=Config.STEP2_CONFIG['base_url'])
        
        run_step2(
            todo_data=todo_step2,
            gt_analysis_path=Config.GT_ANALYSIS_FILE,
            client_obj=client_s2,
            model_name=Config.STEP2_CONFIG['model'],
            work_root=str(Config.WORK_ROOT),
            checkpoint_path=str(chk2),
            gt_smiles_dir=Config.GT_SMILES_DIR,
            nproc=Config.WORKERS
        )
    else:
        print("⏩ Step 2 Skipped.")

    data = merge_checkpoint_to_data(data, str(chk2))

    # Step 3
    step3_finished_indices = load_checkpoint_indices(str(chk3))
    todo_step3 = []
    for item in data:
        idx = str(item['index'])
        if idx in step3_finished_indices: continue
        
        if "smiles" in str(item.get("code_type", "")).lower(): continue
        
        # [Modify] 只要没有 image_evaluation 就跑
        if "image_evaluation" not in item:
            todo_step3.append(item)

    if todo_step3:
        print(f"👁️ Step 3: Evaluating {len(todo_step3)} items...")
        client_s3 = OpenAI(api_key=Config.STEP3_CONFIG['api_key'], base_url=Config.STEP3_CONFIG['base_url'])
        
        run_step3(
            todo_data=todo_step3,
            gt_dir=Config.GT_IMAGE_DIR,
            client=client_s3,
            model_name=Config.STEP3_CONFIG['model'],
            nproc=Config.WORKERS,
            work_root=str(Config.WORK_ROOT),
            checkpoint_path=str(chk3)
        )
    else:
        print("⏩ Step 3 Skipped.")

    data = merge_checkpoint_to_data(data, str(chk3))

    # 收尾
    save_final_results(data, result_jsonl)
    generate_final_report(data, report_md, args.model_name)

if __name__ == "__main__":
    main()