import os
import json
import base64
import re
import time
from threading import Lock
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# -----------------------------------------------------------------------------
# 0. Prompt 配置 (兜底逻辑)
# -----------------------------------------------------------------------------
DEFAULT_PROMPT = """
You are an expert in evaluating the visual quality of generated images against ground truth images.
Subject: {subject}
Figure Type: {figure_type}

Please evaluate the generated image based on the following criteria:
1. Composition Integrity (Weight: {w_comp}%)
2. Textual Legibility (Weight: {w_text}%)
3. Entity Existence (Weight: {w_entity}%)
4. Stylistic Consistency (Weight: {w_style}%)

Return the result in JSON format with "evaluation_details" containing scores for each aspect and "score" as the final weighted score (0-100).
"""

try:
    import eval_image_prompt
    PROMPT_POOL = eval_image_prompt.PROMPT_MAP
except ImportError:
    print("⚠️ Warning: eval_image_prompt.py not found. Using default placeholder prompts.")
    PROMPT_POOL = {"default": DEFAULT_PROMPT}

# -----------------------------------------------------------------------------
# 1. 辅助工具函数
# -----------------------------------------------------------------------------
file_lock = Lock() 

def parse_json_robust(text):
    if not text or not isinstance(text, str): return None
    text = text.strip()
    try: return json.loads(text)
    except: pass
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try: return json.loads(match.group(1))
        except: pass
    match = re.search(r"(\{.*\})", text, re.DOTALL)
    if match:
        try: return json.loads(match.group(1))
        except: pass
    return None

def extract_score(resp_str, prompt_type, weights=None):
    final_score = 0.0
    details = {}
    try:
        data = parse_json_robust(resp_str)
        if data:
            details = data.get("evaluation_details", {})
            if "score" in data:
                final_score = float(data["score"])
            elif details and weights:
                def get_val(k):
                    item = details.get(k, {})
                    if isinstance(item, (int, float)): return item
                    if isinstance(item, dict): return item.get("raw_score", 0)
                    return 0
                
                s_comp = get_val("composition_integrity")
                s_text = get_val("textual_legibility")
                s_entity = get_val("entity_existence")
                s_style = get_val("stylistic_consistency")
                
                final_score = (s_comp * weights["w_comp"] + 
                               s_text * weights["w_text"] + 
                               s_entity * weights["w_entity"] + 
                               s_style * weights["w_style"]) / 10.0
            else:
                final_score = 0.0
        else:
            match = re.search(r"score[:\s]*(\d+(?:\.\d+)?)", resp_str, re.IGNORECASE)
            if match:
                final_score = float(match.group(1))
                
    except Exception as e:
        print(f"❌ extract_score parsing error: {e}")
        final_score = 0.0
    
    final_score = max(0, min(100, final_score))
    return int(round(final_score)), details

# -----------------------------------------------------------------------------
# 2. 评测执行核心
# -----------------------------------------------------------------------------

WEIGHT_PROFILES = {
    "Geometry":         {"w_comp": 30, "w_text": 15, "w_entity": 35, "w_style": 20},
    "Charts & Plots":   {"w_comp": 30, "w_text": 20, "w_entity": 35, "w_style": 15},
    "Diagrams":         {"w_comp": 30, "w_text": 20, "w_entity": 35, "w_style": 15},
    "Tables":           {"w_comp": 35, "w_text": 35, "w_entity": 15, "w_style": 15},
    "Equations":        {"w_comp": 30, "w_text": 35, "w_entity": 20, "w_style": 15},
    "Chemistry":        {"w_comp": 30, "w_text": 20, "w_entity": 35, "w_style": 15},
    "Default":          {"w_comp": 30, "w_text": 20, "w_entity": 30, "w_style": 20}
}

def eval_one_item(item, gt_root, client, model_name, prompt_type, work_root):
    index = item.get("index", "unk")
    
    code_type = str(item.get("code_type", "")).lower()
    if "smiles" in code_type:
        item["image_evaluation"] = {"score": 0, "details": "Skipped: code_type is smiles"}
        return item
    
    if not item.get("execution_success", False):
        item["image_evaluation"] = {"score": 0, "details": "Skipped: execution_success is False"}
        return item

    rel_gen_path = item.get("generated_file", "")
    if not rel_gen_path:
        item["image_evaluation"] = {"score": 0, "details": "Skipped: No generated file path"}
        return item
        
    gen_path = os.path.join(work_root, rel_gen_path)
    gt_path = os.path.join(gt_root, f"{index}.png")

    if not (os.path.exists(gt_path) and os.path.exists(gen_path)):
        missing = []
        if not os.path.exists(gt_path): missing.append("GT")
        if not os.path.exists(gen_path): missing.append("Pred")
        item["image_evaluation"] = {"score": 0, "details": f"File Missing: {', '.join(missing)}"}
        return item

    weights = WEIGHT_PROFILES["Default"]
    if prompt_type == "default":
        fig_type = item.get("figure_type", "")
        subject = item.get("subject", "")
        matched = False
        
        for k, v in [("Charts & Plots", ["line-graph", "bar-chart", "pie-chart", "scatter-plot", "3d-plot", "function-related"]), 
                     ("Geometry", ["plane-geometry", "solid-geometry","analytical-geometry"]), 
                     ("Diagrams", ["flow-chart", "schematic","circuit","relationship-diagram"])]:
            if any(t in fig_type for t in v): 
                weights = WEIGHT_PROFILES[k]
                matched = True
                break
        
        if not matched:
            if subject == "Chemistry": weights = WEIGHT_PROFILES["Chemistry"]
            elif "table" in fig_type: weights = WEIGHT_PROFILES["Tables"]
            elif subject == "Math": weights = WEIGHT_PROFILES["Equations"]
        
        tmpl = PROMPT_POOL.get(prompt_type, PROMPT_POOL.get("default", DEFAULT_PROMPT))
        prompt_text = tmpl.format(
            subject=subject, figure_type=fig_type, code_type=item.get("code_type", ""), **weights
        )
    else:
        prompt_text = PROMPT_POOL.get(prompt_type, DEFAULT_PROMPT)

    max_retries = 3
    for attempt in range(max_retries):
        try:
            def img_to_uri(p):
                with open(p, "rb") as f: return f"data:image/png;base64,{base64.b64encode(f.read()).decode()}"
            
            msgs = [{"role": "user", "content": [
                {"type": "text", "text": prompt_text},
                {"type": "image_url", "image_url": {"url": img_to_uri(gt_path)}},
                {"type": "image_url", "image_url": {"url": img_to_uri(gen_path)}}
            ]}]
            
            res = client.chat.completions.create(
                model=model_name, 
                messages=msgs,
                max_tokens=4096,
                timeout=60
            )
            content = res.choices[0].message.content
            
            score, raw_details = extract_score(content, prompt_type, weights)
            
            # [Modify] 更改输出字段名为 image_evaluation
            item["image_evaluation"] = {
                "score": score, 
                "details": content
            }
            break
        except Exception as e:
            error_str = str(e)
            print(f"❌ Error [Idx={index}, Try={attempt+1}]: {error_str}")
            
            if "image length and width" in error_str:
                item["image_evaluation"] = {"score": 0, "details": "Image dimension error"}
                return item
            
            if attempt == max_retries - 1:
                item["image_evaluation"] = {"score": 0, "details": f"API Error: {error_str}"}
            else:
                time.sleep(2)
    
    return item

# -----------------------------------------------------------------------------
# 3. 对外接口函数
# -----------------------------------------------------------------------------

def run_step3(todo_data, gt_dir, client, model_name, nproc, work_root, checkpoint_path=None, prompt_type="default"):
    results = []
    finished_indices = set()
    finished_items = []

    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"🔄 Step 3: Loading checkpoint from {checkpoint_path}")
        try:
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        finished_indices.add(item.get("index"))
                        finished_items.append(item)
            print(f"   -> Found {len(finished_indices)} finished items.")
        except Exception as e:
            print(f"⚠️ Warning: Failed to read checkpoint: {e}")

    real_todo = [item for item in todo_data if item.get("index") not in finished_indices]

    if not real_todo:
        print("⏩ Step 3: All items already evaluated.")
        return finished_items

    print(f"🚀 Starting Step 3 Evaluation: {len(real_todo)} items (Model={model_name})")

    if checkpoint_path:
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    with ThreadPoolExecutor(max_workers=nproc) as executor:
        futures = [
            executor.submit(eval_one_item, item, gt_dir, client, model_name, prompt_type, work_root) 
            for item in real_todo
        ]
        
        for future in tqdm(as_completed(futures), total=len(real_todo), desc="Evaluating Step 3"):
            try:
                res_item = future.result()
                results.append(res_item)
                
                if checkpoint_path:
                    with file_lock:
                        with open(checkpoint_path, 'a', encoding='utf-8') as f:
                            f.write(json.dumps(res_item, ensure_ascii=False) + "\n")
                        
            except Exception as e:
                print(f"❌ Uncaught exception in thread: {e}")

    return finished_items + results