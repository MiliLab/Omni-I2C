import os
import json
import re
import time
import sys
import logging
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path

# -------------------------
# 配置日志
# -------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [Thread-%(threadName)s] - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# -------------------------
# 0. 全局变量
# -------------------------
EVAL_MODEL = None
client = None

try:
    from eval_prompts import PYTHON_PROMPT, LATEX_PROMPT, HTML_PROMPT, SVG_PROMPT
except ImportError:
    PYTHON_PROMPT = "Evaluate Python:\nChecklist: {gt_analysis}\nGT: {gt_code}\nPred: {pred_code}"
    LATEX_PROMPT = "Evaluate LaTeX:\nChecklist: {gt_analysis}\nGT: {gt_code}\nPred: {pred_code}"
    HTML_PROMPT = "Evaluate HTML:\nChecklist: {gt_analysis}\nGT: {gt_code}\nPred: {pred_code}"
    SVG_PROMPT = "Evaluate SVG:\nChecklist: {gt_analysis}\nGT: {gt_code}\nPred: {pred_code}"

PROMPT_MAP = {
    "python": PYTHON_PROMPT,
    "latex": LATEX_PROMPT,
    "html": HTML_PROMPT,
    "svg": SVG_PROMPT
}

# -------------------------
# 2. 清洗逻辑
# -------------------------
def clean_python(code: str, keep_comments: bool):
    if not code: return ""
    lines = code.split('\n')
    cleaned = []
    for line in lines:
        stripped = line.replace(" ", "")
        if "plt.savefig" in stripped or "plt.close" in stripped:
            continue
        if not keep_comments:
            line = re.sub(r'#.*', '', line)
            if not line.strip(): continue 
        cleaned.append(line)
    return "\n".join(cleaned).strip()

def clean_latex(code: str, keep_comments: bool):
    if not code: return ""
    lines = code.split('\n')
    cleaned = []
    for line in lines:
        if keep_comments:
            cleaned.append(line.rstrip())
        else:
            content = re.sub(r'(?<!\\)%.*', '', line)
            if content.strip():
                cleaned.append(content.strip())
    return "\n".join(cleaned).strip()

def clean_xml(code: str, keep_comments: bool):
    if not code: return ""
    if not keep_comments:
        code = re.sub(r'', '', code, flags=re.DOTALL)
    lines = code.split('\n')
    cleaned = [line.rstrip() for line in lines if line.strip()]
    return "\n".join(cleaned).strip()

def apply_cleaning(code, code_type, mode):
    keep_comments = (mode == "with_comments")
    ctype = code_type.lower()
    if "python" in ctype: return clean_python(code, keep_comments)
    elif "latex" in ctype: return clean_latex(code, keep_comments)
    elif "html" in ctype or "svg" in ctype: return clean_xml(code, keep_comments)
    # 增强 SMILES 清洗
    elif "smiles" in ctype: 
        return code.strip().replace('\n', '').replace('\r', '').replace('"', '').replace("'", "")
    return code.strip()

# -------------------------
# 3. 评测核心
# -------------------------
try:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    logger.error("❌ RDKit not found! All SMILES evaluations will be 0.")

def evaluate_smiles_metric(gt_smiles, pred_smiles, idx="unknown"):
    """
    计算 Tanimoto Similarity
    """
    if not RDKIT_AVAILABLE:
        return 0.0
    
    gt = str(gt_smiles).strip()
    pred = str(pred_smiles).strip()

    if not gt or not pred:
        return 0.0

    try:
        # 解析 GT
        mol_gt = Chem.MolFromSmiles(gt)
        if not mol_gt:
            logger.warning(f"[Index {idx}] Failed to parse GT SMILES: '{gt[:50]}...'")
            return 0.0
        
        # 解析 Pred
        mol_pred = Chem.MolFromSmiles(pred)
        if not mol_pred:
            # 尝试截取第一段
            possible = pred.split()[0]
            mol_pred = Chem.MolFromSmiles(possible)
            if not mol_pred:
                logger.warning(f"[Index {idx}] Failed to parse Pred SMILES: '{pred[:50]}...'")
                return 0.0
        
        mfgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        fp_gt = mfgen.GetFingerprint(mol_gt)
        fp_pred = mfgen.GetFingerprint(mol_pred)
        
        return round(DataStructs.TanimotoSimilarity(fp_gt, fp_pred) * 100.0, 2)
    except Exception as e:
        logger.error(f"[Index {idx}] RDKit Calc Error: {e}")
        return 0.0

def parse_response_json(text):
    if not text: return None
    text = text.strip()
    try: return json.loads(text)
    except: pass
    pattern = r"```(?:json)?\s*(.*?)\s*```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        try: return json.loads(match.group(1))
        except: pass
    start, end = text.find('{'), text.rfind('}')
    if start != -1 and end != -1 and end > start:
        try: return json.loads(text[start:end+1])
        except: pass
    return None

def _coerce_num(x, is_float=False):
    if isinstance(x, bool): return None
    if x is None: return None
    try:
        return float(x) if is_float else int(float(x))
    except:
        return None

def _validate_and_calculate_score(d, gt_n_logic=None, gt_n_var=None):
    if not isinstance(d, dict): return (None, None)
    if "logic_evaluation" not in d or "parameter_evaluation" not in d: return (None, None)

    l_blk = d["logic_evaluation"]
    p_blk = d["parameter_evaluation"]
    
    gen_impl_func = _coerce_num(l_blk.get("gen_implemented_functions"), is_float=True)
    gen_impl_param = _coerce_num(p_blk.get("gen_implemented_parameters"), is_float=False)

    if gen_impl_func is None or gen_impl_param is None:
        return (None, "missing_gen_counts")

    final_gt_func = gt_n_logic if gt_n_logic is not None else _coerce_num(l_blk.get("gt_total_functions"))
    final_gt_param = gt_n_var if gt_n_var is not None else _coerce_num(p_blk.get("gt_total_parameters"))

    if final_gt_func is None or final_gt_param is None:
        return (None, "missing_totals")
    
    if final_gt_func <= 0 or final_gt_param <= 0:
        return (None, "zero_or_negative")

    logic_score = round((gen_impl_func / final_gt_func) * 100.0, 2)
    parameter_score = round((gen_impl_param / final_gt_param) * 100.0, 2)

    l_blk.update({
        "gt_total_functions": int(final_gt_func),
        "gen_implemented_functions": float(gen_impl_func),
        "logic_score": float(logic_score)
    })
    p_blk.update({
        "gt_total_parameters": int(final_gt_param),
        "gen_implemented_parameters": int(gen_impl_param),
        "parameter_score": float(parameter_score)
    })
    
    if not l_blk.get("reasoning"): l_blk["reasoning"] = "N/A"
    if not p_blk.get("reasoning"): p_blk["reasoning"] = "N/A"

    return (d, None)

def call_deepseek_with_validation(prompt, idx, gt_n_logic=None, gt_n_var=None, max_retries=10):
    global client, EVAL_MODEL
    zero_error_count = 0

    for attempt in range(max_retries):
        temp = min(0.1 * zero_error_count, 1.0)
        try:
            completion = client.chat.completions.create(
                model=EVAL_MODEL,
                temperature=temp,
                messages=[{"role": "user", "content": prompt}],
                extra_body={"enable_thinking": True},
                timeout=180, 
                max_tokens=4096
            )
            text = completion.choices[0].message.content
        except Exception as e:
            sleep_time = 2 ** (attempt + 1)
            logger.error(f"[Index {idx}] API Error (Attempt {attempt+1}/{max_retries}): {e}. Sleeping {sleep_time}s")
            
            if attempt < max_retries - 1:
                time.sleep(sleep_time)
                continue
            return None

        obj = parse_response_json(text)
        valid_obj, error_type = _validate_and_calculate_score(obj, gt_n_logic, gt_n_var) if obj else (None, "parse_fail")
        
        if valid_obj: 
            return valid_obj
        
        logger.warning(f"[Index {idx}] Validation failed: {error_type}. Retrying...")
        
        if attempt < max_retries - 1:
            if error_type == "zero_or_negative": zero_error_count += 1
            time.sleep(1)
            
    logger.error(f"[Index {idx}] Failed after {max_retries} attempts.")
    return None

# -------------------------
# 4. 单条处理逻辑
# -------------------------
def process_item(item, work_root, mode, gt_analysis_map, gt_smiles_dir):
    """
    [Modify] gt_smiles_dir: SMILES GT 文件夹路径
    """
    raw_type = item.get("code_type", "").lower()
    target_type = next((t for t in ["python", "latex", "html", "svg", "smiles"] if t in raw_type), None)
    idx = str(item.get("index"))

    if not target_type or not item.get("execution_success", True):
        return {
            "index": item.get("index"), "code_type": target_type, "msg": "exec_fail"
        }

    try:
        # 1. 寻找预测代码文件 (通用逻辑)
        rel_img_path = item.get("generated_file", "")
        ext_map = {"python": ".py", "smiles": ".smi", "html": ".html", "svg": ".svg", "latex": ".tex"}
        target_ext = next((ext for key, ext in ext_map.items() if key in target_type), None)
        
        if not rel_img_path or not target_ext:
            return {"index": item.get("index"), "code_type": target_type, "msg": "path_mapping_error"}

        full_img_path = os.path.join(work_root, rel_img_path)
        code_path = os.path.splitext(full_img_path)[0] + target_ext
        
        if not os.path.exists(code_path):
            logger.warning(f"[Index {idx}] File not found: {code_path}")
            return {"index": item.get("index"), "code_type": target_type, "msg": f"file_not_found: {code_path}"}
        
        with open(code_path, "r", encoding="utf-8") as f:
            pred_code = apply_cleaning(f.read(), target_type, mode)

        # ==========================================================
        # 2. SMILES 独立处理逻辑 (从文件夹读取 GT)
        # ==========================================================
        if target_type == "smiles":
            if not gt_smiles_dir or not os.path.exists(gt_smiles_dir):
                logger.error(f"[Index {idx}] GT_SMILES_DIR not found: {gt_smiles_dir}")
                return {"index": item.get("index"), "code_type": "smiles", "msg": "gt_dir_missing"}
            
            # 假设 GT 文件名为 {index}.smi
            gt_filename = f"{idx}.smi"
            gt_path = os.path.join(gt_smiles_dir, gt_filename)
            
            if not os.path.exists(gt_path):
                # 尝试 .txt 扩展名
                gt_path_txt = os.path.join(gt_smiles_dir, f"{idx}.txt")
                if os.path.exists(gt_path_txt):
                    gt_path = gt_path_txt
                else:
                    logger.warning(f"[Index {idx}] GT SMILES file not found: {gt_path}")
                    return {"index": item.get("index"), "code_type": "smiles", "msg": "gt_file_not_found"}

            try:
                with open(gt_path, "r", encoding="utf-8") as f:
                    gt_smiles_str = f.read().strip()
            except Exception as e:
                logger.error(f"[Index {idx}] Failed to read GT file: {e}")
                return {"index": item.get("index"), "code_type": "smiles", "msg": "gt_read_error"}
            
            # 清洗 GT 字符串
            gt_smiles_clean = apply_cleaning(gt_smiles_str, "smiles", mode)

            # 计算 RDKit 分数
            score = evaluate_smiles_metric(gt_smiles_clean, pred_code, idx)
            
            return {
                "index": item.get("index"), 
                "code_type": "smiles", 
                "rdkit_evaluation": {
                    "score": score, 
                    "reasoning": "RDKit TanimotoSimilarity"
                },
                "msg": "success_rdkit"
            }

        # ==========================================================
        # 3. 其他类型 (Python/LaTeX/HTML)
        # ==========================================================
        analysis_data = gt_analysis_map.get(idx)
        gt_content = ""
        
        if analysis_data and "answer" in analysis_data:
            gt_content = analysis_data["answer"]
        
        if not gt_content:
            logger.warning(f"[Index {idx}] Missing GT code in analysis.")
            return {"index": item.get("index"), "code_type": target_type, "msg": "missing_gt_code_in_analysis"}

        gt_code = apply_cleaning(gt_content, target_type, mode)

        if target_type in PROMPT_MAP:
            gt_n_logic = None
            gt_n_var = None
            prompt = ""

            has_checklist = analysis_data and "gt_analysis_details" in analysis_data
            
            if has_checklist:
                gt_details = analysis_data.get("gt_analysis_details", {})
                gt_n_logic = analysis_data.get("gt_n_logic")
                gt_n_var = analysis_data.get("gt_n_var")
                
                base_tmpl = PROMPT_MAP[target_type]
                
                if "{gt_analysis}" not in base_tmpl:
                    prompt = f"Checklist: {json.dumps(gt_details)}\n" + base_tmpl.format(gt_code=gt_code, pred_code=pred_code)
                else:
                    prompt = base_tmpl.format(
                        gt_analysis=json.dumps(gt_details), 
                        gt_code=gt_code,
                        pred_code=pred_code
                    )
            else:
                base_tmpl = PROMPT_MAP[target_type]
                try:
                    prompt = base_tmpl.format(gt_code=gt_code, pred_code=pred_code)
                except KeyError:
                    prompt = base_tmpl.replace("{gt_analysis}", "N/A").format(gt_code=gt_code, pred_code=pred_code)

            full_eval = call_deepseek_with_validation(prompt, idx, gt_n_logic, gt_n_var)
            
            if full_eval is None:
                logger.error(f"[Index {idx}] Judge Failed (None returned).")
                return {"index": item.get("index"), "code_type": target_type, "msg": "judge_fail"}
            
            full_eval.update({"index": item.get("index"), "code_type": target_type, "msg": "success_llm"})
            return full_eval

    except Exception as e:
        logger.error(f"[Index {idx}] Uncaught Exception: {str(e)}", exc_info=True)
        return {"index": item.get("index"), "code_type": target_type, "msg": f"error: {str(e)}"}

# -------------------------
# 5. 入口函数
# -------------------------
def run_step2(todo_data, gt_analysis_path, client_obj, model_name, work_root, checkpoint_path, gt_smiles_dir, nproc=16, mode="no_comments"):
    """
    [Modify] 增加 gt_smiles_dir 参数
    """
    global client, EVAL_MODEL
    client = client_obj
    EVAL_MODEL = model_name

    logger.info(f"[*] Starting Step 2 Evaluation. Model: {EVAL_MODEL}, Workers: {nproc}")
    logger.info(f"[*] Work Root: {work_root}")
    logger.info(f"[*] Checkpoint Path: {checkpoint_path}")
    logger.info(f"[*] GT SMILES Dir: {gt_smiles_dir}")

    # 1. 扫描 Checkpoint 中的已完成 Index
    processed_indices = set()
    if os.path.exists(checkpoint_path):
        logger.info(f"[*] Scanning existing checkpoint: {checkpoint_path}")
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    line = line.strip()
                    if not line: continue
                    data = json.loads(line)
                    if "index" in data:
                        processed_indices.add(str(data["index"]))
                except Exception:
                    continue
        logger.info(f"[*] Found {len(processed_indices)} completed items in checkpoint.")

    # 2. 根据 Index 进行断点过滤
    actual_todo = []
    skipped_count = 0
    for item in todo_data:
        idx = str(item.get("index"))
        if idx in processed_indices:
            skipped_count += 1
        else:
            actual_todo.append(item)
            
    logger.info(f"[*] Todo: {len(todo_data)} -> Actual Run: {len(actual_todo)} (Skipped {skipped_count})")

    if not actual_todo:
        logger.info("[*] Nothing to do after breakpoint check.")
        return []

    # 3. 加载 GT Analysis Map
    logger.info(f"[*] Loading GT Analysis from {gt_analysis_path} ...")
    gt_analysis_map = {}
    try:
        with open(gt_analysis_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                item = json.loads(line)
                if "index" in item:
                    gt_analysis_map[str(item["index"])] = item
        logger.info(f"[*] Loaded {len(gt_analysis_map)} analysis records.")
    except Exception as e:
        logger.error(f"[!] Error loading GT analysis: {e}")
        # 这里如果加载失败，非 SMILES 任务会失败，但 SMILES 任务因为是独立读取文件，依然可以运行
        # 为了稳健性，这里打印错误但不退出（或者根据实际需求处理）
        pass

    results = []
    
    # 4. 执行
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    
    active_futures = set()
    
    with open(checkpoint_path, "a", encoding="utf-8") as f_chk, \
         ThreadPoolExecutor(max_workers=nproc) as executor, \
         tqdm(total=len(actual_todo), desc="Evaluating (Step 2)", dynamic_ncols=True) as pbar:

        for item in actual_todo:
            # [Modify] 传入 gt_smiles_dir
            future = executor.submit(
                process_item, 
                item, work_root, mode, gt_analysis_map, gt_smiles_dir
            )
            active_futures.add(future)

            if len(active_futures) >= nproc * 2:
                done, active_futures = wait(active_futures, return_when=FIRST_COMPLETED)
                for fut in done:
                    try:
                        res = fut.result()
                        if res:
                            results.append(res)
                            f_chk.write(json.dumps(res, ensure_ascii=False) + "\n")
                            f_chk.flush()
                    except Exception as e:
                        logger.error(f"Future result error: {e}")
                    finally:
                        pbar.update(1)

        while active_futures:
            done, active_futures = wait(active_futures, return_when=FIRST_COMPLETED)
            for fut in done:
                try:
                    res = fut.result()
                    if res:
                        results.append(res)
                        f_chk.write(json.dumps(res, ensure_ascii=False) + "\n")
                        f_chk.flush()
                except Exception as e:
                    logger.error(f"Future result error: {e}")
                finally:
                    pbar.update(1)

    logger.info(f"[+] Step 2 Done. Processed {len(results)} new items.")
    return results