import json
import os
import re
import shutil
import subprocess
import concurrent.futures
import threading 
import atexit 
import traceback
import io
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image, ImageChops

# 设置 Matplotlib 后端为 Agg (非交互模式)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ==========================================
# 0. 动态依赖导入与环境检查
# ==========================================
try:
    from playwright.sync_api import sync_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    print("⚠️ Warning: Playwright not found. HTML/SVG rendering will fail.")

try:
    from rdkit import Chem
    from rdkit.Chem import Draw
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("⚠️ Warning: RDKit not found. SMILES rendering will fail.")

# ==========================================
# 1. HTML/SVG 渲染器 (基于 Playwright)
# ==========================================

# 获取当前脚本文件的绝对路径目录
CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 本地库映射 (用于离线渲染 ECharts 等)
LOCAL_LIB_MAP = {
    "echarts": os.path.join(CURRENT_SCRIPT_DIR, "libs", "echarts.min.js"),
    "jquery": os.path.join(CURRENT_SCRIPT_DIR, "libs", "jquery.min.js")
}

VIEWPORT_SIZE = {"width": 1920, "height": 1080}
thread_local_storage = threading.local()

class HTMLRenderer:
    """
    负责启动无头浏览器并渲染 HTML/SVG 内容为图片。
    包含本地库注入和图片裁剪功能。
    """
    def __init__(self):
        self.playwright = None
        self.browser = None
        self.context = None
        self.lib_content = {}
        self._preload_libs()

    def _preload_libs(self):
        """预加载本地 JS 库"""
        for key, path in LOCAL_LIB_MAP.items():
            if os.path.exists(path):
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        self.lib_content[key] = f.read()
                except Exception as e:
                    print(f"❌ Failed to load local lib {key}: {e}")

    def start(self):
        if not PLAYWRIGHT_AVAILABLE:
            raise RuntimeError("Playwright is not installed.")
        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch(headless=True)
        self.context = self.browser.new_context(viewport=VIEWPORT_SIZE)

    def stop(self):
        try:
            if self.context: self.context.close()
            if self.browser: self.browser.close()
            if self.playwright: self.playwright.stop()
        except Exception:
            pass

    def _inject_local_libs(self, html_code):
        """将 CDN 链接替换为本地 JS 内容，实现离线渲染"""
        new_html = html_code
        for key, js_code in self.lib_content.items():
            pattern = re.compile(f'<script[^>]*src="[^"]*{key}[^"]*"[^>]*></script>', re.IGNORECASE)
            if pattern.search(new_html):
                replacement = f'<script>\n{js_code}\n</script>'
                new_html = pattern.sub(lambda match: replacement, new_html)
        
        adaptive_style = "<style>body { margin: 0; padding: 0; overflow: hidden; min-height: 100vh; }</style>"
        return new_html + adaptive_style

    def _trim_whitespace(self, image_bytes):
        """去除图片周围的纯白背景"""
        try:
            with Image.open(io.BytesIO(image_bytes)) as im:
                bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
                diff = ImageChops.difference(im, bg)
                diff = ImageChops.add(diff, diff, 2.0, -100)
                bbox = diff.getbbox()
                if bbox:
                    margin = 10
                    bbox = (max(0, bbox[0]-margin), max(0, bbox[1]-margin), 
                            min(im.width, bbox[2]+margin), min(im.height, bbox[3]+margin))
                    cropped = im.crop(bbox)
                    
                    buf = io.BytesIO()
                    cropped.save(buf, format='PNG')
                    return buf.getvalue()
                return image_bytes 
        except Exception:
            return image_bytes

    def render_to_bytes(self, html_code, selector=None):
        if not html_code: return None
        page = None
        try:
            offline_html = self._inject_local_libs(html_code)
            
            page = self.context.new_page()
            page.set_viewport_size({"width": 1920, "height": 1080})
            
            page.set_content(offline_html, wait_until="domcontentloaded", timeout=10000)
            
            if selector:
                try:
                    locator = page.locator(selector).first
                    locator.wait_for(state="attached", timeout=5000)
                    
                    box = locator.bounding_box()
                    if not box or box['width'] == 0 or box['height'] == 0:
                        page.evaluate(f"""
                            const el = document.querySelector('{selector}');
                            if (el) {{
                                if (el.clientWidth === 0) el.style.width = '500px';
                                if (el.clientHeight === 0) el.style.height = '500px';
                            }}
                        """)
                        locator = page.locator(selector).first

                    png_bytes = locator.screenshot(type='png', animations="disabled")
                    return png_bytes
                except Exception:
                    pass

            page.wait_for_timeout(2000) 
            png_bytes = page.screenshot(type='png', full_page=True, animations="disabled")
            return self._trim_whitespace(png_bytes)
            
        except Exception as e:
            raise e
        finally:
            if page: page.close()

def get_thread_renderer():
    if not PLAYWRIGHT_AVAILABLE: return None
    if not hasattr(thread_local_storage, 'renderer'):
        renderer = HTMLRenderer()
        try:
            renderer.start()
            thread_local_storage.renderer = renderer
        except Exception as e:
            print(f"❌ Renderer start failed in thread: {e}")
            return None
    return thread_local_storage.renderer

@atexit.register
def cleanup_renderers():
    if hasattr(thread_local_storage, 'renderer'):
        thread_local_storage.renderer.stop()

# ==========================================
# 2. 代码提取与辅助工具
# ==========================================

def find_pdflatex():
    candidates = [
        "/usr/local/texlive/2025/bin/x86_64-linux/pdflatex", 
        "/usr/bin/pdflatex", 
        "/Library/TeX/texbin/pdflatex",
        "pdflatex"
    ]
    for c in candidates:
        if shutil.which(c): return shutil.which(c)
    return None

def is_image_blank(image_path, threshold=5):
    try:
        with Image.open(image_path) as img:
            img_gray = img.convert('L')
            pixels = np.array(img_gray)
            if np.std(pixels) < threshold:
                return True
            return False
    except Exception:
        return False

def extract_code_by_type(text, code_type):
    if not text: return ""
    if len(text) > 50000: text = text[:50000]

    code_type = str(code_type).lower()
    tag = ""
    
    if "python" in code_type: tag = "python"
    elif "html" in code_type: tag = "html"
    elif "latex" in code_type or "tikz" in code_type: tag = "latex"
    elif "svg" in code_type: tag = "(?:xml|svg)"
    elif "smiles" in code_type: tag = "(?:text|smi|smiles)?"

    if tag:
        matches = re.findall(f"```(?:{tag})(.*?)```", text, re.DOTALL | re.IGNORECASE)
        if matches: return matches[0].strip()
        match = re.search(f"```(?:{tag})(.*)", text, re.DOTALL | re.IGNORECASE)
        if match: return match.group(1).strip()

    fallback = re.findall(r"```(.*?)```", text, re.DOTALL)
    if fallback: return fallback[0].strip()
    
    if "html" in code_type and ("<!DOCTYPE html>" in text or "<html" in text):
        pattern = re.compile(r"(<!DOCTYPE html>|<html).*</html>", re.DOTALL | re.IGNORECASE)
        match = pattern.search(text)
        if match: return match.group(0).strip()
        
    if "smiles" in code_type:
        return text.replace("```", "").strip()

    return ""

# ==========================================
# 3. 各语言验证逻辑
# ==========================================

def validate_python(code, base_path):
    target_py = str(base_path) + ".py"
    target_png = str(base_path) + ".png"
    
    code = re.sub(r"plt\.savefig\(.*\n*", "", code, flags=re.S)
    code = re.sub(r"plt\.show\(.*\n*", "", code, flags=re.S)
    code = code.strip() + f'\nplt.savefig(r"{target_png}", bbox_inches="tight")'
    
    try:
        with open(target_py, "w", encoding="utf-8") as f: f.write(code)
        
        result = subprocess.run(
            ["python", target_py], 
            timeout=60, 
            capture_output=True, 
            check=False
        )
        
        if result.returncode != 0:
            stderr_output = result.stderr.decode('utf-8', errors='ignore')
            return False, f"Python Runtime Error (Exit Code {result.returncode}):\n{stderr_output}", None

        if os.path.exists(target_png) and os.path.getsize(target_png) > 0:
            return True, None, target_png
        
        return False, "Python executed successfully but PNG was not created.", None
    except subprocess.TimeoutExpired:
        return False, "Python Execution Timeout (>60s)", None
    except Exception as e:
        return False, f"Python Script Error:\n{traceback.format_exc()}", None

def validate_latex(code, base_path, pdflatex_path):
    base_path_str = str(base_path)
    output_dir = os.path.dirname(base_path_str)
    
    target_tex = base_path_str + ".tex"
    target_pdf = base_path_str + ".pdf"
    target_png = base_path_str + ".png"
    target_log = base_path_str + ".log"
    
    if "\\documentclass" not in code:
        code = "\\documentclass[border=0pt]{standalone}\n\\usepackage{tikz}\n\\usepackage{amsmath}\n\\usetikzlibrary{arrows.meta,shapes,calc,positioning,matrix}\n\\begin{document}\n" + code + "\n\\end{document}"
    
    try:
        with open(target_tex, "w", encoding="utf-8") as f: f.write(code)
        
        compile_cmd = [pdflatex_path, "-interaction=nonstopmode", f"-output-directory={output_dir}", target_tex]
        result = subprocess.run(compile_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30, check=False)
        
        if not os.path.exists(target_pdf):
            error_report = f"LaTeX Compile Failed (Return Code {result.returncode}).\n"
            if result.stdout:
                error_report += f"STDOUT Head:\n{result.stdout.decode('latin-1', errors='ignore')[:500]}...\n"
            
            if os.path.exists(target_log):
                try:
                    with open(target_log, 'r', encoding='latin-1', errors='ignore') as log_f:
                        log_content = log_f.read()
                        critical_errors = [line for line in log_content.split('\n') if line.strip().startswith('!')]
                        if critical_errors:
                            error_report += "CRITICAL ERRORS found in LOG:\n" + "\n".join(critical_errors) + "\n"
                        error_report += f"\nLog Tail (last 2000 chars):\n{log_content[-2000:]}"
                except: pass
            return False, error_report, None

        subprocess.run(["pdftocairo", "-png", "-r", "300", "-singlefile", target_pdf, base_path_str], check=True)
        
        if os.path.exists(target_png):
            return True, None, target_png
        return False, "PDF generated but PNG conversion failed.", None

    except Exception as e:
        return False, f"LaTeX System Error: {e}", None
    finally:
        for ext in [".aux", ".log", ".pdf"]:
            if os.path.exists(base_path_str + ext):
                try: os.remove(base_path_str + ext)
                except: pass

def validate_svg(code, base_path):
    if not PLAYWRIGHT_AVAILABLE: return False, "playwright missing", None
    
    match = re.search(r'(<svg[^>]*>.*?</svg>)', code, re.DOTALL | re.IGNORECASE)
    if match:
        clean_code = match.group(1)
    else:
        start_idx = code.find("<svg")
        if start_idx != -1: clean_code = code[start_idx:]
        else: return False, "No <svg> tag found", None

    if len(clean_code) > 100000: return False, "SVG code too long", None
    
    target_svg = str(base_path) + ".svg"
    target_png = str(base_path) + ".png"

    try:
        with open(target_svg, "w", encoding="utf-8") as f: f.write(clean_code)
        
        html_wrapper = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ margin: 0; padding: 0; background-color: white; }}
                svg {{ 
                    display: block !important; 
                    min-width: 100px; min-height: 100px;
                    overflow: visible !important; 
                    font-family: "Noto Sans CJK SC", sans-serif;
                }}
            </style>
        </head>
        <body>{clean_code}</body>
        </html>
        """
        
        renderer = get_thread_renderer()
        if not renderer: return False, "Renderer init failed", None
        
        image_bytes = renderer.render_to_bytes(html_wrapper, selector="svg")
        
        if image_bytes:
            with open(target_png, "wb") as f: f.write(image_bytes)
            if is_image_blank(target_png): return False, "Rendered SVG is Blank", target_png
            return True, None, target_png
        return False, "Browser render returned None", None
    except Exception:
        return False, f"SVG Rendering Error:\n{traceback.format_exc()}", None

def validate_html(code, base_path):
    if not PLAYWRIGHT_AVAILABLE: return False, "playwright missing", None
    
    target_html = str(base_path) + ".html"
    target_png = str(base_path) + ".png"
    
    try:
        with open(target_html, "w", encoding="utf-8") as f: f.write(code)
        
        renderer = get_thread_renderer()
        if not renderer: return False, "Renderer init failed", None
        
        image_bytes = renderer.render_to_bytes(code)
        
        if image_bytes:
            with open(target_png, "wb") as f: f.write(image_bytes)
            if is_image_blank(target_png): return False, "Rendered Page is Blank", target_png
            return True, None, target_png
        return False, "Render return None", None
    except Exception:
        return False, f"HTML Error:\n{traceback.format_exc()}", None

def validate_smiles(code, base_path):
    if not RDKIT_AVAILABLE: return False, "rdkit missing", None
    if len(code) > 5000: return False, "SMILES too long", None
    
    target_smi = str(base_path) + ".smi"
    target_png = str(base_path) + ".png"
    
    try:
        with open(target_smi, "w", encoding="utf-8") as f: f.write(code)
        mol = Chem.MolFromSmiles(code.strip())
        if not mol: return False, "Invalid SMILES Syntax", None
        
        Draw.MolToFile(mol, target_png)
        if os.path.exists(target_png): return True, None, target_png
        return False, "PNG Generation Failed", None
    except Exception:
        return False, f"SMILES Error:\n{traceback.format_exc()}", None

# ==========================================
# 4. 任务分发
# ==========================================

def process_item(item, output_dir, pdflatex_path):
    idx = item.get("index", "unknown")
    c_type = str(item.get("code_type", "python")).lower()
    
    type_dir = output_dir / c_type
    type_dir.mkdir(parents=True, exist_ok=True)
    base_path = type_dir / str(idx)
    
    code = extract_code_by_type(item.get("prediction", ""), c_type)
    res = {"index": item["index"], "code_type": c_type, "execution_success": False, "error_msg": "", "generated_file": None}
    
    if not code:
        res["error_msg"] = "No code extracted"
        return res

    s, m, f = False, "", None
    
    if "python" in c_type:
        s, m, f = validate_python(code, base_path)
    elif "latex" in c_type or "tikz" in c_type:
        if not pdflatex_path: s, m, f = False, "No pdflatex found", None
        else: s, m, f = validate_latex(code, base_path, pdflatex_path)
    elif "svg" in c_type:
        s, m, f = validate_svg(code, base_path)
    elif "html" in c_type:
        s, m, f = validate_html(code, base_path)
    elif "smiles" in c_type:
        s, m, f = validate_smiles(code, base_path)
    else:
        m = f"Unknown type: {c_type}"

    # 处理生成文件路径 (转为相对路径)
    rel_path = None
    if f and os.path.exists(f):
        try:
            rel_path = os.path.relpath(f, start=output_dir.parent)
        except:
            rel_path = str(f)
            
    res.update({
        "execution_success": s, 
        "error_msg": m if not s else "", 
        "generated_file": rel_path
    })
    return res

# ==========================================
# 5. 主入口 (适配 Pipeline)
# ==========================================

def run_step1(data_list, work_dir, checkpoint_path, workers=16):
    """
    Step 1 入口函数
    :param checkpoint_path: Step 1 结果的中间存储文件 (JSONL)
    """
    print(f"🚀 [Step 1] Executing Code... Workers: {workers}")
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 读取 Checkpoint，过滤已执行项 (Success or Fail 都算已执行)
    processed_indices = set()
    if os.path.exists(checkpoint_path):
        print(f"🔍 [Step 1] Loading checkpoint: {checkpoint_path}")
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    if line.strip():
                        item = json.loads(line)
                        if "index" in item:
                            processed_indices.add(str(item["index"]))
                except: pass
        print(f"   -> Found {len(processed_indices)} items processed in checkpoint.")

    todo_list = [item for item in data_list if str(item["index"]) not in processed_indices]
    
    print(f"   -> Total: {len(data_list)}, To Run: {len(todo_list)}")
    if not todo_list:
        print("✅ [Step 1] No items to process.")
        return []

    pdflatex_path = find_pdflatex()
    results = []

    # 2. 执行并写入 Checkpoint
    # 确保目录存在
    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)

    with open(checkpoint_path, "a", encoding="utf-8") as f_chk, \
         concurrent.futures.ThreadPoolExecutor(max_workers=workers) as exe:
        
        futures = {exe.submit(process_item, item, work_dir, pdflatex_path): item for item in todo_list}
        
        for f in tqdm(concurrent.futures.as_completed(futures), total=len(todo_list), desc="Step 1 Exec"):
            item = futures[f]
            try:
                res = f.result()
                # 无论成功失败，都写入文件
                f_chk.write(json.dumps(res, ensure_ascii=False) + "\n")
                f_chk.flush()
                results.append(res)
            except Exception as e:
                # 捕获极端的 Crash
                err_res = {
                    "index": item["index"],
                    "execution_success": False,
                    "error_msg": f"Script Exception: {traceback.format_exc()}",
                    "code_type": item.get("code_type", "unknown")
                }
                f_chk.write(json.dumps(err_res, ensure_ascii=False) + "\n")
                f_chk.flush()
                results.append(err_res)

    print(f"✅ [Step 1] Done. Processed {len(results)} items.")
    return results