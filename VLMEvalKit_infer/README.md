# VLMEvalKit_infer

🚀 **VLMEvalKit_infer** 是一个专为多模态大模型（VLM）设计的高性能推理评估工具包。基于 [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) 开发，针对最新的硬件架构（如 NVIDIA Blackwell）和加速框架进行了深度优化。

---

## 🛠 环境配置

为了确保最佳性能（尤其是针对新一代显卡），请严格按照以下步骤配置你的推理环境。

### 1. 安装 PyTorch 与 CUDA 环境

我们建议使用 **CUDA 12.8** 以获得最佳的硬件驱动支持。

```bash
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128

```

### 2. 编译安装 Flash-Attn 2

针对不同架构，你可以选择常规安装或针对 **sm120 (Blackwell)** 架构的优化编译。

* **常规安装：**
```bash
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
MAX_JOBS=4 pip install flash-attn --no-build-isolation

```


* **Blackwell 架构定制编译 (推荐)：**
如果你使用的是 NVIDIA Blackwell 系列显卡(例如:5090)，指定架构可以显著缩短编译时间并优化性能：
```bash
export TORCH_CUDA_ARCH_LIST="12.0" 
MAX_JOBS=4 pip install flash-attn --no-build-isolation

```



### 3. 安装项目主体

克隆并以可编辑模式安装推理套件：

```bash
git clone https://github.com/MiliLab/Omni-I2C.git
cd VLMEvalKit_infer
pip install -e .

```

---

## 🚀 推理加速工具

根据你选择的模型和硬件需求，安装对应的加速后端：

### LMDeploy

```bash
# 请根据实际版本号替换 ${LMDEPLOY_VERSION} 和 ${PYTHON_VERSION}
pip install https://github.com/InternLM/lmdeploy/releases/download/v0.11.1/lmdeploy-0.11.1+cu128-cp310-cp310-manylinux2014_x86_64.whl --extra-index-url https://download.pytorch.org/whl/cu128

```

### vLLM


```bash
uv pip install vllm==0.11.0

```

---

## 📖 更多参考

本项目基于 OpenCompass 社区的成果进行扩展。更多关于评测集和基础功能的详细信息，请参考：

* **官方仓库：** [OpenCompass VLMEvalKit](https://github.com/open-compass/VLMEvalKit)
