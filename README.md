# PaddleOCR-VL 独立推理工程

## 项目简介

本工程基于 [PaddleOCR-VL-1.6](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6)，实现文档图片的版面分析与内容识别，输出结构化 HTML。

**核心目标**：脱离对官方工程（PaddlePaddle、PaddleX、PaddleOCR 等）的直接依赖，改用通用的 Python 生态实现，兼容最新版本的 Python、PyTorch 和 Transformers，可直接安装运行。

推理流程：
1. **版面检测**：使用 PP-DocLayoutV3 ONNX 模型检测文档中各区域的类型和位置（标题、正文、表格、公式、图片等）；
2. **内容识别**：对每个区域裁图后送入 PaddleOCR-VL-1.6 视觉语言模型进行识别；
3. **HTML 生成**：将识别结果（含 OTSL 表格、LaTeX 公式）渲染为可在浏览器中直接查看的 HTML 文件。

## 环境准备

### 安装依赖

```bash
pip install pillow numpy opencv-python torch transformers tokenizers onnxruntime
```

> 如有 GPU，可将 `onnxruntime` 替换为 `onnxruntime-gpu`，版面检测模型将自动使用 CUDA。

### 获取版面检测模型（pp_doclayoutv3.onnx）

版面检测模型需单独获取，有以下两种方式。

**方式一：从百度网盘下载（推荐）**

百度网盘链接：https://pan.baidu.com/s/1PT2EEPwZ8KN4XglSn5KxWw?pwd=3w18

下载后将 `pp_doclayoutv3.onnx` 放到本项目的 `onnx/` 目录下，或在运行时通过 `--layout-onnx` 参数指定完整路径。

**方式二：使用 paddle2onnx 自行转换（需要 PaddlePaddle 环境）**

```bash
# 1. 安装转换工具
pip install paddle2onnx paddlepaddle

# 2. 下载官方模型（自动保存到 ~/.paddlex/official_models/）
python -c "from paddlex import create_model; create_model('PP-DocLayoutV3')"

# 3. 转换为 ONNX
paddle2onnx \
  --model_dir ~/.paddlex/official_models/PP-DocLayoutV3 \
  --model_filename inference.json \
  --params_filename inference.pdiparams \
  --save_file pp_doclayoutv3.onnx \
  --opset_version 16 --enable_onnx_checker True
```

### 获取 VL 识别模型（PaddleOCR-VL-1.6）

脚本支持从 HuggingFace 自动下载，但网络较慢时耗时较长，**建议提前手动下载**后通过 `--model-path` 参数指定本地路径。

```bash
# 方式一：使用 huggingface-cli
hf download PaddlePaddle/PaddleOCR-VL-1.6 \
    --local-dir /your/local/path/paddleocr-vl-1.6

# 方式二：使用 git lfs
git clone https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6 \
    /your/local/path/paddleocr-vl-1.6
```

## 程序说明

### inference.py — 命令行批量推理

对一张或多张文档图片执行推理，将结果保存为 HTML 文件。

**用法：**

```bash
python inference.py <输入图片或目录> --out <输出路径> [可选参数]
```

**参数：**

| 参数 | 说明 |
|---|---|
| `images` | 输入图片路径（文件或目录），支持 jpg / jpeg / png |
| `--out` | 输出 HTML 路径；单文件时直接写入该路径，多文件时作为前缀并自动补充序号后缀 |
| `--model-path` | VL 模型的本地目录或 HuggingFace repo id |
| `--layout-onnx` | PP-DocLayoutV3 版面检测 ONNX 模型路径 |
| `--recursive` | 递归扫描输入目录（仅对目录有效） |

> 输入为目录时，默认只扫描第一层文件（按文件名排序），加 `--recursive` 后递归扫描所有子目录。

**示例：**

```bash
# 处理单张图片
CUDA_VISIBLE_DEVICES=0 python inference.py sample/contract.jpg \
    --out output/contract.html \
    --model-path /your/local/path/paddleocr-vl-1.6 \
    --layout-onnx /your/local/path/pp_doclayoutv3.onnx

# 处理整个目录（输出：output/results_00.html, output/results_01.html, ...）
python inference.py sample/zcsq \
    --out output/results.html \
    --model-path /your/local/path/paddleocr-vl-1.6 \
    --layout-onnx /your/local/path/pp_doclayoutv3.onnx
```

**输出说明：**

生成的 HTML 文件包含：
- 正文文本、标题的结构化排版
- 表格（OTSL 格式转换为 HTML `<table>`）
- 公式（LaTeX，通过 MathJax 渲染）
- 图片、图表区域的占位说明

直接用浏览器打开即可查看渲染结果。

---

### scripts/server.py — HTTP 推理服务

将模型以服务形式常驻内存，通过 HTTP 接口接收请求，避免每次推理重复加载模型的开销。适合需要频繁调用推理的场景。

**额外依赖：**

```bash
pip install flask
```

**启动：**

```bash
python scripts/server.py \
    --layout-onnx /path/to/pp_doclayoutv3.onnx \
    --model-path /path/to/paddleocr-vl-1.6 \
    [--host 0.0.0.0] [--port 5000]
```

**接口：**

`GET /health` — 健康检查，模型加载完成后返回 `200 ok`，否则返回 `503`。

`POST /process` — 处理单张图片。

请求体（JSON）：

| 字段 | 必填 | 说明 |
|---|:---:|---|
| `image` | ✅ | base64 编码的图片（JPEG / PNG），支持 `data:image/...;base64,` 前缀 |
| `format` | | `"json"`（默认）或 `"html"` |

- `format=json`：返回结构化 JSON，格式为 `{"blocks": [{"label": "...", "content": "..."}, ...]}`
- `format=html`：返回完整 HTML 页面（含 MathJax），可直接在浏览器中展示

> 服务以单线程模式运行以避免 GPU 并发冲突。生产环境建议使用 gunicorn 并设置 `--workers 1`。
