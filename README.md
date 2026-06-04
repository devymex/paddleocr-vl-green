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
pip install -r requirements.txt
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

### scripts/server.py — HTTP 推理服务

将模型以服务形式常驻内存，通过 HTTP 接口接收请求，避免每次推理重复加载模型的开销。支持多进程（多卡）并发推理，并发请求进入 FIFO 队列并自动分发给空闲 worker。

**启动：**

```bash
# CPU 单 worker
python scripts/server.py \
    --layout-onnx /path/to/pp_doclayoutv3.onnx \
    --model-path /path/to/paddleocr-vl-1.6 \
    --device cpu

# 单 GPU（1 个 worker，使用 GPU 0）
CUDA_VISIBLE_DEVICES=0 python scripts/server.py \
    --layout-onnx /path/to/pp_doclayoutv3.onnx \
    --model-path /path/to/paddleocr-vl-1.6 \
    --device cuda:0

# 多卡多 worker（2 张卡，每卡 2 个 worker，共 4 个 worker）
CUDA_VISIBLE_DEVICES=0,1 python scripts/server.py \
    --layout-onnx /path/to/pp_doclayoutv3.onnx \
    --model-path /path/to/paddleocr-vl-1.6 \
    --device cuda:0,0,1,1 \
    [--host 0.0.0.0] [--port 5000]
```

**`--device` 参数说明：**

| 值 | worker 数 | 说明 |
|---|:---:|---|
| `cpu` | 1 | 单 CPU worker |
| `cuda` | 1 | 等价于 `cuda:0`（简写） |
| `cuda:0` | 1 | GPU 0 上的 1 个 worker |
| `cuda:0,1` | 2 | GPU 0 和 GPU 1 各 1 个 worker |
| `cuda:0,0,1,1` | 4 | GPU 0 和 GPU 1 各 2 个 worker |

GPU 序号为逻辑编号，受环境变量 `CUDA_VISIBLE_DEVICES` 控制。例如设置 `CUDA_VISIBLE_DEVICES=2,3` 后，`--device cuda:0,1` 实际使用物理 GPU 2 和 3。

**并发处理机制：**

- 主进程（Flask）以多线程模式接收并发 HTTP 请求；
- 每个推理请求进入共享 FIFO 任务队列；
- 各 worker 空闲时自动从队列取任务，实现负载均衡；
- 每个 worker 运行独立的模型副本，互不干扰；
- worker 使用 `spawn` 启动方式，避免 fork + CUDA 的潜在冲突。

**接口：**

`GET /health` — 健康检查，全部 worker 加载完成后返回 `200 ok`，否则返回 `503`。

`POST /process` — 处理单张图片。

请求体（JSON）：

| 字段 | 必填 | 说明 |
|---|:---:|---|
| `image` | ✅ | base64 编码的图片（JPEG / PNG），支持 `data:image/...;base64,` 前缀 |
| `format` | | `"json"`（默认）或 `"html"` |

- `format=json`：返回结构化 JSON，格式为 `{"blocks": [{"label": "...", "content": "..."}, ...]}`
- `format=html`：返回完整 HTML 页面（含 MathJax），可直接在浏览器中展示

### scripts/test.py — 并发性能测试

用于测试 HTTP 推理服务的并发处理能力和输出正确性。该脚本向运行中的服务器发送多个并发请求，使用相同的测试图片，验证响应的 HTML 与预期输出是否一致，并汇总统计结果。

**功能：**

- 等待服务器 `/health` 端点就绪；
- 读取本地测试图片（默认 `sample/contract.jpg`）并转换为 base64；
- 向 `/process` 端点发送指定数量的并发请求；
- 逐一比对响应的 HTML 内容与预期文件（默认 `output/contract.html`）；
- 输出统计汇总（成功/失败/错误数）。

**用法：**

```bash
# 默认配置：本地服务器 127.0.0.1:5000，8 个并发请求
python scripts/test.py

# 自定义并发数和服务器地址
python scripts/test.py --concurrency 16 --host 192.168.1.100 --port 8080

# 指定不同的测试图片和预期输出
python scripts/test.py \
    --image sample/wlyy/a_0.jpg \
    --expected output/wlyy/a_0.html

# 调整超时时间
python scripts/test.py \
    --health-timeout 60 \
    --request-timeout 120
```

**参数说明：**

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--host` | `0.0.0.0` | 服务器绑定地址（自动调整为 `127.0.0.1` 用于客户端连接） |
| `--port` | `5000` | 服务器端口 |
| `--concurrency` | `8` | 并发请求数 |
| `--image` | `sample/contract.jpg` | 测试图片路径 |
| `--expected` | `output/contract.html` | 预期 HTML 输出路径 |
| `--health-timeout` | `120` | 等待服务器就绪的超时时间（秒） |
| `--request-timeout` | `60` | 单个请求的超时时间（秒） |

**返回值：**

- `0`：所有请求成功且输出匹配预期
- `2`：测试图片或预期输出文件不存在
- `3`：服务器未在规定时间内就绪
- `4`：存在请求错误或输出不匹配
