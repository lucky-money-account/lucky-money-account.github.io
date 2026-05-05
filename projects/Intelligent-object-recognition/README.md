# 智能对象识别系统

基于深度学习的多模式图像识别系统，集成 PyTorch 与 TensorFlow/Keras，提供美观的可视化操作页面，支持四种识别模式应对不同场景需求。

## 功能特性

- **四种识别模式**：通用物体识别、数字识别、动物识别、场景识别，一键切换
- **深度模型驱动**：PyTorch MobileNetV2 (ImageNet) + Keras CNN (MNIST)，本地推理无需联网
- **可视化操作页面**：深色渐变主题、拖拽上传、实时置信度可视化
- **秒级启动**：模型后台异步加载，服务即开即用
- **环境隔离**：独立 Anaconda 虚拟环境，零污染全局 Python 环境

## 技术栈

| 层次 | 技术 |
|------|------|
| 前端 | HTML5 + CSS3 + 原生 JavaScript |
| 后端 | Flask 3.x |
| 通用识别 | YOLOv8n 目标检测 + MobileNetV2 分类 |
| 数字识别 | TensorFlow 2.21 + Keras CNN |
| 目标检测 | Ultralytics YOLOv8 |
| 图像处理 | Pillow + OpenCV |
| 科学计算 | NumPy + SciPy |

## 系统架构

```
浏览器 (index.html)
    │
    ▼
Flask Server (backend/app.py)
    │
    ├── /api/status    → 模型加载状态查询
    ├── /api/modes     → 获取可用识别模式
    └── /api/predict   → 图片识别接口
            │
            ├── 通用识别 → YOLOv8n 检测目标数
            │       ├── 单目标 → MobileNetV2 精细分类(1000类)
            │       └── 多目标 → YOLO 检测 + 边界框标注
            ├── 动物/场景 → MobileNetV2 + 关键词过滤
            └── 数字识别  → Keras CNN + 自动分割
```

## 环境要求

- Anaconda（推荐）或 Python 3.10+
- Windows / Linux / macOS
- CPU 即可运行，无需 GPU

## 安装

```bash
# 1. 创建并激活虚拟环境
conda create -n objrec python=3.10 -y
conda activate objrec

# 2. 安装依赖
pip install -r requirements.txt

# 3. 训练数字识别模型（如已训练可跳过）
python backend/train_digits.py
```

## 启动

```bash
conda activate objrec
python run.py
```

启动后打开浏览器访问 **http://127.0.0.1:5000**

也可直接双击 `run.bat` 启动。

## 项目结构

```
智能对象识别/
├── run.py                    # 启动入口
├── run.bat                   # Windows 一键启动
├── requirements.txt          # Python 依赖
├── backend/
│   ├── __init__.py
│   ├── app.py                # Flask API 路由
│   ├── config.py             # 配置项与关键词分类表
│   ├── models.py             # 模型加载、预处理、预测
│   └── train_digits.py       # MNIST 数字模型训练脚本
├── frontend/
│   ├── index.html            # 可视化操作页面
│   ├── css/
│   │   └── style.css         # 深色渐变主题样式
│   └── js/
│       └── app.js            # 前端交互与 API 调用
├── saved_models/
│   └── digit_model.h5        # 训练好的数字识别模型
├── data/                     # 训练数据目录
└── uploads/                  # 临时上传目录（自动清理）
```

## 识别模式详解

### 1. 通用物体识别
- 模型：YOLOv8n (目标检测) + PyTorch MobileNetV2 (分类)
- 自动判断图中是单物体还是多物体
- 单物体：MobileNetV2 精细分类 1000 类 ImageNet
- 多物体：YOLOv8n 检测 + 边界框标注，返回带框图像

### 2. 数字识别
- 模型：Keras CNN，在 MNIST 数据集上训练，测试准确率 **99%+**
- 覆盖：手写数字 0-9
- 场景：手写数字识别、验证码、表单识别

### 3. 动物识别
- 基于通用模型，通过关键词过滤（200+ 动物类名）筛选动物相关预测
- 覆盖：哺乳动物、鸟类、鱼类、爬行动物、昆虫等
- 场景：宠物识别、野生动物分类

### 4. 场景识别
- 基于通用模型，通过关键词过滤筛选场景相关预测
- 覆盖：自然景观（山水、海滩）、城市场景（街道、建筑）、室内场景
- 场景：风景分类、地标识别

## API 接口

### `GET /api/status`
获取模型加载状态。

```json
{
  "loaded": true,
  "loading": false,
  "pytorch": true,
  "digit": true
}
```

### `GET /api/modes`
获取所有可用识别模式列表。

### `POST /api/predict`
上传图片进行识别。

| 参数 | 类型 | 说明 |
|------|------|------|
| image | File | 图片文件 (jpg/png/gif/bmp/webp) |
| mode | String | 识别模式: general/digit/animal/scene |

```json
{
  "mode": "general",
  "top_label": "golden retriever",
  "top_confidence": 0.9532,
  "results": [
    {"label": "golden retriever", "confidence": 0.9532},
    {"label": "Labrador retriever", "confidence": 0.0314}
  ]
}
```

## 训练

数字识别模型使用 MNIST 数据集（6 万张训练图 + 1 万张测试图）：

```bash
python backend/train_digits.py
```

训练输出：
```
==================================================
MNIST 数字识别模型训练
==================================================
[1/4] 加载 MNIST 数据集...
  训练集: 60000 张图片
  测试集: 10000 张图片
[2/4] 创建模型...
[3/4] 训练模型 (10 epochs)...
[4/4] 评估模型...
  测试准确率: 99.17%
模型已保存到: saved_models/digit_model.h5
```

## 注意事项

- 首次运行通用识别模式时，PyTorch 会自动下载 MobileNetV2 权重（~14MB），使用清华镜像源可加速
- Flask 运行在 5000 端口，确保端口未被占用
- 上传的图片会在识别完成后自动删除，不会持久化存储
- 所有依赖安装在 `objrec` 虚拟环境中，不影响全局 Python

## License

MIT
