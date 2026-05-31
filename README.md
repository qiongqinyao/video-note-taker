# AI 网课笔记助手

> 不只是总结视频说了什么，而是把视频变成一份可以拿来复习的学习笔记。

输入 B站/YouTube 链接或本地视频，自动生成**知识卡片式 Markdown 笔记**——有层级、有对比表、有伪代码、带视频时间戳，直接打开就能学。

## 效果预览

笔记输出为结构化的知识卡片，每张卡片独立可读：

```markdown
## 归并操作

> 📍 视频时段：00:18 - 02:41

> 每次比较两个有序数组的最小元素，取较小者放入结果

**归并操作**是将两个已经有序的数组合并成一个有序数组的过程...

## 复杂度分析
### 时间复杂度
### 空间复杂度

| 维度 | 自底向上 | 自顶向下 |
|------|---------|---------|
| 核心思路 | 从单个元素开始 | 递归二分 |
...
```

## 功能特性

- 🔗 **B站/YouTube/本地视频** — 粘贴链接或本地文件，自动下载音频
- 🎯 **知识卡片式笔记** — 按概念/过程/对比/警示四种类型拆分
- 📊 **自动对比表** — 检测到多种实现方式时强制生成对比
- 💻 **伪代码生成** — 算法类内容自动生成教科书风格伪代码
- ⏱️ **视频时间戳** — 每张卡片标注视频时段，随时回溯
- 📋 **三种模式** — 极简（知识点清单）/ 推荐 / 详细，按需切换
- ⚡ **并行生成** — 多张卡片同时生成，1小时视频 ~20分钟出结果

## 快速开始

### 环境要求

- Python 3.10+
- FFmpeg（系统级安装）
- NVIDIA GPU + CUDA（可选，用于本地语音识别加速）

### 安装

```bash
# 1. 克隆仓库
git clone https://github.com/yourname/video-note-taker.git
cd video-note-taker

# 2. 创建虚拟环境
python -m venv venv

# 3. 激活虚拟环境
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 4. 安装依赖
pip install -r requirements.txt

# 5. 配置 API Key
cp .env.example .env
# 编辑 .env，填入你的 DeepSeek API Key
```

### 安装 FFmpeg

- **Windows**: 下载 [ffmpeg.org](https://ffmpeg.org/download.html)，解压后将 `bin/` 加入系统 PATH
- **macOS**: `brew install ffmpeg`
- **Linux**: `sudo apt install ffmpeg`

### 配置 LLM（二选一）

**方案 A：通义千问 API（推荐，每月 100 万 token 免费）**

**获取 Key：**

1. 打开 [dashscope.console.aliyun.com](https://dashscope.console.aliyun.com/apiKey)
2. 用支付宝/淘宝/阿里云账号登录
3. 点击「创建 API Key」→ 复制保存

**配置：** 在 `.env` 中写入：

```
LLM_PROVIDER=qwen
QWEN_API_KEY=你的key
```

默认使用 `qwen-plus` 模型（性价比最高）。百联千问后台有一百多个模型不用管，只用记三个：

| 模型 | 定位 | 适用场景 |
|------|------|---------|
| `qwen-turbo` | 快、免费额度耐用 | 大量视频批量处理 |
| `qwen-plus` ⭐ | 均衡（默认推荐） | 日常使用，质量速度兼顾 |
| `qwen-max` | 最强 | 复杂课程，要最好效果 |

**换模型：** 在 `.env` 加一行 `QWEN_MODEL=qwen-max`，不改则默认 `qwen-plus`。

> 100 万 token ≈ 30 个 15 分钟视频，个人学习完全够用。用完付费也很便宜。

**方案 B：DeepSeek API（付费，效果更好）**

1. 访问 [platform.deepseek.com](https://platform.deepseek.com) 注册并充值
2. 在 API Keys 页面创建 Key
3. `.env` 中设置：`DEEPSEEK_API_KEY=你的key`

费用很低，一个 15 分钟视频约 0.05-0.15 元。

## 使用

```bash
# 从 B站 链接生成笔记
python main.py "https://www.bilibili.com/video/BV1xx411c7mD"

# 从本地视频生成
python main.py "uploads/my_lecture.mp4"

# 选择模式
python main.py --level minimal "URL"    # 极简模式：只记核心要点
python main.py --level detailed "URL"   # 详细模式：完整展开
python main.py --level standard "URL"   # 推荐模式：默认
```

## 项目结构

```
video-note-taker/
├── main.py                  # 一键入口
├── download.py              # URL → 音频下载 + 元数据提取
├── extract_audio.py         # 本地视频 → 音频提取
├── transcribe_audio.py      # 语音转文字（faster-whisper）
├── generate_notes_v2.py     # 知识卡片生成（DeepSeek API）
├── requirements.txt
├── .env.example
├── LICENSE
├── uploads/                 # 本地视频放这里
└── output/
    ├── audio/               # 提取的音频文件
    ├── transcripts/         # 转录文本
    └── notes/               # 最终笔记（.md）
```

## 常见问题

**Q: 没有 NVIDIA 显卡能用吗？**
可以。修改 `transcribe_audio.py` 中 `device="cuda"` 为 `device="cpu"`，速度会慢一些。

**Q: 转录准确率不高怎么办？**
当前使用 faster-whisper medium 模型。如果追求更高准确率，可以换用 large 模型（需要更大显存），或者接入云端 ASR 服务。

**Q: 费用多少？**
千问模式每月 100 万 token 免费，约等于 30 个 15 分钟视频。DeepSeek 模式一个视频约 0.05-0.15 元。

## License

MIT
