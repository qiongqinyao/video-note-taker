"""
第三阶段 V2：结构化文本 → 知识卡片式学习笔记
引擎：DeepSeek V4 API
核心改进：卡片组装模式替代章节作文模式，每张卡片独立生成
用法：python generate_notes_v2.py
"""

from openai import OpenAI
from pathlib import Path
from datetime import date
import json
import os
import time
from urllib.parse import quote
from concurrent.futures import ThreadPoolExecutor, as_completed

# 自动加载 .env 文件
def _load_dotenv():
    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.exists():
        return
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip()
            if key and key not in os.environ:
                os.environ[key] = value
_load_dotenv()

# ============================================================
# 配置
# ============================================================
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
QWEN_API_KEY = os.environ.get("QWEN_API_KEY", "")
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "deepseek")  # deepseek / qwen
DETAIL_LEVEL = "standard"
# ============================================================

if LLM_PROVIDER == "qwen":
    client = OpenAI(
        api_key=QWEN_API_KEY,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    MODEL_NAME = os.environ.get("QWEN_MODEL", "qwen-plus")
else:
    client = OpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url="https://api.deepseek.com",
    )
    MODEL_NAME = "deepseek-chat"


def api_call_with_retry(fn, max_retries=3, base_delay=2):
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            wait = base_delay * (2 ** attempt)
            print(f"  [重试] API 调用失败（{e}），{wait}秒后重试（{attempt + 1}/{max_retries - 1}）...")
            time.sleep(wait)


# ============================================================
# Prompt 1：知识卡片清单分析
# ============================================================
CARD_ANALYSIS_PROMPT = """你是一位课程内容分析师。你的任务是将网课转录文本拆分为一组「知识卡片」。

每张知识卡片是一个可独立阅读的知识单元，读者不需要看前面的卡片就能看懂这一张。

## 卡片类型

- concept：定义/概念/术语。一张卡片讲一个核心概念。
- process：算法/步骤/过程。一张卡片讲一个完整的操作流程。
- comparison：两种方法的对比。一张卡片放一个对比表格。
- warning：易错点/注意事项。一张卡片一个警示点。

## 拆分原则

1. 每张卡片只聚焦一个知识点，不跨概念
2. 同一个具体例子（如某一组数组合并过程）只能出现在一张卡片中
3. 每张卡片标注视频时间范围：start_time（起始秒数）、end_time（结束秒数）
4. 卡片总数 3-8 张，由实际内容决定，不要凑数也不要遗漏
5. 每张卡片给一个 label——一句话灵魂标签，说清楚这个知识点的核心
6. 禁止「总览卡」：第一张卡片不能是对后续所有内容的概括
7. 每张卡片必须是独立的知识单元，两张卡片的内容不重叠

## 对比卡规则（强制）

如果视频中出现了同一概念的不同实现方式/方法/变体（如自底向上 vs 自顶向下、链表 vs 数组、RIP vs OSPF），在逐一讲完每种之后，**必须**额外生成一张 comparison 类型卡片做对比。
- 即使视频只简单提了一句区别，也要生成对比卡
- 对比卡放在所有被对比的卡片之后

## 层级分组规则（强制）

以下情况 **必须** 使用 group 字段将多张卡片归入同一个上级主题：

1. **同一类别的并列项**：如 LeNet-5、AlexNet、Inception 都是「常见 CNN 模型」
2. **同一概念的多个维度**：如「时间复杂度」和「空间复杂度」都属于「复杂度分析」
3. **同一操作的多种实现**：如「自底向上」和「自顶向下」都属于归并排序的「两种实现方式」

group 字段内的子卡片必须是**同一类别、同一层级**的并列项，不要为了分组强行把不相关的东西塞在一起。并列项数量 ≥ 2 时就必须分组。

## 输出 JSON

{
  "course_title": "课程标题（10字以内）",
  "cards": [
    {
      "type": "concept",
      "title": "归并操作",
      "label": "每次比较两个有序数组的最小元素，取较小者放入结果",
      "start_time": 18,
      "end_time": 161,
      "summary": "归并操作的定义、步骤和一个具体示例"
    },
    {
      "type": "concept",
      "title": "时间复杂度",
      "label": "归并轮数 log n × 每轮 O(n) = O(n log n)",
      "group": "复杂度分析",
      "start_time": 449,
      "end_time": 472,
      "summary": "推导归并排序的时间复杂度 O(n log n)"
    }
  ]
}"""

# ============================================================
# 卡片系统 Prompt（公共部分）
# ============================================================
CARD_SYSTEM_PROMPT = """你是一位学习笔记整理师。你的任务是为一个知识点生成一张独立的卡片。

## 铁律

1. **独立完整**：这张卡片必须可以单独阅读，不依赖任何其他卡片
2. **不写元描述**：禁止「老师讲到」「视频中说」「接下来」「前面说过」「我们学习了」
3. **重新组织语言**：不要照搬转录原文的措辞，用自己的语言凝练
4. **每句话都要有信息量**：删除所有铺垫、过渡、闲聊
5. **只写本卡片的内容**：不要把其他知识点塞进来，卡片标题已经限定了范围
6. **禁止「核心要点」和「小结」段落**：卡片本身就是最小知识单元

## 标题格式（必须严格遵守）

- 标题只用 ## 纯文字，例如 ## 归并操作
- **绝对禁止**在标题行加：时间戳、方括号标签如 [concept]、反引号、HTML 标签、任何装饰符号
- 标题只包含知识点的中文名称，其他什么都不加

## 正文格式

- 第一行正文用 > 📍 视频时段：MM:SS - MM:SS 标注时间段
- 紧接着写实质内容
- 代码块必须标注语言：```pseudocode 或 ```python 或 ```java 等
- 用 **粗体** 标记首次出现的核心术语
- 用 > 引用块放关键结论或公式
- 用 ⚠️ 标记易错点
- 禁止用 ⭐
- 中文和英文/数字之间加空格"""


def build_card_user_prompt(card_info, transcript_segment):
    """根据卡片类型构建 user prompt，按 DETAIL_LEVEL 调整"""
    title = card_info.get("title", "未命名")
    label = card_info.get("label", "")
    start = card_info.get("start_time", 0)
    end = card_info.get("end_time", 0)
    card_type = card_info.get("type", "concept")
    summary = card_info.get("summary", "")
    group = card_info.get("group", "")
    heading_prefix = "###" if group else "##"

    ts_start = f"{int(start // 60):02d}:{int(start % 60):02d}"
    ts_end = f"{int(end // 60):02d}:{int(end % 60):02d}"

    # 模式提示
    mode_map = {
        "minimal": "极简模式：只写定义和关键结论，不举例子、不写伪代码、不生成对比卡。",
        "standard": "推荐模式：一个例子、必要时写伪代码和对比卡，适可而止。",
        "detailed": "详细模式：比推荐模式稍多一些展开，类比推荐模式多一两句解释即可，不冗余。",
    }
    mode_hint = mode_map.get(DETAIL_LEVEL, mode_map["standard"])

    header = f"""## 当前知识点信息

- 标题：{title}
- 类型：{card_type}
- 灵魂标签：{label}
- 内容概要：{summary}
- 视频时间：{ts_start} - {ts_end}
- 当前模式：{mode_hint}

## 原始转录文本（对应时间段）

{transcript_segment}"""

    if card_type == "concept":
        if DETAIL_LEVEL == "minimal":
            body = """（只写定义：1-2 句。不写例子。）

（如果有核心结论，用 > 引用块放这里。没有就省略）

（如果有易错点，用 ⚠️ 一句话。没有就省略）"""
        elif DETAIL_LEVEL == "detailed":
            body = """（定义段落：2-3 句，可以比推荐模式多一句解释或背景说明）

（如果有公式或关键结论，用 > 引用块放这里）

（如果有易错点，用 ⚠️ 一句话放这里。没有就省略）"""
        else:
            body = """（定义段落：1-2 句，不要超过 3 句）

（如果有公式或关键结论，用 > 引用块放这里）

（如果有易错点，用 ⚠️ 一句话放这里。没有就省略）"""

        template = f"""{header}

请按以下模板生成一张概念卡。标题只用纯文字，不加任何标签或时间戳：

{heading_prefix} {title}

> 📍 视频时段：{ts_start} - {ts_end}

> >> {label}

{body}"""
        return template

    elif card_type == "process":
        if DETAIL_LEVEL == "minimal":
            body = f"""{heading_prefix} {title}

> 📍 视频时段：{ts_start} - {ts_end}

> >> {label}

（核心思想：1 句）

（步骤：只列关键步骤，每步一行，3-5 步）

（不要写示例，不要写伪代码）

> （关键结论，用引用块）"""
        elif DETAIL_LEVEL == "detailed":
            body = f"""{heading_prefix} {title}

> 📍 视频时段：{ts_start} - {ts_end}

> >> {label}

（核心思想：2-3 句讲清楚本质和背景）

（步骤：用有序列表，每步一行，具体可执行）

（示例：使用转录文本中老师演示的具体数据，逐步演示）

（伪代码：如果本卡片是算法/操作过程，写一段经典伪代码。```pseudocode 标记。注释比推荐模式稍详细。如果内容不是算法，跳过此段）

> （关键结论或公式，用引用块）

（如果有易错点，用 ⚠️ 一句话。没有就省略）"""
        else:
            body = f"""{heading_prefix} {title}

> 📍 视频时段：{ts_start} - {ts_end}

> >> {label}

（核心思想：1-2 句讲清楚这个过程的本质）

（步骤：用有序列表，每步一行，3-6 步。每一步写得具体、可执行）

（示例：使用转录文本中老师演示的具体数据，逐步演示计算过程。如果老师用的是一组数，就还原这组数）

（伪代码：如果本卡片描述的是一个算法/操作过程，用经典教科书风格写一段伪代码。必须用 ```pseudocode 标记语言类型。格式要求：
- 函数名用英文大写，如 MERGE(A, B)、MERGE_SORT(arr)
- 不加语言特定的语法噪音（不加 public static、不加类型声明、不加分号）
- 用缩进表示层级
- 变量名用英文简写，如 left, right, mid, tmp
- 行内用 // 加简短注释
- 如果本卡片内容不是算法（如纯概念讲解、网络协议过程），不写伪代码，跳过此段）

> （关键结论或公式，用引用块）

（如果有易错点，用 ⚠️ 一句话。没有就省略）"""

        template = f"""{header}

请按以下模板生成一张过程卡。标题只用纯文字，不加任何标签或时间戳：

{body}"""
        return template

    elif card_type == "comparison":
        body = f"""{heading_prefix} {title}

> 📍 视频时段：{ts_start} - {ts_end}

> >> {label}

（1-2 句说明对比背景：为什么要对比这两种方法/实现？）

| 维度 | （方法A名） | （方法B名） |
|------|-----------|-----------|
| 核心思路 | ... | ... |
| 实现方式 | ... | ... |
| （如有更多维度，最多 {'6' if DETAIL_LEVEL == 'detailed' else '4'} 行） | ... | ... |

> （关键结论：什么时候用 A、什么时候用 B？）"""

        template = f"""{header}

请按以下模板生成一张对比卡。标题只用纯文字，不加任何标签或时间戳：

{body}"""
        return template

    elif card_type == "warning":
        if DETAIL_LEVEL == "minimal":
            body = f"""{heading_prefix} {title}

> 📍 视频时段：{ts_start} - {ts_end}

> ⚠️ {label}

（一句话说清易错点）"""
        else:
            body = f"""{heading_prefix} {title}

> 📍 视频时段：{ts_start} - {ts_end}

> ⚠️ {label}

（说清楚哪里容易出错，1-2 句）

**正确做法**：（一句话纠正）

（{'2-3 句补充说明' if DETAIL_LEVEL == 'detailed' else '1-2 句补充说明'}，解释为什么容易犯这个错误）"""

        template = f"""{header}

请按以下模板生成一张警示卡。标题只用纯文字，不加任何标签或时间戳：

{body}"""
        return template

    else:
        # 未知类型回退为概念卡
        return build_card_user_prompt({**card_info, "type": "concept"}, transcript_segment)


def auto_group_cards(result):
    """自动检测同类卡片并强制归组（LLM 分组不可靠时的兜底）"""
    cards = result.get("cards", [])
    if not cards:
        return result

    # 全量扫描：检测「时间+空间」配对
    time_idx = space_idx = None
    for i, c in enumerate(cards):
        if c.get("group"):
            continue
        t = c.get("title", "")
        if "时间" in t:
            time_idx = i
        if "空间" in t:
            space_idx = i
    if time_idx is not None and space_idx is not None:
        cards[time_idx]["group"] = "复杂度分析"
        cards[space_idx]["group"] = "复杂度分析"

    # 全量扫描：检测「自底+自顶」配对
    bottom_idx = top_idx = None
    for i, c in enumerate(cards):
        if c.get("group"):
            continue
        t = c.get("title", "")
        if "自底" in t:
            bottom_idx = i
        if "自顶" in t:
            top_idx = i
    if bottom_idx is not None and top_idx is not None:
        cards[bottom_idx]["group"] = "两种实现方式"
        cards[top_idx]["group"] = "两种实现方式"

    # 尾部检测：最后 3+ 张短标题同类型卡片 → 常见模型/变体
    for tail_start in range(len(cards) - 2, -1, -1):
        tail = cards[tail_start:]
        if len(tail) < 3:
            continue
        if any(c.get("group") for c in tail):
            continue
        if any(c.get("type") != tail[0].get("type") for c in tail):
            continue
        if all(len(c.get("title", "")) <= 20 for c in tail):
            # 最后一截短标题同类卡片，归组
            for c in tail:
                c["group"] = "常见模型"
            break

    return result


def analyze_cards(transcript_text):
    """分析转录文本，输出知识卡片清单"""
    print("=" * 50)
    print("  [Step 1/3] 正在分析知识点结构...")
    print("=" * 50)
    print()

    # 极简模式提示
    mode_note = ""
    if DETAIL_LEVEL == "minimal":
        mode_note = "\n\n当前为极简模式，请不要生成 comparison 类型的卡片，也不要对知识点做过多细分。"

    try:
        response = api_call_with_retry(
            lambda: client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": CARD_ANALYSIS_PROMPT + mode_note},
                    {"role": "user", "content": f"请分析以下课程转录文本：\n{transcript_text}"},
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
                max_tokens=4000,
            )
        )

        result = json.loads(response.choices[0].message.content)

        course_title = result.get("course_title", "未知课程")
        cards = result.get("cards", [])

        print(f"  课程标题: {course_title}")
        print(f"  卡片数量: {len(cards)} 张")
        for i, card in enumerate(cards, 1):
            ct = card.get("type", "concept")
            t = card.get("title", "未知")
            lb = card.get("label", "")
            print(f"    {i}. [{ct}] {t}")
            if lb:
                print(f"       >> {lb}")
        print()
        print("  卡片分析完成！")
        print()

        # 自动分组兜底：LLM 未设置 group 时，按规则强制归组
        result = auto_group_cards(result)

        return result

    except json.JSONDecodeError as e:
        print(f"  [错误] JSON 解析失败: {e}")
        return None
    except Exception as e:
        print(f"  [错误] API 调用失败: {e}")
        return None


def extract_time_segment(full_text, start_time, end_time):
    """根据时间戳截取转录文本片段"""
    lines = full_text.split("\n")
    segment_lines = []
    for line in lines:
        if line.startswith("[") and "]" in line[:10]:
            try:
                time_str = line[1:].split("]")[0]
                parts = time_str.split(":")
                seconds = int(parts[0]) * 60 + int(parts[1])
                if start_time <= seconds < end_time:
                    segment_lines.append(line)
                elif seconds >= end_time:
                    break
            except Exception:
                segment_lines.append(line)
        else:
            segment_lines.append(line)
    return "\n".join(segment_lines) if segment_lines else full_text


def generate_card(card_info, full_transcript):
    """生成单张知识卡片"""
    card_type = card_info.get("type", "concept")
    title = card_info.get("title", "未命名")
    start = card_info.get("start_time", 0)
    end = card_info.get("end_time", 0)

    segment = extract_time_segment(full_transcript, start, end)
    user_prompt = build_card_user_prompt(card_info, segment)

    token_map = {"minimal": 1500, "standard": 3000, "detailed": 5000}
    max_tokens = token_map.get(DETAIL_LEVEL, 3000)

    try:
        response = api_call_with_retry(
            lambda: client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": CARD_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.5,
                max_tokens=max_tokens,
            )
        )
        return response.choices[0].message.content

    except Exception as e:
        print(f"  [错误] 卡片「{title}」生成失败: {e}")
        return f"## {title}\n\n> >> {card_info.get('label', '')}\n\n> 卡片生成出错: {e}\n"


def assemble_cards(structure, card_contents, metadata=None):
    """组装：YAML Front Matter + Obsidian wiki-link TOC + 知识卡片"""
    course_title = structure.get("course_title", "未命名课程")

    if metadata:
        original_title = metadata.get("original_title", "")
        source = metadata.get("source_url") or metadata.get("source", "本地视频")
        uploader = metadata.get("uploader", "")
    else:
        original_title = ""
        source = "本地视频"
        uploader = ""

    parts = []

    # YAML Front Matter
    parts.append("---")
    parts.append(f"title: {course_title}")
    parts.append(f"date: {date.today().isoformat()}")
    parts.append(f"source: {source}")
    if uploader:
        parts.append(f"uploader: {uploader}")
    if original_title:
        parts.append(f"original_title: {original_title}")
    parts.append("tags: []")
    parts.append("---")
    parts.append("")

    # Title
    parts.append(f"# {course_title}")
    parts.append("")
    source_desc = source if source.startswith("http") else "本地视频"
    parts.append(f"> 本文由 AI 自动生成，基于 {source_desc} 转录文本整理。每张卡片可独立阅读。")
    parts.append("")
    parts.append("---")
    parts.append("")

    # TOC — standalone & group labels share flat numbering, sub-items indented
    card_list = structure.get("cards", [])
    last_group = None
    toc_num = 0
    for card_info in card_list:
        title = card_info.get("title", "未知")
        group = card_info.get("group", "")
        anchor = quote(title, safe='')
        if group and group != last_group:
            toc_num += 1
            parts.append(f"{toc_num}. {group}")
            last_group = group
        elif not group:
            toc_num += 1
            parts.append(f"{toc_num}. [{title}](#{anchor})")
        if group:
            parts.append(f"    - [{title}](#{anchor})")
    parts.append("")
    parts.append("---")
    parts.append("")

    # Cards — handle groups: insert ## group heading, ### for sub-cards
    last_group = None
    for i, (card_info, card_content) in enumerate(zip(card_list, card_contents), 1):
        group = card_info.get("group", "")

        # Insert group heading when group starts
        if group and group != last_group:
            parts.append(f"## {group}")
            parts.append("")
            last_group = group
        elif not group:
            last_group = None

        fixed_lines = []
        for line in card_content.strip().split("\n"):
            # 统一降级：确保不超过 ##
            if line.startswith("# ") and not line.startswith("## ") and not line.startswith("### "):
                line = "##" + line[1:] if not group else "###" + line[1:]
            if line.startswith("## ") and group:
                # 有 group 的卡片标题降为 ###
                line = "#" + line
            # 删除标题行中的时间戳残余
            if (line.startswith("## ") or line.startswith("### ")) and "`[" in line:
                import re
                line = re.sub(r'\s*`\[[^]]*\]`', '', line).rstrip()
            # 删除 HTML 标签残余
            if "<a " in line or "<span " in line:
                import re
                line = re.sub(r'\s*<(a|span)\s[^>]*></(a|span)>', '', line).rstrip()
                if not line.strip():
                    continue
            fixed_lines.append(line)
        parts.append("\n".join(fixed_lines))
        parts.append("")

        # 卡片间分隔
        if i < len(card_list):
            parts.append("---")
            parts.append("")

    return "\n".join(parts)


# ============================================================
# 主程序入口
# ============================================================
if __name__ == "__main__":
    print()
    print("=" * 56)
    print("   第三阶段 V2：文本 → 知识卡片式笔记")
    print("   引擎: DeepSeek V4 | 模式: 卡片组装")
    print(f"   详细程度: {DETAIL_LEVEL}")
    print("=" * 56)
    print()

    if LLM_PROVIDER != "qwen" and not DEEPSEEK_API_KEY:
        print("[错误] 未设置 DEEPSEEK_API_KEY")
        print("   方法1：创建 .env 文件，写入 DEEPSEEK_API_KEY=你的key")
        print("   方法2：在终端执行 set DEEPSEEK_API_KEY=你的key")
        print("   方法3：改用免费千问 → .env 设置 LLM_PROVIDER=qwen + QWEN_API_KEY=你的key")
        exit(1)
    if LLM_PROVIDER == "qwen" and not QWEN_API_KEY:
        print("[错误] 未设置 QWEN_API_KEY")
        print("   去 https://dashscope.console.aliyun.com/apiKey 获取 Key（阿里云账号登录）")
        print("   然后 .env 中写入 QWEN_API_KEY=你的key")
        exit(1)

    transcript_name = "test_plain.txt"
    transcript_path = f"output/transcripts/{transcript_name}"

    if not Path(transcript_path).exists():
        print(f"[错误] 找不到转录文件 {transcript_path}")
        exit(1)

    print(f"[读取] 正在加载转录文件: {transcript_name}")
    with open(transcript_path, "r", encoding="utf-8") as f:
        full_transcript = f.read()

    word_count = len(full_transcript)
    print(f"   总字数: {word_count} 字")
    print()

    # Step 1：分析卡片结构
    structure = analyze_cards(full_transcript)
    if structure is None:
        print("[失败] 卡片分析出错。")
        exit(1)

    card_list = structure.get("cards", [])
    if not card_list:
        print("[失败] 未检测到任何知识点。")
        exit(1)

    # Step 2：逐卡片生成
    print("=" * 50)
    print("  [Step 2/3] 正在逐卡片生成...")
    print("=" * 50)
    print()

    # 并行生成所有卡片（最多 4 并发）
    print(f"  并行生成 {len(card_list)} 张卡片（最多 4 并发）...")
    print()
    card_results = {}  # index → content
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(generate_card, card_info, full_transcript): i
            for i, card_info in enumerate(card_list)
        }
        for future in as_completed(futures):
            idx = futures[future]
            card_info = card_list[idx]
            ct = card_info.get("type", "concept")
            title = card_info.get("title", f"卡片{idx + 1}")
            try:
                content = future.result()
            except Exception as e:
                content = f"## {title}\n\n> 生成失败: {e}\n"
            card_results[idx] = content
            print(f"  ({idx + 1}/{len(card_list)}) [{ct}] {title} — {len(content)} 字")

    card_contents = [card_results[i] for i in range(len(card_list))]
    print()
    print("  所有卡片生成完毕！")
    print()

    # Step 3：组装
    print("=" * 50)
    print("  [Step 3/3] 正在组装最终笔记...")
    print("=" * 50)
    print()

    final_note = assemble_cards(structure, card_contents)

    # Save
    note_path = "output/notes/test_v3.md"
    Path("output/notes").mkdir(parents=True, exist_ok=True)

    with open(note_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(final_note)

    print(f"  笔记已保存: {note_path}")
    print(f"  笔记总字数: {len(final_note)} 字")
    print(f"  卡片数量: {len(card_contents)} 张")
    print()

    print("=" * 56)
    print("   全部完成！")
    print("=" * 56)
    print()
    print(f"  输出文件: {note_path}")
    print("=" * 56)
