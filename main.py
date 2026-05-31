"""
一键执行：URL 或本地视频 → 知识卡片式学习笔记
自动检测输入类型，串行执行三阶段流水线
用法：python main.py
     python main.py "https://www.bilibili.com/video/BV1xx411c7mD"
     python main.py "uploads/test.mp4"
     python main.py --level minimal "url"
     python main.py --level detailed
"""

import sys
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from download import download_from_url
from extract_audio import extract_audio
from transcribe_audio import transcribe_audio, save_transcript
from generate_notes_v2 import (
    analyze_cards,
    generate_card,
    assemble_cards,
    DETAIL_LEVEL,
    DEEPSEEK_API_KEY,
    QWEN_API_KEY,
    LLM_PROVIDER,
)

LEVEL_LABELS = {
    "minimal": "极简模式",
    "standard": "推荐模式",
    "detailed": "详细模式",
}


def detect_input_type(user_input):
    """判断输入类型"""
    if user_input.startswith("http://") or user_input.startswith("https://"):
        return "url"
    lower = user_input.lower()
    for ext in (".mp4", ".mov", ".mkv", ".avi", ".flv", ".webm"):
        if lower.endswith(ext):
            return "local"
    return "invalid"


def main():
    parser = argparse.ArgumentParser(description="AI 网课笔记助手")
    parser.add_argument(
        "input", nargs="?", default=None,
        help="视频 URL 或本地视频路径"
    )
    parser.add_argument(
        "--level", choices=["minimal", "standard", "detailed"],
        default="standard",
        help="笔记详细程度（默认: standard 推荐模式）"
    )
    args = parser.parse_args()

    # 应用用户选择的详细程度
    global DETAIL_LEVEL
    DETAIL_LEVEL = args.level

    print("=" * 56)
    print("  AI 网课笔记助手 — 知识卡片式笔记")
    print("  支持: URL（B站/YouTube）| 本地视频文件")
    print(f"  笔记模式: {LEVEL_LABELS.get(DETAIL_LEVEL, DETAIL_LEVEL)}")
    print("=" * 56)
    print()

    if LLM_PROVIDER != "qwen" and not DEEPSEEK_API_KEY:
        print("[错误] 未设置 DEEPSEEK_API_KEY")
        print("   方法1：创建 .env 文件，写入 DEEPSEEK_API_KEY=你的key")
        print("   方法2：在终端执行 set DEEPSEEK_API_KEY=你的key")
        print("   方法3：改用免费千问 → .env 设置 LLM_PROVIDER=qwen + QWEN_API_KEY=你的key")
        return
    if LLM_PROVIDER == "qwen" and not QWEN_API_KEY:
        print("[错误] 未设置 QWEN_API_KEY")
        print("   去 https://dashscope.console.aliyun.com/apiKey 获取 Key（阿里云账号登录）")
        return

    user_input = args.input
    if not user_input:
        user_input = input("请输入视频 URL 或本地视频路径: ").strip()

    if not user_input:
        print("[错误] 未输入任何内容")
        return

    input_type = detect_input_type(user_input)

    if input_type == "invalid":
        print("[错误] 无法识别输入类型")
        print("  支持: http:// 或 https:// 开头的 URL")
        print("  支持: .mp4 .mov .mkv .avi .flv .webm 结尾的本地文件")
        return

    metadata = None
    audio_path = None

    # ================================================================
    # 阶段 1：获取音频
    # ================================================================
    if input_type == "url":
        print(f"[模式] URL 下载模式")
        print(f"   URL: {user_input}")
        print()

        result = download_from_url(user_input)
        if result is None:
            print("[失败] 下载阶段失败，流程终止。")
            return

        audio_path = result["audio_path"]
        metadata = {
            "title": result["title"],
            "uploader": result["uploader"],
            "source_url": result["source_url"],
            "source": result["source_url"],
        }

    elif input_type == "local":
        print(f"[模式] 本地视频模式")
        print(f"   文件: {user_input}")
        print()

        audio_path = extract_audio(user_input)
        if audio_path is None:
            print("[失败] 音频提取失败，流程终止。")
            return

        video_file = Path(user_input)
        metadata = {
            "title": video_file.stem,
            "uploader": "",
            "source_url": "",
            "source": "本地视频",
        }

    print("[阶段 1/3] 音频准备完成")
    print()

    # ================================================================
    # 阶段 2：语音转文字
    # ================================================================
    print("=" * 56)
    print("  阶段 2/3：语音识别")
    print("=" * 56)
    print()

    audio_file = Path(audio_path)
    plain_path = f"output/transcripts/{audio_file.stem}_plain.txt"

    # 转录缓存：已有则跳过
    if Path(plain_path).exists():
        print(f"[跳过] 转录文件已存在，直接使用: {Path(plain_path).name}")
        print()
    else:
        segments = transcribe_audio(audio_path)
        if segments is None:
            print("[失败] 语音识别失败，流程终止。")
            return
        plain_path = save_transcript(segments, audio_file.name)

    print("[阶段 2/3] 语音识别完成")
    print()

    # ================================================================
    # 阶段 3：卡片式笔记生成
    # ================================================================
    print("=" * 56)
    print("  阶段 3/3：生成知识卡片笔记")
    print("=" * 56)
    print()

    with open(plain_path, "r", encoding="utf-8", newline="\n") as f:
        full_transcript = f.read()

    print(f"[读取] 转录文本: {Path(plain_path).name}")
    print(f"   总字数: {len(full_transcript)} 字")
    print()

    # 3a：分析卡片结构
    structure = analyze_cards(full_transcript)
    if structure is None:
        print("[失败] 卡片分析失败，流程终止。")
        return

    card_list = structure.get("cards", [])
    if not card_list:
        print("[失败] 未检测到任何知识点。")
        return

    # LLM 生成的短标题用于文档，视频原标题记入 metadata
    if metadata:
        metadata["original_title"] = metadata.get("title", "")

    # 3b：逐卡片生成
    print("=" * 50)
    print("  逐卡片生成...")
    print("=" * 50)
    print()

    # 并行生成所有卡片（最多 4 并发）
    print(f"  并行生成 {len(card_list)} 张卡片（最多 4 并发）...")
    print()
    card_results = {}
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

    # 3c：组装
    print("=" * 50)
    print("  组装最终笔记...")
    print("=" * 50)
    print()

    final_note = assemble_cards(structure, card_contents, metadata)

    # 保存——用 LLM 短标题，而非视频长标题
    safe_title = structure.get("course_title", "笔记")
    for ch in r'<>:"/\|?*':
        safe_title = safe_title.replace(ch, "_")
    safe_title = safe_title.strip(". ")

    note_path = f"output/notes/{safe_title}.md"
    Path("output/notes").mkdir(parents=True, exist_ok=True)

    with open(note_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(final_note)

    print(f"  笔记已保存: {note_path}")
    print(f"  笔记总字数: {len(final_note)} 字")
    print(f"  卡片数量: {len(card_contents)} 张")
    print()

    print("=" * 56)
    print("  全部完成！")
    print("=" * 56)
    print()
    print(f"  输出文件: {note_path}")
    if metadata and metadata.get("source_url"):
        print(f"  原始来源: {metadata['source_url']}")
    print(f"  笔记模式: {LEVEL_LABELS.get(DETAIL_LEVEL, DETAIL_LEVEL)}")
    print("=" * 56)


if __name__ == "__main__":
    main()
