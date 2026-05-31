"""
第二阶段：音频 → 文字（语音识别）
专为 GTX 1050 Ti 4GB 优化：medium 模型 + GPU + int8 量化
用法：python transcribe_audio.py
"""

from faster_whisper import WhisperModel
from pathlib import Path
import json


def transcribe_audio(audio_path):
    """
    把音频文件转成带时间戳的文字段列表

    参数:
        audio_path: 音频文件路径，例如 "output/audio/test.wav"

    返回:
        文字段列表，每个元素: {"start": 0.0, "end": 5.2, "text": "..."}
    """
    audio_file = Path(audio_path)

    # ===== 第1步：检查音频文件是否存在 =====
    if not audio_file.exists():
        print(f"[错误] 找不到音频文件 {audio_path}")
        print("   请先运行第一阶段脚本（extract_audio.py）生成音频")
        return None

    # ===== 第2步：加载模型 =====
    print("=" * 50)
    print("  加载 Whisper 模型")
    print("=" * 50)
    print()
    print(f"  模型大小: medium（适合 4GB 显存）")
    print(f"  运行设备: CUDA (GPU)")
    print(f"  计算精度: int8（量化，省显存）")
    print()
    print("  注意：第一次运行会下载模型文件（约 1.5GB），请耐心等待...")
    print()

    try:
        model = WhisperModel(
            "medium",                   # 模型大小
            device="cuda",             # 用你的 GTX 1050 Ti
            compute_type="int8",       # int8 量化，4GB 显存刚好够
        )
    except Exception as e:
        print(f"[错误] 模型加载失败: {e}")
        print()
        print("常见原因和解决办法：")
        print("   1. 显存不够 → 换成 model_size='small'")
        print("   2. CUDA 版本不兼容 → 重装 GPU 版 PyTorch")
        print("   3. faster-whisper 没装好 → 重新 pip install faster-whisper")
        return None

    print("  模型加载完成！")
    print()

    # ===== 第3步：执行语音识别 =====
    print("=" * 50)
    print("  正在识别语音...")
    print("=" * 50)
    print()
    print(f"  音频文件: {audio_file.name}")

    # 先看看音频有多长
    import wave
    try:
        with wave.open(str(audio_file), 'rb') as wf:
            duration_sec = wf.getnframes() / wf.getframerate()
        print(f"  音频时长: {duration_sec:.0f} 秒（约 {duration_sec/60:.1f} 分钟）")
        if duration_sec > 300:
            print(f"  预计耗时: {duration_sec/60*0.15:.0f} - {duration_sec/60*0.3:.0f} 分钟")
    except:
        pass

    print()
    print("  请耐心等待，不要关闭窗口...")
    print()

    try:
        segments, info = model.transcribe(
            str(audio_file),
            language="zh",                          # 指定中文
            beam_size=5,                            # beam search 宽度
            vad_filter=True,                        # 跳过静音段落
            vad_parameters=dict(
                min_silence_duration_ms=500,        # 超过 0.5 秒的静音跳过
            ),
            initial_prompt="这是一段中文网课视频的录音，内容涉及知识点讲解。",
        )
    except Exception as e:
        print(f"[错误] 语音识别出错: {e}")
        return None

    # ===== 第4步：显示检测结果 =====
    print()
    print("=" * 50)
    print("  检测结果")
    print("=" * 50)
    print()
    print(f"  检测语言: {info.language}（概率: {info.language_probability:.2%}）")
    print(f"  音频总长: {info.duration:.0f} 秒（{info.duration/60:.1f} 分钟）")
    print()

    # ===== 第5步：收集文字段 =====
    results = []
    total_text = ""

    for segment in segments:
        seg_dict = {
            "start": round(segment.start, 2),
            "end": round(segment.end, 2),
            "text": segment.text.strip(),
        }
        results.append(seg_dict)
        total_text += segment.text.strip()

    # ===== 第6步：显示转录样本（前 15 段）=====
    print("=" * 50)
    print("  转录样本（前 15 段）")
    print("=" * 50)
    print()

    for i, seg in enumerate(results[:15]):
        m = int(seg["start"] // 60)
        s = int(seg["start"] % 60)
        print(f"  [{m:02d}:{s:02d}] {seg['text']}")

    if len(results) > 15:
        print(f"  ...")
        print(f"  （共 {len(results)} 段，此处只显示前 15 段）")

    print()
    print(f"  语音识别完成！")
    print(f"  总段数: {len(results)}")
    print(f"  总字数: {len(total_text)} 字")
    print()

    return results


def save_transcript(segments, audio_name):
    """
    保存转录结果（3 种格式）
    """
    # 去掉 .wav 后缀，作为基础文件名
    base_name = audio_name.replace(".wav", "")
    output_dir = Path("output/transcripts")
    output_dir.mkdir(parents=True, exist_ok=True)

    # ----- 版本1：带时间戳的文本（给人看）-----
    txt_path = output_dir / f"{base_name}.txt"
    with open(txt_path, "w", encoding="utf-8", newline="\n") as f:
        for seg in segments:
            m = int(seg["start"] // 60)
            s = int(seg["start"] % 60)
            f.write(f"[{m:02d}:{s:02d}] {seg['text']}\n")

    # ----- 版本2：JSON 格式（给程序用）-----
    json_path = output_dir / f"{base_name}.json"
    with open(json_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)

    # ----- 版本3：纯文本无时间戳（给第三阶段 LLM 用）-----
    plain_path = output_dir / f"{base_name}_plain.txt"
    with open(plain_path, "w", encoding="utf-8", newline="\n") as f:
        for seg in segments:
            f.write(seg["text"] + "\n")

    print("=" * 50)
    print("  转录已保存为 3 个文件")
    print("=" * 50)
    print()
    print(f"  带时间戳版: {txt_path}")
    print(f"  JSON 格式:  {json_path}")
    print(f"  纯文本版:   {plain_path}  ← 下一阶段用这个")
    print()

    return str(plain_path)


# ============================================================
# 主程序入口
# ============================================================
if __name__ == "__main__":
    print()
    print("=" * 50)
    print("   第二阶段：音频 → 语音转文字")
    print("   配置: medium / GPU / int8 量化")
    print("=" * 50)
    print()

    # ====================================================
    # 把这里改成你的音频文件名
    # ====================================================
    audio_name = "test.wav"
    # ====================================================

    audio_path = f"output/audio/{audio_name}"

    # 执行识别
    segments = transcribe_audio(audio_path)

    if segments:
        # 保存结果
        save_transcript(segments, audio_name)
        print("=" * 50)
        print("  第二阶段完成！")
        print("  下一步：进入第三阶段（笔记生成）")
        print("=" * 50)
    else:
        print("第二阶段失败，请检查上面的错误信息。")
