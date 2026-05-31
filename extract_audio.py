"""
第一阶段：从视频中提取音频
功能：把视频文件里的音频轨道单独拆出来，输出 16kHz 单声道 WAV
依赖：系统需安装 FFmpeg
用法：python extract_audio.py
"""

import subprocess
from pathlib import Path


def extract_audio(video_path, output_dir="output/audio"):
    """
    从视频文件提取音频，输出 16kHz 单声道 WAV

    参数:
        video_path: 视频文件路径，例如 "uploads/test.mp4"
        output_dir: 输出目录，默认 "output/audio"

    返回:
        输出音频文件的路径（字符串），失败返回 None
    """
    video_file = Path(video_path)

    # === 第1步：检查视频文件是否存在 ===
    if not video_file.exists():
        print(f"[错误] 找不到文件 {video_path}")
        print("   请确认视频已放在 uploads/ 目录下")
        return None

    # === 第2步：创建输出目录 ===
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # === 第3步：确定输出文件名 ===
    # 例如 test.mp4 → test.wav
    audio_filename = video_file.stem + ".wav"
    audio_path = output_path / audio_filename

    # === 第4步：构建 FFmpeg 命令 ===
    cmd = [
        "ffmpeg",
        "-i", str(video_file),       # 输入视频文件
        "-vn",                        # 不要视频轨道
        "-acodec", "pcm_s16le",      # 音频编码：PCM 16-bit
        "-ar", "16000",              # 采样率：16kHz
        "-ac", "1",                  # 单声道
        "-y",                        # 覆盖已有文件
        str(audio_path),             # 输出音频文件
    ]

    # === 第5步：执行转换并处理错误 ===
    print(f"[进行中] 正在从视频提取音频...")
    print(f"   输入文件: {video_file.name}")
    print(f"   输出文件: {audio_filename}")
    print(f"   （音频转换耗时约 1-3 分钟，请耐心等待）")
    print()

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        # capture_output=True: 捕获 FFmpeg 的控制台输出
        # check=True: 如果 FFmpeg 返回非零退出码，Python 会抛出异常
        # text=True: 输出以字符串形式返回，而非字节
    except subprocess.CalledProcessError as e:
        print(f"[错误] FFmpeg 执行失败")
        print(f"   错误详情: {e.stderr}")
        return None
    except FileNotFoundError:
        print(f"[错误] 找不到 FFmpeg")
        print(f"   请确认 FFmpeg 已安装，在终端输入 ffmpeg -version 验证")
        return None

    # === 第6步：确认输出文件是否成功生成 ===
    if audio_path.exists():
        size_mb = audio_path.stat().st_size / (1024 * 1024)
        print(f"[完成] 音频提取成功！")
        print(f"   文件位置: {audio_path}")
        print(f"   文件大小: {size_mb:.1f} MB")
        print()
        return str(audio_path)
    else:
        print(f"[错误] 音频文件未生成，未知错误")
        return None


# ============================================================
# 主程序入口
# ============================================================
if __name__ == "__main__":
    print("=" * 50)
    print("  第一阶段：视频 → 音频提取")
    print("=" * 50)
    print()

    # 修改这里为你的视频文件名
    # 默认使用 "test.mp4"，可改为你自己的文件名
    video_name = "test.mp4"
    video_path = f"uploads/{video_name}"

    # 执行提取
    result = extract_audio(video_path)

    if result:
        print("[成功] 第一阶段完成！音频文件已生成，可以运行第二阶段。")
    else:
        print("[失败] 请检查上面的错误信息。")
