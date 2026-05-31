"""
第四阶段前置：从视频 URL 下载音频
功能：输入 B站/YouTube URL → 自动下载最佳音频流 → 转 16kHz 单声道 WAV
依赖：yt-dlp + FFmpeg
用法：python download.py
     python download.py "https://www.bilibili.com/video/BV1xx411c7mD"
"""

import subprocess
import json
import sys
from pathlib import Path


def download_from_url(url, output_dir="output/audio"):
    """
    从视频 URL 下载最佳音频流，转成 16kHz 单声道 WAV，同时提取元数据

    参数:
        url: 视频 URL（B站 / YouTube）
        output_dir: 输出目录，默认 "output/audio"

    返回:
        dict: {"audio_path", "title", "uploader", "source_url", "duration"}
        None: 下载或转换失败
    """
    # ===== 第1步：检查 yt-dlp 是否可用 =====
    try:
        subprocess.run(
            ["yt-dlp", "--version"],
            capture_output=True, check=True,
            encoding="utf-8", errors="replace",
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("[错误] 找不到 yt-dlp，请先安装：pip install yt-dlp")
        return None

    # ===== 第2步：提取视频元数据（--dump-json，不下载视频）=====
    print("[进行中] 正在获取视频信息...")
    print(f"   URL: {url}")
    print()

    try:
        result = subprocess.run(
            [
                "yt-dlp",
                "--dump-json",
                "--no-playlist",
                "--socket-timeout", "30",
                url,
            ],
            capture_output=True,
            text=True, encoding="utf-8", errors="replace",
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        print("[错误] 获取视频信息超时（120 秒），请检查网络或 URL 是否有效")
        return None

    if result.returncode != 0:
        print("[错误] 无法访问该 URL")
        stderr = result.stderr.strip()
        if stderr:
            for line in stderr.split("\n")[-3:]:
                if line.strip():
                    print(f"   {line.strip()}")
        return None

    try:
        info = json.loads(result.stdout)
    except json.JSONDecodeError:
        print("[错误] 无法解析视频信息 JSON")
        return None

    title = info.get("title", "未知标题")
    uploader = info.get("uploader", info.get("channel", "未知上传者"))
    duration = info.get("duration", 0) or 0
    source_url = info.get("webpage_url", url)

    print(f"   标题:   {title}")
    print(f"   上传者: {uploader}")
    if duration:
        print(f"   时长:   {duration} 秒（约 {duration / 60:.1f} 分钟）")
    print()

    # ===== 第3步：清理文件名中的特殊字符并截断 =====
    safe_title = title
    # B站标题常见格式: "主标题|副标题 p04 课时名" → 取主标题+"p04"部分
    if "|" in safe_title or "｜" in safe_title:
        parts = safe_title.replace("｜", "|").split("|")
        safe_title = parts[0].strip() + " " + " ".join(p for p in parts[1:] if "p" in p.lower()[:3])
    for ch in r'<>:"/\|?*':
        safe_title = safe_title.replace(ch, "_")
    while "__" in safe_title:
        safe_title = safe_title.replace("__", "_")
    safe_title = safe_title.strip(". ")
    # 截断过长的文件名
    if len(safe_title) > 50:
        safe_title = safe_title[:50]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    wav_audio = output_path / f"{safe_title}.wav"

    # 如果 WAV 已存在，跳过下载和转换
    if wav_audio.exists():
        size_mb = wav_audio.stat().st_size / (1024 * 1024)
        print(f"[跳过] 音频文件已存在，无需重复下载")
        print(f"   文件: {wav_audio.name}")
        print(f"   大小: {size_mb:.1f} MB")
        print()
        return {
            "audio_path": str(wav_audio),
            "title": title,
            "uploader": uploader,
            "source_url": source_url,
            "duration": duration,
        }

    # ===== 第4步：下载最佳音频流 =====
    print("[进行中] 正在下载最佳音频流...")
    print(f"   格式: bestaudio [m4a] / bestaudio")
    print(f"   超时限制: 5 分钟")
    print()

    try:
        subprocess.run(
            [
                "yt-dlp",
                "-f", "bestaudio[ext=m4a]/bestaudio",
                "-o", str(output_path / f"{safe_title}.%(ext)s"),
                "--no-playlist",
                "--socket-timeout", "30",
                url,
            ],
            capture_output=True,
            text=True, encoding="utf-8", errors="replace",
            timeout=300,
            check=True,
        )
    except subprocess.TimeoutExpired:
        print("[错误] 下载超时（5 分钟），请检查网络后重试")
        return None
    except subprocess.CalledProcessError as e:
        print("[错误] 下载失败")
        stderr = e.stderr.strip()
        if stderr:
            for line in stderr.split("\n")[-5:]:
                if line.strip():
                    print(f"   {line.strip()}")
        return None

    # 找到下载的音频文件（扩展名取决于实际下载的格式）
    downloaded = None
    for f in output_path.iterdir():
        if f.stem == safe_title and f.suffix != ".wav":
            downloaded = f
            break

    if downloaded is None:
        print("[错误] 下载完成但未找到音频文件")
        return None

    size_mb = downloaded.stat().st_size / (1024 * 1024)
    print(f"   下载完成: {downloaded.name}（{size_mb:.1f} MB）")
    print()

    # ===== 第5步：FFmpeg 转 16kHz 单声道 WAV =====
    print("[进行中] 正在转换为 16kHz 单声道 WAV...")
    print(f"   输入: {downloaded.name}")
    print(f"   输出: {wav_audio.name}")
    print()

    cmd = [
        "ffmpeg",
        "-i", str(downloaded),
        "-vn",                     # 丢弃视频轨道
        "-acodec", "pcm_s16le",    # 音频编码：PCM 16-bit
        "-ar", "16000",            # 采样率：16kHz（Whisper 要求）
        "-ac", "1",                # 单声道
        "-y",                      # 覆盖已有文件
        str(wav_audio),
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print("[错误] 音频转换失败")
        print(f"   {e.stderr.strip()[-500:]}")
        return None
    except FileNotFoundError:
        print("[错误] 找不到 FFmpeg，请确认已安装并在 PATH 中")
        return None

    # ===== 第6步：删除原始下载的音频（保留 WAV）=====
    if downloaded != wav_audio:
        downloaded.unlink(missing_ok=True)

    # ===== 第7步：确认输出 =====
    if not wav_audio.exists():
        print("[错误] WAV 文件未生成，未知错误")
        return None

    wav_size = wav_audio.stat().st_size / (1024 * 1024)
    print(f"[完成] 音频下载并转换成功！")
    print(f"   文件: {wav_audio}")
    print(f"   大小: {wav_size:.1f} MB")
    print()

    return {
        "audio_path": str(wav_audio),
        "title": title,
        "uploader": uploader,
        "source_url": source_url,
        "duration": duration,
    }


# ============================================================
# 主程序入口
# ============================================================
if __name__ == "__main__":
    print("=" * 50)
    print("  视频 URL → 音频下载 + 格式转换")
    print("  支持: B站 / YouTube")
    print("=" * 50)
    print()

    if len(sys.argv) > 1:
        url = sys.argv[1]
    else:
        url = input("请输入视频 URL（B站/YouTube）: ").strip()

    if not url:
        print("[错误] 未输入 URL")
        exit(1)

    result = download_from_url(url)

    if result:
        print("=" * 50)
        print("  下载成功！")
        print("=" * 50)
        print()
        print(f"  标题:    {result['title']}")
        print(f"  上传者:  {result['uploader']}")
        print(f"  时长:    {result['duration']} 秒（约 {result['duration'] / 60:.1f} 分钟）")
        print(f"  来源:    {result['source_url']}")
        print(f"  音频:    {result['audio_path']}")
        print()
        print("  下一步：修改 transcribe_audio.py 中的音频文件名，运行语音识别")
        print("=" * 50)
    else:
        print()
        print("[失败] 下载未成功，请检查上面的错误信息。")
        exit(1)
