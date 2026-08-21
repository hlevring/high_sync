"""Run all same-named audio/video pairs and collect performance metrics.

Example (PowerShell):
    python benchmark_inference.py --input_dir .\test_cases

The input directory should contain pairs such as ``001.mp4`` and ``001.wav``.
Outputs are written to ``output`` by default.
"""

import argparse
import csv
import gc
import sys
import time
from pathlib import Path

import soundfile as sf
import torch

from inference import inference_process


VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg"}


def find_pairs(input_dir: Path):
    videos = {p.stem.lower(): p for p in input_dir.rglob("*")
              if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS}
    audios = {p.stem.lower(): p for p in input_dir.rglob("*")
              if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS}
    names = sorted(videos.keys() & audios.keys())
    missing_audio = sorted(videos.keys() - audios.keys())
    missing_video = sorted(audios.keys() - videos.keys())
    return [(name, videos[name], audios[name]) for name in names], missing_audio, missing_video


def audio_duration(path: Path) -> float:
    return float(sf.info(str(path)).duration)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", type=Path, required=True,
                        help="Folder containing same-named audio/video pairs")
    parser.add_argument("--output_dir", type=Path, default=Path("output"))
    parser.add_argument("--config", type=Path,
                        default=Path("configs/inference/default.yaml"))
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    pairs, missing_audio, missing_video = find_pairs(input_dir)
    if not pairs:
        raise SystemExit(f"No matching audio/video pairs found in {input_dir}")

    rows = []
    for index, (name, video, audio) in enumerate(pairs, 1):
        output_video = output_dir / f"{name}.mp4"
        print(f"[{index}/{len(pairs)}] {name}: {video.name} + {audio.name}", flush=True)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        gc.collect()
        started = time.perf_counter()
        status = "ok"
        error = ""
        try:
            inference_process(argparse.Namespace(
                config=str(args.config.resolve()),
                source_video=str(video),
                driving_audio=str(audio),
                output=str(output_video),
                inference_frames_num=1500,
            ))
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception as exc:
            status = "failed"
            error = f"{type(exc).__name__}: {exc}"
            print(f"  FAILED: {error}", file=sys.stderr, flush=True)
        elapsed = time.perf_counter() - started
        duration = audio_duration(audio)
        peak_vram = (torch.cuda.max_memory_reserved() / (1024 ** 3)
                     if torch.cuda.is_available() else 0.0)
        rows.append({
            "case": name,
            "video": str(video),
            "audio": str(audio),
            "output": str(output_video),
            "status": status,
            "runtime_seconds": round(elapsed, 3),
            "audio_duration_seconds": round(duration, 3),
            "runtime_audio_ratio": round(elapsed / duration, 4) if duration else "",
            "peak_vram_gib": round(peak_vram, 3),
            "error": error,
        })

    results_path = output_dir / "benchmark_results.csv"
    with results_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nFinished. Results: {results_path}")
    if missing_audio:
        print("Videos without matching audio:", ", ".join(missing_audio))
    if missing_video:
        print("Audio without matching video:", ", ".join(missing_video))


if __name__ == "__main__":
    main()
