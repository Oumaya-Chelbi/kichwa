#!/usr/bin/env python3
"""
mp4_to_wav.py
Convertit tous les .mp4 en .wav 16kHz mono (ne réécrit pas si le .wav existe).

Usage:
    python mp4_to_wav.py --input_dir ./data --output_dir ./data_wavs --target_sr 16000

Pré-requis:
    ffmpeg doit être installé et disponible dans le PATH.
"""

import os
import argparse
import subprocess
from pathlib import Path
import shlex

def ffmpeg_installed():
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except FileNotFoundError:
        return False

def convert_mp4_to_wav(input_path, output_path, target_sr=16000):
    """
    Appelle ffmpeg pour extraire l'audio et convertir en WAV 16k mono PCM S16.
    Commande utilisée:
      ffmpeg -y -i "input.mp4" -ar 16000 -ac 1 -vn -acodec pcm_s16le "output.wav"
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        print(f"[SKIP] {output_path} exists")
        return
    cmd = [
        "ffmpeg",
        "-y",               # overwrite output if exists (we already check)
        "-i", str(input_path),
        "-ar", str(target_sr),  # set sampling rate
        "-ac", "1",         # mono
        "-vn",              # no video
        "-acodec", "pcm_s16le",
        str(output_path)
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"[OK] {input_path} -> {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] ffmpeg failed for {input_path}: {e}")

def process_all_mp4(input_dir, output_dir, target_sr=16000):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    for mp4_path in input_dir.rglob("*.[mM][pP]4"):
        rel = mp4_path.relative_to(input_dir)
        out_wav = output_dir.joinpath(rel).with_suffix(".wav")
        convert_mp4_to_wav(mp4_path, out_wav, target_sr=target_sr)

def main():
    parser = argparse.ArgumentParser(description="Convert mp4 -> wav (16kHz mono) using ffmpeg")
    parser.add_argument("--input_dir", type=str, default="./data/killkan/data", help="Dossier contenant les .mp4")
    parser.add_argument("--output_dir", type=str, default="./data_wavs", help="Dossier où écrire les .wav")
    parser.add_argument("--target_sr", type=int, default=16000, help="Sampling rate cible (default 16000)")
    args = parser.parse_args()

    if not ffmpeg_installed():
        print("Erreur: ffmpeg n'est pas installé ou non trouvé dans le PATH. Installe ffmpeg avant d'exécuter ce script.")
        return

    process_all_mp4(args.input_dir, args.output_dir, args.target_sr)

if __name__ == "__main__":
    main()
