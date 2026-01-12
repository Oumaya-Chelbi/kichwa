import os
from pathlib import Path
import pandas as pd

TXT_DIR = Path("data_txt")
WAV_DIR = Path("data_wavs")
OUT_DIR = Path("CORPUS_qug")
OUT_DIR.mkdir(exist_ok=True)

entries = []  # Pour construire le futur TSV

# Parcours de tous les .txt
for txt_file in TXT_DIR.rglob("*.txt"):
    # Exemple : data_txt/chapter1/1/1.txt → ["data_txt","chapter1","1","1.txt"]
    parts = txt_file.parts

    # On extrait chapter / sousdossier / nom
    # Exemple: chapter1, 1, 1.txt
    chapter = parts[-3]
    subfolder = parts[-2]
    base = txt_file.stem  # "1"

    # Construction du nom unique : chapter1_1_1
    unique_id = f"{chapter}_{subfolder}_{base}"

    # Chemin WAV correspondant (même structure)
    wav_file = WAV_DIR / chapter / subfolder / (base + ".wav")

    if not wav_file.exists():
        print(f"[WARN] Pas de WAV pour {txt_file}")
        continue

    # Nouveau chemins dans CORPUS_qug
    new_txt = OUT_DIR / f"{unique_id}.txt"
    new_wav = OUT_DIR / f"{unique_id}.wav"

    # Copie
    new_txt.write_text(txt_file.read_text(), encoding="utf-8")
    new_wav.write_bytes(wav_file.read_bytes())

    # Pour le TSV
    entries.append({
        "audio_file": new_wav.name,
        "transcription": txt_file.read_text().strip()
    })

# Construction du TSV final
df = pd.DataFrame(entries)
df.to_csv(OUT_DIR / "corpus_metadata.tsv", sep="\t", index=False)

print("Corpus construit dans CORPUS_qug/")
print("TSV écrit : CORPUS_qug/corpus_metadata.tsv")
