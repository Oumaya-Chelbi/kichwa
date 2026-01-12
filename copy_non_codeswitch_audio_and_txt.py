import os
import shutil
import pandas as pd

# ===== CONFIG =====
TSV_FILE = "codeswitch.tsv"
SRC_DIR = "CORPUS_qug"
DST_DIR = "CORPUS_qug_NO_Cswitch"

os.makedirs(DST_DIR, exist_ok=True)

# ===== LOAD TSV =====
df = pd.read_csv(TSV_FILE, sep="\t")

# audios avec code-switching
cs_files = set(
    df[df["has_codeswitch"] == 1]["audio_file"].tolist()
)

copied_wav = 0
copied_txt = 0
skipped = 0

# ===== PARCOURT DU CORPUS =====
for fname in os.listdir(SRC_DIR):

    # on s'intéresse seulement aux wav et txt
    if not (fname.endswith(".wav") or fname.endswith(".txt")):
        continue

    # retrouver le wav associé
    base = fname.replace(".wav", "").replace(".txt", "")
    wav_name = base + ".wav"

    # si ce wav est CS → on ignore
    if wav_name in cs_files:
        skipped += 1
        continue

    src_path = os.path.join(SRC_DIR, fname)
    dst_path = os.path.join(DST_DIR, fname)

    if os.path.exists(src_path):
        shutil.copy2(src_path, dst_path)
        if fname.endswith(".wav"):
            copied_wav += 1
        else:
            copied_txt += 1

print("===== Résumé =====")
print(f"WAV copiés : {copied_wav}")
print(f"TXT copiés : {copied_txt}")
print(f"Fichiers ignorés (code-switching) : {skipped}")
print("Corpus sans code-switching prêt dans :", DST_DIR)
