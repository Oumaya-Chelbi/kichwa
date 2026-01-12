import os
import shutil
import pandas as pd

#  CONFIG 
TSV_FILE = "codeswitch.tsv"
SRC_DIR = "CORPUS_qug"
DST_DIR = "CORPUS_qug_Cswitch"

os.makedirs(DST_DIR, exist_ok=True)

#  LOAD TSV 
df = pd.read_csv(TSV_FILE, sep="\t")

copied_wav = 0
copied_txt = 0
missing_txt = 0
missing_wav = 0

for _, row in df.iterrows():
    if row["has_codeswitch"] == 1:
        wav_name = row["audio_file"]
        txt_name = wav_name.replace(".wav", ".txt")

        wav_src = os.path.join(SRC_DIR, wav_name)
        txt_src = os.path.join(SRC_DIR, txt_name)

        wav_dst = os.path.join(DST_DIR, wav_name)
        txt_dst = os.path.join(DST_DIR, txt_name)

        # copy wav
        if os.path.exists(wav_src):
            shutil.copy2(wav_src, wav_dst)
            copied_wav += 1
        else:
            print(f"[WARNING] wav manquant : {wav_name}")
            missing_wav += 1

        # copy txt
        if os.path.exists(txt_src):
            shutil.copy2(txt_src, txt_dst)
            copied_txt += 1
        else:
            print(f"[WARNING] txt manquant : {txt_name}")
            missing_txt += 1

print(" Résumé ")
print(f"WAV copiés : {copied_wav}")
print(f"TXT copiés : {copied_txt}")
print(f"WAV manquants : {missing_wav}")
print(f"TXT manquants : {missing_txt}")
print("Corpus code-switching prêt dans :", DST_DIR)
