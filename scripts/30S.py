import os
import glob
import shutil
import soundfile as sf

SRC_DIR = "CORPUS_qug_NO_Cswitch"
DST_DIR = "CORPUS_qug_NO_Cswitch_30"
MAX_DURATION = 30.0  # secondes

os.makedirs(DST_DIR, exist_ok=True)

wav_files = glob.glob(os.path.join(SRC_DIR, "*.wav"))

kept = 0
skipped = 0

for wav_path in wav_files:
    # lire audio
    data, sr = sf.read(wav_path)
    duration = len(data) / sr

    if duration <= MAX_DURATION:
        base = os.path.basename(wav_path)
        txt_path = os.path.splitext(wav_path)[0] + ".txt"

        # copier le wav
        shutil.copy2(wav_path, os.path.join(DST_DIR, base))

        # copier le txt associé s'il existe
        if os.path.exists(txt_path):
            shutil.copy2(
                txt_path,
                os.path.join(DST_DIR, os.path.basename(txt_path))
            )

        kept += 1
    else:
        skipped += 1

print(f"Gardés (<= {MAX_DURATION} s) : {kept} fichiers wav")
print(f"Ignorés (> {MAX_DURATION} s) : {skipped} fichiers wav")
print(f"Résultats dans : {DST_DIR}")
