import os
import glob
import pandas as pd
import soundfile as sf
import librosa
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor


CODESWITCH_PATH = "./CORPUS_qug_Cswitch"
MODEL_DIR = "jonatasgrosman/wav2vec2-large-xlsr-53-spanish"  # baseline
# ou MODEL_DIR = "./results_qug_wav2vec_spanish/best_model" pour le modèle fine-tuné
PROCESSOR_DIR = MODEL_DIR  # si tu as un processor custom, mets son chemin

OUT_TSV = "./predictions_codeswitch_baseline.tsv"

device = "cuda" if torch.cuda.is_available() else "cpu"

processor = Wav2Vec2Processor.from_pretrained(PROCESSOR_DIR)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_DIR).to(device)
model.eval()

rows = []
wav_files = glob.glob(os.path.join(CODESWITCH_PATH, "*.wav"))

for wav in wav_files:
    txt = wav[:-4] + ".txt"
    ref = ""
    if os.path.exists(txt):
        with open(txt, "r", encoding="utf-8") as f:
            ref = f.read().strip()

    # Lecture audio avec soundfile
    waveform, sr = sf.read(wav)  # np.array [num_samples] ou [num_samples, channels]
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)  # mixage mono si stéréo

    if sr != 16000:
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=16000)
        sr = 16000

    with torch.no_grad():
        inputs = processor(waveform, sampling_rate=sr, return_tensors="pt")
        input_values = inputs.input_values.to(device)
        logits = model(input_values).logits
        pred_ids = torch.argmax(logits, dim=-1)
        hyp = processor.batch_decode(pred_ids)[0]
        hyp = hyp.replace("|", " ").strip()

    utt_id = os.path.basename(wav)[:-4]
    rows.append({"utt_id": utt_id, "ref": ref, "hyp": hyp})

df = pd.DataFrame(rows)
df.to_csv(OUT_TSV, sep="\t", index=False)
print("Saved predictions to", OUT_TSV)
