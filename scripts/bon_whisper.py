import os
import argparse
import pandas as pd
from glob import glob
import torch
import librosa
from ft_whisper import (
    initialize_model_and_processors,
    WhisperFineTuner,
)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

parser = argparse.ArgumentParser()
parser.add_argument(
    "-o",
    "--output",
    help="output TSV file",
    default="transcriptions_ft_whisper_qug_codeswitch.tsv",
)
args = parser.parse_args()

# 1. Charger modèle fine-tuné
MODEL_NAME = "openai/whisper-base"       
LANGUAGE = "Spanish"                   
CHECKPOINT_PATH = "./SAVE_DIR/model.pt" 

# Recharger les composants Whisper (même init que dans ft_whisper.py)
whisper_model, feature_extractor, tokenizer, processor = initialize_model_and_processors(
    MODEL_NAME, LANGUAGE
)

print("[INFO] Loading fine-tuned model...")
model = WhisperFineTuner.load_from_checkpoint(
    CHECKPOINT_PATH,
    model=whisper_model,
    processor=processor,
)
model = model.to("cuda" if torch.cuda.is_available() else "cpu")
model.eval()

# 2. Charger les fichiers audio de CORPUS_qug_Cswitch
AUDIO_DIR = "CORPUS_qug_Cswitch"
audio_files = sorted(glob(os.path.join(AUDIO_DIR, "*.wav")))
print(f"[INFO] Found {len(audio_files)} audio files in {AUDIO_DIR}.")

results = []

# 3. Transcription
for audio_path in audio_files:
    print(f"[INFO] Processing {audio_path}")

    # Load audio en 16 kHz
    audio, sr = librosa.load(audio_path, sr=16000)

    # Extract features
    inputs = processor(
        audio,
        sampling_rate=16000,
        return_tensors="pt",
    ).to(model.device)

    # Forward (on utilise model.model comme dans ton code)
    with torch.no_grad():
        generated_ids = model.model.generate(inputs["input_features"])

    # Decode
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # Durée
    duration = librosa.get_duration(path=audio_path)

    results.append({
        "audio_file": os.path.basename(audio_path),
        "transcription": text.strip(),
        "duration": duration,
    })

# 4. Save TSV
df = pd.DataFrame(results)
df.to_csv(args.output, sep="\t", index=False)
print(f"[INFO] Saved transcriptions to: {args.output}")
