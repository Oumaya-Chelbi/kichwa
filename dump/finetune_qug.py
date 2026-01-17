import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"

import sys
sys.modules["tensorflow"] = None
sys.modules["keras"] = None
sys.modules["tf_keras"] = None

import glob
import re
import json
import pandas as pd
import numpy as np
from datasets import Dataset

import torch
import soundfile as sf
from torch import nn
from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    TrainingArguments,
    Trainer
)
from jiwer import wer, cer
from sklearn.model_selection import train_test_split

# ------------------------------------------------------
# Config
# ------------------------------------------------------
BASE_PATH = "./CORPUS_qug_NO_Cswitch"
RESAMPLED_DIR = "./resampled_qug_no_cswitch"
RESULTS_DIR = "./results_qug_wav2vec_spanish"
MODEL_NAME = "jonatasgrosman/wav2vec2-large-xlsr-53-spanish"

os.makedirs(RESAMPLED_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ------------------------------------------------------
# 1) Charger wav + txt
# ------------------------------------------------------
def load_qug_no_cswitch(base_path):
    rows = []
    for wav in glob.glob(os.path.join(base_path, "*.wav")):
        txt = wav[:-4] + ".txt"
        if os.path.exists(txt):
            with open(txt, "r", encoding="utf-8") as f:
                transcription = f.read().strip()
            rows.append({"path": wav, "sentence": transcription})
    return pd.DataFrame(rows)

print("Loading QUG NO_Cswitch data...")
df = load_qug_no_cswitch(BASE_PATH)
print(df.head())
print(df.columns)
print("Nb utt:", len(df))

# ------------------------------------------------------
# 2) Copie en 16k (sans resampling Python)
#    -> suppose que les wav sont déjà en 16 kHz
# ------------------------------------------------------
def resample_and_save(df, target_sr=16000):
    audio_paths = []
    for _, r in df.iterrows():
        orig = r["path"]
        name = os.path.basename(orig)
        new_path = os.path.join(RESAMPLED_DIR, name)
        if not os.path.exists(new_path):
            waveform, sr = sf.read(orig)
            if waveform.ndim > 1:
                waveform = waveform.mean(axis=1)  # mono
            if sr != target_sr:
                raise ValueError(
                    f"Sample rate {sr} != {target_sr}. "
                    "Merci de convertir les audios en 16 kHz en amont."
                )
            sf.write(new_path, waveform, sr)
        audio_paths.append(new_path)
    df["audio"] = audio_paths
    return df

# ------------------------------------------------------
# 3) Split train / dev / test
# ------------------------------------------------------
def train_dev_test_split(df, dev_size=0.1, test_size=0.1, seed=42):
    train_df, temp_df = train_test_split(
        df, test_size=dev_size + test_size,
        random_state=seed, shuffle=True
    )
    relative_test_size = test_size / (dev_size + test_size)
    dev_df, test_df = train_test_split(
        temp_df, test_size=relative_test_size,
        random_state=seed, shuffle=True
    )
    return train_df, dev_df, test_df

# ------------------------------------------------------
# 4) Nettoyage + vocab + processor
# ------------------------------------------------------
def clean_text(s: str) -> str:
    s = s.lower()
    s = s.replace("'", "’")
    s = re.sub(r"[^a-zà-öø-ÿ0-9’ ]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def build_vocab_from_corpus(sentences, vocab_file):
    cleaned_sentences = [clean_text(s) for s in sentences]

    vocab = set()
    for s in cleaned_sentences:
        for c in s:
            vocab.add(c)
    vocab = sorted(list(vocab))
    vocab_dict = {c: i for i, c in enumerate(vocab)}

    if " " in vocab_dict:
        vocab_dict["|"] = vocab_dict[" "]
        del vocab_dict[" "]
    else:
        vocab_dict["|"] = len(vocab_dict)

    vocab_dict["[UNK]"] = len(vocab_dict)
    vocab_dict["[PAD]"] = len(vocab_dict)

    with open(vocab_file, "w", encoding="utf-8") as f:
        json.dump(vocab_dict, f, ensure_ascii=False)

    return vocab_file, cleaned_sentences

def prepare_hf_dataset(df, processor):
    # pas de datasets.Audio pour éviter librosa/numba
    ds = Dataset.from_pandas(df[["audio", "sentence"]].reset_index(drop=True))

    def preprocess(batch):
        audio_arrays = []
        for path in batch["audio"]:
            waveform, sr = sf.read(path)
            if waveform.ndim > 1:
                waveform = waveform.mean(axis=1)
            if sr != 16000:
                raise ValueError(
                    f"Sample rate {sr} != 16000. "
                    "Merci de convertir les audios en 16 kHz en amont."
                )
            audio_arrays.append(waveform)

        inputs = processor(
            audio_arrays, sampling_rate=16000,
            return_tensors=None, padding=False
        )
        with processor.as_target_processor():
            labels = processor(batch["sentence"]).input_ids
        return {"input_values": inputs["input_values"], "labels": labels}

    return ds.map(preprocess, batched=True, remove_columns=ds.column_names)

def compute_metrics(pred, processor):
    pred_ids = np.argmax(pred.predictions, axis=-1)
    pred_str = processor.batch_decode(pred_ids)

    label_ids = pred.label_ids
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(label_ids, group_tokens=False)

    def clean_for_metric(t):
        return t.replace("|", " ").strip()

    pred_str = [clean_for_metric(s) for s in pred_str]
    label_str = [clean_for_metric(s) for s in label_str]

    wers = [wer(l, p) for l, p in zip(label_str, pred_str)]
    cers = [cer(l, p) for l, p in zip(label_str, pred_str)]
    return {"wer": float(np.mean(wers)), "cer": float(np.mean(cers))}

# ------------------------------------------------------
# DataCollator CTC
# ------------------------------------------------------
from dataclasses import dataclass
from typing import Union

@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features):
        input_features = [{"input_values": f["input_values"]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                return_tensors="pt",
            )

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        batch["labels"] = labels
        return batch

# ------------------------------------------------------
# MAIN
# ------------------------------------------------------
print("Loading QUG NO_Cswitch data...")
df = load_qug_no_cswitch(BASE_PATH)

df["sentence"] = df["sentence"].apply(clean_text)

df = resample_and_save(df, target_sr=16000)

train_df, dev_df, test_df = train_dev_test_split(df)

vocab_path = os.path.join(RESULTS_DIR, "vocab.json")
vocab_file, _ = build_vocab_from_corpus(df["sentence"].tolist(), vocab_path)

tokenizer = Wav2Vec2CTCTokenizer(
    vocab_file=vocab_file,
    unk_token="[UNK]",
    pad_token="[PAD]",
    word_delimiter_token="|",
)
feature_extractor = Wav2Vec2FeatureExtractor(
    feature_size=1,
    sampling_rate=16000,
    padding_value=0.0,
    do_normalize=True,
    return_attention_mask=True,
)
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

train_ds = prepare_hf_dataset(train_df, processor)
dev_ds = prepare_hf_dataset(dev_df, processor)
test_ds = prepare_hf_dataset(test_df, processor)

data_collator = DataCollatorCTCWithPadding(processor)

model = Wav2Vec2ForCTC.from_pretrained(
    MODEL_NAME,
    pad_token_id=processor.tokenizer.pad_token_id,
    ctc_loss_reduction="mean",
    ignore_mismatched_sizes=True,
)

old_vocab_size = model.lm_head.out_features
hidden_size = model.lm_head.in_features
new_vocab_size = len(processor.tokenizer)

new_lm_head = nn.Linear(hidden_size, new_vocab_size)
with torch.no_grad():
    new_lm_head.weight[:old_vocab_size, :] = model.lm_head.weight
    new_lm_head.bias[:old_vocab_size] = model.lm_head.bias
model.lm_head = new_lm_head
model.config.vocab_size = new_vocab_size

training_args = TrainingArguments(
    output_dir=RESULTS_DIR,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=5,
    learning_rate=1e-5,
    logging_steps=50,
    fp16=torch.cuda.is_available(),
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
)

def compute_metrics_wrapper(pred):
    return compute_metrics(pred, processor)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=dev_ds,
    data_collator=data_collator,
    tokenizer=processor,
    compute_metrics=compute_metrics_wrapper,
)

trainer.train()
metrics_test = trainer.predict(test_ds)
print("Test metrics:", compute_metrics(metrics_test, processor))

trainer.save_model(os.path.join(RESULTS_DIR, "best_model"))
processor.save_pretrained(os.path.join(RESULTS_DIR, "best_processor"))
