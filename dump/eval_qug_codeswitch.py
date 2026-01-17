import pandas as pd
from jiwer import wer, cer

TSV_PATH = "./predictions_codeswitch.tsv"

df = pd.read_csv(TSV_PATH, sep="\t")

refs = df["ref"].tolist()
hyps = df["hyp"].tolist()

w = wer(refs, hyps)
c = cer(refs, hyps)

print("WER:", w)
print("CER:", c)
