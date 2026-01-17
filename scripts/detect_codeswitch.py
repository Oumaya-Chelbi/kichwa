import pandas as pd
import re

CONLLU_FILE = "kc_killkan-ud-test.conllu"
OUTPUT_TSV = "codeswitch.tsv"

lang_re = re.compile(r"Lang=([a-zA-Z]+)")

sent_langs = {}
current_sent = None
langs_in_sent = set()

with open(CONLLU_FILE, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()

        # nouveau segment
        if line.startswith("# sent_id"):
            if current_sent is not None:
                sent_langs[current_sent] = set(langs_in_sent)

            current_sent = line.split("=")[1].strip()
            langs_in_sent = set()

        # ligne token
        elif line and not line.startswith("#"):
            cols = line.split("\t")
            if len(cols) >= 10:
                misc = cols[9]   #  ICI est Lang
                m = lang_re.search(misc)
                if m:
                    langs_in_sent.add(m.group(1))

# dernier segment
if current_sent is not None:
    sent_langs[current_sent] = set(langs_in_sent)

#  sent_id → audio
rows = []

for sent_id, langs in sent_langs.items():
    chap, idx = sent_id.split("_")
    chap = int(chap)
    idx = int(idx) + 1   # ton mapping confirmé

    audio_file = f"Chapter{chap}_{idx}_{idx}.wav"

    has_cs = 1 if len(langs) > 1 else 0

    rows.append({
        "audio_file": audio_file,
        "has_codeswitch": has_cs
    })

df = pd.DataFrame(rows)
df.to_csv(OUTPUT_TSV, sep="\t", index=False)

print("codeswitch.tsv généré correctement.")
