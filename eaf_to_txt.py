#!/usr/bin/env python3
"""
eaf_to_txt.py
Extrait les transcriptions depuis les fichiers .eaf (ELAN) et crée .txt correspondants.

Usage:
    python eaf_to_txt.py --input_dir ./data --output_dir ./data_txt

Le script parcourt récursivement input_dir, cherche les .eaf et écrit un .txt
du même nom dans output_dir (sous-arborescence conservée).
"""

import os
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

def extract_text_from_eaf(eaf_path):
    """
    Lit un fichier .eaf (XML ELAN) et retourne une chaîne de transcription brute.
    - Récupère toutes les valeurs d'annotation (ANNOTATION_VALUE) présentes dans
      les tiers (TIER).
    - Trie les annotations par time order si possible (begin time), sinon par l'ordre d'apparition.
    - Concatène les valeurs en tenant compte des espaces et en nettoyant les retours à la ligne.
    """
    try:
        tree = ET.parse(eaf_path)
        root = tree.getroot()
    except Exception as e:
        print(f"[ERROR] Impossible de parser {eaf_path}: {e}")
        return ""

    # namespace handling (ELAN documents often have a default namespace)
    ns = {}
    # build namespace map if present
    for k, v in root.attrib.items():
        if k.startswith("xmlns"):
            # either "xmlns" or "xmlns:prefix"
            parts = k.split(":")
            if len(parts) == 1:
                ns["elan"] = v
            else:
                ns[parts[-1]] = v

    # helper to find tags with/without namespace
    def findall(element, tag):
        # try with no namespace
        nodes = element.findall(tag)
        if nodes:
            return nodes
        # try with common namespace prefixes
        for prefix_uri in ns.values():
            nodes = element.findall(f"{{{prefix_uri}}}{tag}")
            if nodes:
                return nodes
        return []

    # Build a mapping from TIME_SLOT ID -> time value (for alignable annotations)
    time_order = {}
    time_slots = findall(root, "TIME_ORDER")
    if time_slots:
        # TIME_SLOT children
        for to in time_slots:
            for ts in list(to):
                # ts tag could be TIME_SLOT or namespaced
                if ts.tag.endswith("TIME_SLOT"):
                    tid = ts.attrib.get("TIME_SLOT_ID")
                    tvalue = ts.attrib.get("TIME_VALUE")
                    if tid:
                        time_order[tid] = int(tvalue) if tvalue is not None else None

    # collect annotations as (start_time, value) or (None, value)
    annots = []

    # iterate tiers
    for tier in findall(root, "TIER"):
        # find all ANNOTATIONs in the tier
        for annotation in findall(tier, "ANNOTATION"):
            # Try ALIGNABLE_ANNOTATION first
            a = None
            for child in list(annotation):
                tagname = child.tag
                if tagname.endswith("ALIGNABLE_ANNOTATION"):
                    a = child
                    # begin/end time
                    ts1 = child.attrib.get("TIME_SLOT_REF1")
                    ts2 = child.attrib.get("TIME_SLOT_REF2")
                    start = time_order.get(ts1) if ts1 else None
                    # value:
                    val_el = None
                    for c2 in list(child):
                        if c2.tag.endswith("ANNOTATION_VALUE"):
                            val_el = c2
                            break
                    val = val_el.text if (val_el is not None and val_el.text) else ""
                    annots.append((start, val.strip()))
                elif tagname.endswith("REF_ANNOTATION"):
                    # REF_ANNOTATION may contain a child with annotation value
                    val_el = None
                    for c2 in list(child):
                        if c2.tag.endswith("ANNOTATION_VALUE"):
                            val_el = c2
                            break
                    val = val_el.text if (val_el is not None and val_el.text) else ""
                    # no start time for ref annotations
                    annots.append((None, val.strip()))
                else:
                    # ignore other tags
                    pass

    # If we have start times, sort by them (None go to the end)
    annots_sorted = sorted(annots, key=lambda x: (x[0] is None, x[0] if x[0] is not None else 10**12))

    # join non-empty values with a space, and normalize whitespace
    pieces = [a for _, a in annots_sorted if a and a.strip()]
    text = " ".join(pieces)
    # simple cleanup: collapse multiple spaces and strip
    text = " ".join(text.split())
    return text

def process_all_eaf(input_dir, output_dir):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    for eaf_path in input_dir.rglob("*.eaf"):
        rel = eaf_path.relative_to(input_dir)
        out_path = output_dir.joinpath(rel).with_suffix(".txt")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.exists():
            print(f"[SKIP] {out_path} already exists")
            continue
        txt = extract_text_from_eaf(str(eaf_path))
        if txt:
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(txt)
            print(f"[OK] Wrote {out_path} (chars: {len(txt)})")
        else:
            # write empty file or skip? we'll write a small notice
            with open(out_path, "w", encoding="utf-8") as f:
                f.write("")
            print(f"[WARN] Empty transcription for {eaf_path}, created empty {out_path}")

def main():
    parser = argparse.ArgumentParser(description="Extract transcriptions from .eaf into .txt files")
    parser.add_argument("--input_dir", type=str, default="./data/killkan/dataea", help="Dossier racine contenant les .eaf")
    parser.add_argument("--output_dir", type=str, default="./data_txt", help="Dossier où écrire les .txt")
    args = parser.parse_args()
    process_all_eaf(args.input_dir, args.output_dir)

if __name__ == "__main__":
    main()
