"""
Generate all four 2x2 ablation MFA pronunciation dictionaries from transcripts/.

Usage:
    python scripts/generate_dictionaries.py
    python scripts/generate_dictionaries.py --phase 1
    python scripts/generate_dictionaries.py --export-txt mfa_transcripts
"""

import argparse
import csv
import os
from g2p_rules import (
    ablation_conjunct_only_g2p,
    ablation_schwa_only_g2p,
    assamese_naive_g2p,
    brajavali_liturgical_g2p,
    tokenize_text,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ARMS = {
    "proposed": brajavali_liturgical_g2p,
    "schwa_only": ablation_schwa_only_g2p,
    "conjunct_only": ablation_conjunct_only_g2p,
    "naive_baseline": assamese_naive_g2p,
}


def load_tsv_rows(path: str) -> list:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def collect_vocabulary(phase: int = 2) -> list:
    p1_path = os.path.join(REPO_ROOT, "transcripts", "phase1_120_phrases.tsv")
    p2_path = os.path.join(REPO_ROOT, "transcripts", "phase2_train_403_utterances.tsv")

    words = set()
    for row in load_tsv_rows(p1_path):
        words.update(tokenize_text(row["text"]))
    if phase == 2:
        for row in load_tsv_rows(p2_path):
            words.update(tokenize_text(row["text"]))
    return sorted(words)


def export_mfa_txt(out_dir: str) -> None:
    """Exports individual UTF-8 .txt files structured for Montreal Forced Aligner."""
    p1_path = os.path.join(REPO_ROOT, "transcripts", "phase1_120_phrases.tsv")
    p2_path = os.path.join(REPO_ROOT, "transcripts", "phase2_train_403_utterances.tsv")

    p1_rows = load_tsv_rows(p1_path)
    p2_rows = load_tsv_rows(p2_path)

    for r in p1_rows:
        gdir = os.path.join(out_dir, "phase1", r["genre"])
        tdir = os.path.join(out_dir, "phase2", "test", "S1")
        os.makedirs(gdir, exist_ok=True)
        os.makedirs(tdir, exist_ok=True)
        for d in (gdir, tdir):
            with open(os.path.join(d, f"{r['utterance_id']}.txt"), "w", encoding="utf-8") as f:
                f.write(r["text"])

    for r in p2_rows:
        sdir = os.path.join(out_dir, "phase2", "train", r["speaker"])
        os.makedirs(sdir, exist_ok=True)
        with open(os.path.join(sdir, f"{r['utterance_id']}.txt"), "w", encoding="utf-8") as f:
            f.write(r["text"])

    print(f"Exported 120 Phase 1 / S1 and 403 Phase 2 train .txt files to: {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Brajavali 2x2 ablation MFA dictionaries.")
    parser.add_argument("--phase", type=int, choices=[1, 2], default=2, help="1 (408 words) or 2 (1,963 words)")
    parser.add_argument("--out-dir", default=os.path.join(REPO_ROOT, "dictionaries"), help="Output directory for .dict files")
    parser.add_argument("--export-txt", default=None, help="Optional directory to unpack .txt files for MFA")
    args = parser.parse_args()

    if args.export_txt:
        export_mfa_txt(args.export_txt)

    vocab = collect_vocabulary(phase=args.phase)
    os.makedirs(args.out_dir, exist_ok=True)

    dicts = {}
    for arm_name, fn in ARMS.items():
        mapping = {}
        out_path = os.path.join(args.out_dir, f"{arm_name}.dict")
        with open(out_path, "w", encoding="utf-8") as f:
            for word in vocab:
                ipa = fn(word)
                assert ipa, f"Empty pronunciation for {word} in {arm_name}"
                mapping[word] = ipa
                f.write(f"{word}\t{ipa}\n")
        dicts[arm_name] = mapping
        print(f"Wrote {len(vocab)} entries -> {out_path}")

    prop = dicts["proposed"]
    diff_naive = sum(1 for w in vocab if prop[w] != dicts["naive_baseline"][w])
    diff_schwa = sum(1 for w in vocab if prop[w] != dicts["schwa_only"][w])
    diff_conj = sum(1 for w in vocab if prop[w] != dicts["conjunct_only"][w])
    print(
        f"Phase {args.phase} ({len(vocab)} words): "
        f"Proposed!=Naive={diff_naive}, Proposed!=SchwaOnly(CD)={diff_schwa}, Proposed!=ConjOnly(SR)={diff_conj}"
    )


if __name__ == "__main__":
    main()
