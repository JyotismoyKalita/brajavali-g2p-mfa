"""
Standalone evaluation metrics for Brajavali forced alignment and corpus verification.

Capabilities:
  1. Standalone corpus audit (default when run with no alignment arguments):
     - Verifies 0.00% dictionary OOV across Phase 1 (408 words) and Phase 2 (1,963 words).
     - Computes Phase 2 train-to-test unseen vocabulary overlap (S2-S6 vs. held-out S1).
     - Measures empirical word-final schwa (/ɔ/) durations across the 24 spoken Praat
       gold TextGrids in gold_textgrids/.
  2. External boundary evaluation (--pred-dir):
     - Scores predicted MFA TextGrids against gold_textgrids/ using:
       * Word Boundary Error (WBE, ms; pooled and per-file means)
       * Word-scoped DP Levenshtein Phone Boundary Error (PBE, ms) and phone deletion rate
       * 1,000-resample cluster-robust 95% file bootstrap confidence intervals (seed=42)
     - Optional pairwise comparison (--compare-dir) runs paired two-sided Wilcoxon
       Signed-Rank tests across the 24 files.

Usage:
    python scripts/evaluate_alignment.py
    python scripts/evaluate_alignment.py --pred-dir path/to/output_textgrids --modality spoken
"""

import argparse
import csv
import math
import os
import re
import statistics

from g2p_rules import tokenize_text

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_textgrid(filepath: str) -> dict:
    """Reads a Praat TextGrid supporting UTF-8 and UTF-16 encodings."""
    content = None
    for enc in ("utf-8", "utf-16", "utf-16-be", "utf-16-le", "utf-8-sig", "latin-1"):
        try:
            with open(filepath, "r", encoding=enc) as f:
                content = f.read()
            break
        except UnicodeDecodeError:
            continue
    if content is None:
        raise ValueError(f"Could not decode TextGrid: {filepath}")

    tiers = {}
    tier_blocks = re.findall(
        r'class = "IntervalTier"\s+name = "([^"]+)"\s+xmin = ([\d\.]+)\s+xmax = ([\d\.]+)\s+intervals: size = (\d+)(.*?)(?=item \[\d+\]|\Z)',
        content,
        re.DOTALL,
    )
    for name, _, _, _, intervals_text in tier_blocks:
        intervals = re.findall(r'xmin = ([\d\.]+)\s+xmax = ([\d\.]+)\s+text = "(.*?)"', intervals_text)
        tiers[name.lower()] = [(float(xmin), float(xmax), text.strip()) for xmin, xmax, text in intervals]
    return tiers


def pair_phones_in_word(pred: list, gold: list) -> list:
    """DP Levenshtein phone alignment within a single word interval."""
    n, m = len(pred), len(gold)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sub = 0 if pred[i - 1][2] == gold[j - 1][2] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + sub)

    i, j, pairs = n, m, []
    while i > 0 and j > 0:
        sub = 0 if pred[i - 1][2] == gold[j - 1][2] else 1
        if dp[i][j] == dp[i - 1][j - 1] + sub:
            pairs.append((pred[i - 1], gold[j - 1]))
            i -= 1
            j -= 1
        elif dp[i][j] == dp[i - 1][j] + 1:
            i -= 1
        else:
            j -= 1
    return pairs[::-1]


def bootstrap_file_ci(file_means: list, n_bootstraps: int = 1000, seed: int = 42) -> tuple:
    """Cluster-robust 95% percentile bootstrap CI over file-level means."""
    import numpy as np

    rng = np.random.RandomState(seed)
    arr = np.array(file_means, dtype=float)
    n = len(arr)
    boot = [np.mean(arr[rng.randint(0, n, size=n)]) for _ in range(n_bootstraps)]
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def wilcoxon_test(x: list, y: list) -> tuple:
    """Paired two-sided Wilcoxon Signed-Rank test returning (W, z, p, r_b)."""
    import numpy as np
    import pandas as pd
    import scipy.stats

    diff = np.array(x) - np.array(y)
    nonzero = diff[diff != 0]
    n = len(nonzero)
    if n == 0:
        return 0.0, 0.0, 1.0, 0.0
    res = scipy.stats.wilcoxon(x, y, alternative="two-sided")
    abs_d = np.abs(nonzero)
    ranks = pd.Series(abs_d).rank().values
    w_pos = float(np.sum(ranks[nonzero > 0]))
    w_neg = float(np.sum(ranks[nonzero < 0]))
    mean_w = n * (n + 1) / 4.0
    std_w = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    z_score = (min(w_pos, w_neg) - mean_w + 0.5) / std_w if std_w > 0 else 0.0
    rb = (w_pos - w_neg) / (w_pos + w_neg) if (w_pos + w_neg) > 0 else 0.0
    return float(res.statistic), z_score, float(res.pvalue), rb


def holm_bonferroni(p_values: list) -> list:
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    m = len(p_values)
    adjusted = [0.0] * m
    cum_max = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        cum_max = max(cum_max, min(1.0, p * (m - rank)))
        adjusted[orig_idx] = cum_max
    return adjusted


def percentile_sorted(sorted_vals: list, q: float) -> float:
    idx = (len(sorted_vals) - 1) * q
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return sorted_vals[lo]
    return sorted_vals[lo] * (hi - idx) + sorted_vals[hi] * (idx - lo)


def run_corpus_summary() -> None:
    p1_tsv = os.path.join(REPO_ROOT, "transcripts", "phase1_120_phrases.tsv")
    p2_tsv = os.path.join(REPO_ROOT, "transcripts", "phase2_train_403_utterances.tsv")
    gold_dir = os.path.join(REPO_ROOT, "gold_textgrids")
    dict_path = os.path.join(REPO_ROOT, "dictionaries", "proposed.dict")

    with open(dict_path, "r", encoding="utf-8") as f:
        lexicon = set(line.split("\t", 1)[0] for line in f if line.strip())

    with open(p1_tsv, "r", encoding="utf-8", newline="") as f:
        p1_rows = list(csv.DictReader(f, delimiter="\t"))
    with open(p2_tsv, "r", encoding="utf-8", newline="") as f:
        p2_rows = list(csv.DictReader(f, delimiter="\t"))

    p1_tokens = [w for r in p1_rows for w in tokenize_text(r["text"])]
    p2_train_tokens = [w for r in p2_rows for w in tokenize_text(r["text"])]
    p1_types = set(p1_tokens)
    p2_train_types = set(p2_train_tokens)
    all_types = p1_types | p2_train_types

    unseen_types = p1_types - p2_train_types
    unseen_tokens = sum(1 for w in p1_tokens if w in unseen_types)

    print("=== Dictionary & Transcript Audit ===")
    print(f"Proposed dictionary entries: {len(lexicon)}")
    print(f"Phase 1 / Test S1 phrases:   {len(p1_rows)} ({len(p1_tokens)} tokens, {len(p1_types)} unique types)")
    print(f"Phase 2 Train (S2-S6):       {len(p2_rows)} ({len(p2_train_tokens)} tokens, {len(p2_train_types)} unique types)")
    print(f"Combined vocabulary types:   {len(all_types)} (Missing in dict: {len(all_types - lexicon)})")
    print(
        f"Phase 2 Unseen Test Vocab:   {len(unseen_types)}/{len(p1_types)} types "
        f"({len(unseen_types)/len(p1_types)*100:.2f}%), "
        f"{unseen_tokens}/{len(p1_tokens)} tokens ({unseen_tokens/len(p1_tokens)*100:.2f}%)"
    )

    schwa_durs = []
    spoken_tgs = sorted(fn for fn in os.listdir(gold_dir) if fn.startswith("spoken_") and fn.endswith(".TextGrid"))
    for fn in spoken_tgs:
        tiers = parse_textgrid(os.path.join(gold_dir, fn))
        words = [w for w in tiers.get("words", []) if w[2]]
        phones = [p for p in tiers.get("phones", []) if p[2]]
        for w in words:
            w_phs = [p for p in phones if w[0] <= (p[0] + p[1]) / 2.0 <= w[1]]
            if w_phs and w_phs[-1][2] == "ɔ":
                schwa_durs.append((w_phs[-1][1] - w_phs[-1][0]) * 1000.0)

    schwa_sorted = sorted(schwa_durs)
    mean_d = statistics.mean(schwa_sorted)
    std_d = statistics.stdev(schwa_sorted)
    med_d = statistics.median(schwa_sorted)
    q1 = percentile_sorted(schwa_sorted, 0.25)
    q3 = percentile_sorted(schwa_sorted, 0.75)
    ge100 = sum(1 for d in schwa_sorted if d >= 100.0)
    print("\n=== Empirical Word-Final Schwa (/ɔ/) Duration (24 Spoken Gold TextGrids) ===")
    print(f"Tokens measured:  {len(schwa_sorted)}")
    print(f"Mean ± SD:        {mean_d:.1f} ± {std_d:.1f} ms")
    print(f"Median [IQR]:     {med_d:.1f} [{q1:.1f}, {q3:.1f}] ms")
    print(f"Range:            [{schwa_sorted[0]:.1f}, {schwa_sorted[-1]:.1f}] ms")
    print(f"Sustained >=100ms:{ge100}/{len(schwa_sorted)} ({ge100/len(schwa_sorted)*100:.1f}%)")


def find_file(directory: str, target_filename: str):
    for root, _, files in os.walk(directory):
        for fn in files:
            if fn.lower() == target_filename.lower():
                return os.path.join(root, fn)
    return None


def score_predictions(pred_dir: str, modality: str) -> dict:
    gold_dir = os.path.join(REPO_ROOT, "gold_textgrids")
    prefix = f"{modality.lower()}_"
    gold_files = sorted(fn for fn in os.listdir(gold_dir) if fn.lower().startswith(prefix) and fn.endswith(".TextGrid"))

    file_wbe, file_pbe, all_wbe, all_pbe = {}, {}, [], []
    total_gold, unmatched_gold = 0, 0

    for g_file in gold_files:
        gold_tiers = parse_textgrid(os.path.join(gold_dir, g_file))
        m = re.search(r"(?:spoken|sung)_[^_]+_(.*)\.textgrid", g_file.lower())
        base_name = m.group(1) if m else os.path.splitext(g_file)[0]
        pred_tg = find_file(pred_dir, f"{base_name}.TextGrid")
        if not pred_tg:
            raise FileNotFoundError(f"Missing predicted TextGrid '{base_name}.TextGrid' under {pred_dir}")

        pred_tiers = parse_textgrid(pred_tg)
        gw_list = [w for w in gold_tiers.get("words", []) if w[2]]
        gp_list = [p for p in gold_tiers.get("phones", []) if p[2]]
        pw_list = [w for w in pred_tiers.get("words", []) if w[2]]
        pp_list = [p for p in pred_tiers.get("phones", []) if p[2]]

        f_w = []
        f_p = []
        for wi in range(min(len(gw_list), len(pw_list))):
            gw, pw = gw_list[wi], pw_list[wi]
            f_w.extend([abs(pw[0] - gw[0]) * 1000.0, abs(pw[1] - gw[1]) * 1000.0])

            g_phs = [p for p in gp_list if gw[0] <= (p[0] + p[1]) / 2.0 <= gw[1]]
            p_phs = [p for p in pp_list if pw[0] <= (p[0] + p[1]) / 2.0 <= pw[1]]
            total_gold += len(g_phs)
            pairs = pair_phones_in_word(p_phs, g_phs)
            unmatched_gold += len(g_phs) - len(pairs)
            for p_p, g_p in pairs:
                f_p.extend([abs(p_p[0] - g_p[0]) * 1000.0, abs(p_p[1] - g_p[1]) * 1000.0])

        all_wbe.extend(f_w)
        all_pbe.extend(f_p)
        file_wbe[base_name] = statistics.mean(f_w) if f_w else 0.0
        file_pbe[base_name] = statistics.mean(f_p) if f_p else 0.0

    return {
        "file_wbe": file_wbe,
        "file_pbe": file_pbe,
        "all_wbe": all_wbe,
        "all_pbe": all_pbe,
        "total_gold": total_gold,
        "unmatched_gold": unmatched_gold,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Brajavali G2P & Forced Alignment Metrics")
    parser.add_argument("--pred-dir", default=None, help="Directory containing predicted MFA TextGrids")
    parser.add_argument("--compare-dir", default=None, help="Optional baseline MFA TextGrid directory for paired Wilcoxon test")
    parser.add_argument("--modality", choices=["spoken", "sung"], default="spoken", help="Gold annotation subset to score")
    args = parser.parse_args()

    if not args.pred_dir:
        run_corpus_summary()
        return

    res = score_predictions(args.pred_dir, args.modality)
    w_vals = list(res["file_wbe"].values())
    w_pool = res["all_wbe"]
    ci_lo, ci_hi = bootstrap_file_ci(w_vals)
    p_vals = list(res["file_pbe"].values())
    p_pool = res["all_pbe"]
    del_rate = res["unmatched_gold"] / res["total_gold"] * 100.0 if res["total_gold"] else 0.0

    print(f"=== {args.modality.title()} Alignment Evaluation ({len(w_vals)} files, {len(w_pool)} word boundaries) ===")
    print(f"Per-file WBE: {statistics.mean(w_vals):.1f} ± {statistics.stdev(w_vals):.1f} ms (95% CI [{ci_lo:.1f}, {ci_hi:.1f}])")
    print(f"Pooled WBE:   {statistics.mean(w_pool):.1f} ± {statistics.stdev(w_pool):.1f} ms")
    for thr in (25.0, 50.0, 100.0):
        pct = sum(1 for v in w_pool if v <= thr) / len(w_pool) * 100.0
        print(f"  < {int(thr)} ms:  {pct:.1f}%")
    print(f"Per-file PBE: {statistics.mean(p_vals):.1f} ± {statistics.stdev(p_vals):.1f} ms (pooled {statistics.mean(p_pool):.1f} ms)")
    print(f"Gold Phone Deletions: {res['unmatched_gold']}/{res['total_gold']} ({del_rate:.1f}%)")

    if args.compare_dir:
        base = score_predictions(args.compare_dir, args.modality)
        keys = sorted(res["file_wbe"].keys())
        w_stat, z_val, p_val, rb = wilcoxon_test(
            [base["file_wbe"][k] for k in keys],
            [res["file_wbe"][k] for k in keys],
        )
        print(f"Paired Wilcoxon (compare vs. pred): W={w_stat:.1f}, z={z_val:.2f}, r_b={rb:.3f}, raw_p={p_val:.4e}")


if __name__ == "__main__":
    main()
