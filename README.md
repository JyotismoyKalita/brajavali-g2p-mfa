# Brajavali Rule-Based IPA G2P & Forced Alignment Resources

Non-audio linguistic resources, scripts, and manual gold annotations accompanying:

> Gaurav Narayan Baruah, Parismita Sarma, Jyotismoy Kalita, Bhagyasri Bora, *"A Rule-Based IPA G2P Pronunciation Dictionary for Brajavali: A Pilot Study Evaluated with Forced Alignment."*


## Repository Contents

```
brajavali-g2p-mfa/
├── README.md
├── LICENSE
├── dictionaries/                 # 4 ablation IPA lexicons (1,963 unique Brajavali words)
│   ├── proposed.dict             # Final schwa retained (/ɔ/) + conjunct decompositions
│   ├── schwa_only.dict           # Word-final schwa retained; 1:1 base consonant mapping
│   ├── conjunct_only.dict        # Word-final schwa deleted; conjunct decompositions active
│   └── naive_baseline.dict       # Word-final schwa deleted; 1:1 base consonant mapping
├── transcripts/                  # Complete orthographic phrase lists (523 spoken / 120 sung)
│   ├── phase1_120_phrases.tsv    # 120 Phase 1 devotional phrases (also Phase 2 held-out S1)
│   └── phase2_train_403_utterances.tsv  # 403 Phase 2 multi-speaker training utterances (S2-S6)
├── gold_textgrids/               # 48 manual Praat ground-truth TextGrids (24 spoken, 24 sung)
└── scripts/
    ├── g2p_rules.py              # Deterministic Brajavali IPA G2P engine (2x2 ablation variants)
    ├── generate_dictionaries.py  # Reconstructs all 4 .dict files from transcripts/
    └── evaluate_alignment.py     # Audits OOV/schwa durations and scores MFA TextGrids (WBE/PBE)
```

## G2P Lexicon Design (2×2 Ablation)

All four dictionaries share canonical Unicode nukta re-composition (`preprocess_assamese_text()` preserving `য় → /j/`, `ড় → /ɹ/`, `ঢ় → /ɹʱ/` across 213 nukta words in the 1,963-word vocabulary and 35 in Phase 1) and isolate two liturgical factors:

1. **Word-Final Schwa Retention (`SR`):** Retains inherent `/ɔ/` word-finally per classical 14-matra *Poyar* and 10+10+14-matra *Dulori* poetic meter (1,018 affected words). Single-consonant words (e.g., `ন → /n ɔ/`) preserve `/ɔ/` across all four arms.
2. **Conjunct Decomposition (`CD`):** Resolves `ক্ষ → /kʰ/` (24 words), subscript *ya-phala* `C + ্ + য → C + /j/` (42 words), and subscript *ba-phala* `C + ্ + ব → C + /w/` excluding homorganic nasals `{ম, ন}` (`অম্বৰ → /ɔ m b ɔ ɹ ɔ/`). `জ্ঞ → /ɡ j/` is also encoded in `g2p_rules.py` for orthographic completeness (0 tokens attested in the current 1,963-word sample).

| Arm File | Final Schwa (`/ɔ/`) | Conjunct Decomposition | Entries | Diff vs. `proposed.dict` |
| :--- | :---: | :---: | :---: | :---: |
| `dictionaries/proposed.dict` | Yes | Yes | 1,963 | — |
| `dictionaries/schwa_only.dict` | Yes | No | 1,963 | 110 words |
| `dictionaries/conjunct_only.dict` | No | Yes | 1,963 | 1,018 words |
| `dictionaries/naive_baseline.dict` | No | No | 1,963 | 1,055 words |

*(Every entry in the 408-word Phase 1 subset is an exact subset of the corresponding 1,963-word file above.)*

## Usage

### 1. Regenerate Dictionaries from Transcripts
Requires Python 3.8+ (standard library only):

```bash
python scripts/generate_dictionaries.py
# Optional: regenerate 408-word Phase 1 subset or unpack individual MFA .txt files
python scripts/generate_dictionaries.py --phase 1 --export-txt mfa_transcripts
```

### 2. Run Standalone Corpus & Gold-Annotation Audit
Runs with standard Python 3 (zero external dependencies required for summary mode):

```bash
python scripts/evaluate_alignment.py
```

To score predicted MFA `.TextGrid` files against `gold_textgrids/` (requires `numpy`, and `scipy` + `pandas` for `--compare-dir`):

```bash
python scripts/evaluate_alignment.py --pred-dir path/to/output_textgrids --modality spoken
```

### 3. Montreal Forced Aligner Parameters
Experiments used MFA v3.4.3 (`train_g2p` disabled; acoustic training from scratch, 16 kHz mono WAV, 13-dim MFCCs + LDA + MLLT + SAT fMLLR) with `beam=10`, `retry_beam=40`, `transition_scale=1.0`, `acoustic_scale=0.083333`, `self_loop_scale=0.1`, `boost_silence=1.0`, and global seed `0`.

## Audio Data Availability & Licensing

- **Audio:** The 643 Brajavali audio recordings (523 spoken, 120 sung) are **not included** due to participant written consent terms and Sattra community stewardship permissions; audio is available from the corresponding author (`gauravbaruah@gauhati.ac.in`) upon reasonable request.
- **License:** Scripts are licensed under the [MIT License](LICENSE). Non-audio linguistic resources (`dictionaries/`, `transcripts/`, `gold_textgrids/`) are released for research and educational use with attribution.
