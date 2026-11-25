"""
Grapheme-to-Phoneme (G2P) Rule Engine for Brajavali Liturgical Recitation
=========================================================================

Converts Assamese-script Brajavali orthography into space-delimited IPA phone
sequences compatible with the Montreal Forced Aligner (MFA v3.4.3).

Four 2x2 factorial variants are implemented:
  1. Proposed (retain_final_schwa=True, decompose_conjuncts=True)
     - Retains word-final inherent /ɔ/ per classical poetic meter.
     - Decomposes Sanskritic conjuncts:
       * ক্ষ -> /kʰ/
       * জ্ঞ -> /ɡ j/ (encoded for orthographic completeness; 0 tokens in sample)
       * য-ফলা (C + ্ + য) -> C + /j/
       * ব-ফলা (C + ্ + ব, C not in {ম, ন}) -> C + /w/
  2. Schwa-Only (retain_final_schwa=True, decompose_conjuncts=False)
  3. Conjunct-Only (retain_final_schwa=False, decompose_conjuncts=True)
  4. Naive Baseline (retain_final_schwa=False, decompose_conjuncts=False)

Canonical Unicode nukta re-composition (য়, ড়, ঢ়) runs identically in
preprocessing across all four variants.
"""

import os
import re
import unicodedata

# Base consonant inventory mapping to IPA
CONSONANTS = {
    # Velar stops
    'ক': 'k', 'খ': 'kʰ', 'গ': 'ɡ', 'ঘ': 'ɡʱ', 'ঙ': 'ŋ',
    # Alveolar fricatives (de-affricated) and palatal nasal
    'চ': 's', 'ছ': 'sʰ', 'জ': 'z', 'ঝ': 'zʱ', 'ঞ': 'n',
    # Alveolar stops (merged historical retroflex)
    'ট': 't', 'ঠ': 'tʰ', 'ড': 'd', 'ঢ': 'dʱ', 'ণ': 'n',
    # Alveolar stops (merged historical dental)
    'ত': 't', 'থ': 'tʰ', 'দ': 'd', 'ধ': 'dʱ', 'ন': 'n',
    # Bilabial stops
    'প': 'p', 'ফ': 'pʰ', 'ব': 'b', 'ভ': 'bʱ', 'ম': 'm',
    # Glides and liquids
    'য': 'z', 'ৰ': 'ɹ', 'ল': 'l', 'ৱ': 'w',
    # Sibilants (merged into voiceless velar fricative /x/)
    'শ': 'x', 'ষ': 'x', 'স': 'x',
    # Glottal fricative
    'হ': 'ɦ',
    # Re-composed nukta graphemes and khanda ta
    'য়': 'j', 'য়': 'j', 'ড়': 'ɹ', 'ঢ়': 'ɹʱ', 'ৎ': 't',
    # Anusvara and visarga
    'ং': 'ŋ', 'ঃ': 'ɦ',
}

# Dependent vowel signs (matras)
VOWEL_SIGNS = {
    'া': 'a',
    'ি': 'i',
    'ী': 'i',
    'ু': 'u',
    'ূ': 'u',
    'ৃ': 'ɹ i',
    'ে': 'e',
    'ৈ': 'oɪ',
    'ো': 'o',
    'ৌ': 'oʊ',
}

# Standalone (independent) vowels
STANDALONE_VOWELS = {
    'অ': 'ɔ',
    'আ': 'a',
    'ই': 'i',
    'ঈ': 'i',
    'উ': 'u',
    'ঊ': 'u',
    'ঋ': 'ɹ i',
    'এ': 'e',
    'ঐ': 'oɪ',
    'ও': 'o',
    'ঔ': 'oʊ',
}

INHERENT_VOWEL = 'ɔ'
HALANT = '্'
CANDRABINDU = 'ঁ'

PUNCTUATION_REGEX = r'[।|!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]'


def preprocess_assamese_text(text: str) -> str:
    """
    Applies Unicode NFC normalization and re-composes nukta-bearing graphemes
    (U+09DF য়, U+09DC ড়, U+09DD ঢ়) decomposed by canonical normalization.
    """
    text = unicodedata.normalize('NFC', text)
    return (
        text.replace('\u09af\u09bc', 'য়')
            .replace('\u09a1\u09bc', 'ড়')
            .replace('\u09a2\u09bc', 'ঢ়')
    )


def convert_word_to_ipa(
    word: str,
    retain_final_schwa: bool = True,
    decompose_conjuncts: bool = True,
) -> str:
    """Converts a single Brajavali word token to a space-separated IPA string."""
    word = preprocess_assamese_text(word)

    phones = []
    i = 0
    n = len(word)

    consonant_count = sum(1 for c in word if c in CONSONANTS)
    has_vowel_sign = any(c in VOWEL_SIGNS or c in STANDALONE_VOWELS for c in word)
    is_monosyllabic_consonant = (consonant_count <= 1 and not has_vowel_sign)

    while i < n:
        char = word[i]

        # Standalone vowel
        if char in STANDALONE_VOWELS:
            phones.append(STANDALONE_VOWELS[char])

        # Special conjunct 'ক্ষ' -> /kʰ/ (or /k x/ when decompose_conjuncts=False)
        elif decompose_conjuncts and char == 'ক' and i + 2 < n and word[i + 1] == HALANT and word[i + 2] == 'ষ':
            phones.append('kʰ')
            i += 2
            if i + 1 < n and word[i + 1] in VOWEL_SIGNS:
                phones.append(VOWEL_SIGNS[word[i + 1]])
                i += 1
            elif i + 1 < n and word[i + 1] == HALANT:
                i += 1
            elif i + 1 < n or retain_final_schwa:
                phones.append(INHERENT_VOWEL)

        # Special conjunct 'জ্ঞ' -> /ɡ j/ (or /z n/ when decompose_conjuncts=False)
        elif decompose_conjuncts and char == 'জ' and i + 2 < n and word[i + 1] == HALANT and word[i + 2] == 'ঞ':
            phones.extend(['ɡ', 'j'])
            i += 2
            if i + 1 < n and word[i + 1] in VOWEL_SIGNS:
                phones.append(VOWEL_SIGNS[word[i + 1]])
                i += 1
            elif i + 1 < n and word[i + 1] == HALANT:
                i += 1
            elif i + 1 < n or retain_final_schwa:
                phones.append(INHERENT_VOWEL)

        # Consonants and following diacritics / conjuncts
        elif char in CONSONANTS:
            phones.append(CONSONANTS[char])

            if i + 1 < n:
                next_char = word[i + 1]

                # Subscript ya-phala (্ + য) -> /j/
                if decompose_conjuncts and next_char == HALANT and i + 2 < n and word[i + 2] == 'য':
                    phones.append('j')
                    i += 2
                    if i + 1 < n and word[i + 1] in VOWEL_SIGNS:
                        phones.append(VOWEL_SIGNS[word[i + 1]])
                        i += 1
                    elif i + 1 < n and word[i + 1] == HALANT:
                        i += 1
                    elif i + 1 < n or retain_final_schwa:
                        phones.append(INHERENT_VOWEL)

                # Subscript ba-phala (্ + ব, excluding homorganic nasals ম/ন) -> /w/
                elif decompose_conjuncts and next_char == HALANT and i + 2 < n and word[i + 2] == 'ব' and char not in ('ম', 'ন'):
                    phones.append('w')
                    i += 2
                    if i + 1 < n and word[i + 1] in VOWEL_SIGNS:
                        phones.append(VOWEL_SIGNS[word[i + 1]])
                        i += 1
                    elif i + 1 < n and word[i + 1] == HALANT:
                        i += 1
                    elif i + 1 < n or retain_final_schwa:
                        phones.append(INHERENT_VOWEL)

                elif next_char in VOWEL_SIGNS:
                    phones.append(VOWEL_SIGNS[next_char])
                    i += 1

                elif next_char == HALANT:
                    i += 1

                elif next_char == CANDRABINDU:
                    i += 1
                    if i + 1 < n and word[i + 1] in VOWEL_SIGNS:
                        phones.append(VOWEL_SIGNS[word[i + 1]])
                        i += 1
                    else:
                        phones.append(INHERENT_VOWEL)

                else:
                    phones.append(INHERENT_VOWEL)

            else:
                if retain_final_schwa or is_monosyllabic_consonant:
                    phones.append(INHERENT_VOWEL)

        elif char == CANDRABINDU:
            pass

        i += 1

    return " ".join(phones)


def brajavali_liturgical_g2p(word: str) -> str:
    """Proposed G2P: word-final schwa retention AND conjunct decomposition."""
    return convert_word_to_ipa(word, retain_final_schwa=True, decompose_conjuncts=True)


def assamese_naive_g2p(word: str) -> str:
    """Naive Baseline G2P: no final schwa retention, no conjunct decomposition."""
    return convert_word_to_ipa(word, retain_final_schwa=False, decompose_conjuncts=False)


def ablation_schwa_only_g2p(word: str) -> str:
    """Schwa-Only Ablation: final schwa retention without conjunct decomposition."""
    return convert_word_to_ipa(word, retain_final_schwa=True, decompose_conjuncts=False)


def ablation_conjunct_only_g2p(word: str) -> str:
    """Conjunct-Only Ablation: conjunct decomposition without final schwa retention."""
    return convert_word_to_ipa(word, retain_final_schwa=False, decompose_conjuncts=True)


def tokenize_text(text: str) -> list:
    """Strips punctuation from an utterance string and returns surface word tokens."""
    return re.sub(PUNCTUATION_REGEX, ' ', text).split()


def extract_vocabulary_from_corpus(corpus_dir: str) -> list:
    """Scans a directory tree of .txt transcripts and returns sorted unique words."""
    words = set()
    for root, _, files in os.walk(corpus_dir):
        for fn in files:
            if fn.endswith('.txt'):
                with open(os.path.join(root, fn), 'r', encoding='utf-8') as tf:
                    words.update(tokenize_text(tf.read()))
    return sorted(words)
