"""
Easy Data Augmentation (EDA) for Text Classification
=====================================================
Implements four augmentation techniques from Wei & Zou (2019):
  1. Synonym Replacement (SR)
  2. Random Insertion (RI)
  3. Random Swap (RS)
  4. Random Deletion (RD)

Uses NLTK WordNet for synonym lookups.
"""

import random
import re
from typing import List, Optional

import nltk
from nltk.corpus import wordnet, stopwords

# Download required NLTK data (only runs once)
for resource in ["wordnet", "omw-1.4", "stopwords", "averaged_perceptron_tagger",
                 "averaged_perceptron_tagger_eng", "punkt", "punkt_tab"]:
    nltk.download(resource, quiet=True)

STOP_WORDS = set(stopwords.words("english"))


# ---------------------------------------------------------------------------
# Helper: get synonyms from WordNet
# ---------------------------------------------------------------------------
def _get_synonyms(word: str) -> List[str]:
    """Return a list of unique synonyms for the given word (excluding itself)."""
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            candidate = lemma.name().replace("_", " ").lower()
            if candidate != word.lower():
                synonyms.add(candidate)
    return list(synonyms)


def _tokenize(text: str) -> List[str]:
    """Simple whitespace + punctuation-aware tokenizer."""
    return text.split()


# ---------------------------------------------------------------------------
# 1. Synonym Replacement
# ---------------------------------------------------------------------------
def synonym_replacement(words: List[str], n: int) -> List[str]:
    """Replace n random non-stopword words with one of their WordNet synonyms."""
    new_words = words.copy()
    # Candidate words: not stopwords and alphabetic
    candidates = [w for w in new_words if w.lower() not in STOP_WORDS and w.isalpha()]
    random.shuffle(candidates)
    num_replaced = 0
    for word in candidates:
        synonyms = _get_synonyms(word)
        if synonyms:
            synonym = random.choice(synonyms)
            new_words = [synonym if w == word else w for w in new_words]
            num_replaced += 1
        if num_replaced >= n:
            break
    return new_words


# ---------------------------------------------------------------------------
# 2. Random Insertion
# ---------------------------------------------------------------------------
def random_insertion(words: List[str], n: int) -> List[str]:
    """Insert n random synonyms of random words at random positions."""
    new_words = words.copy()
    for _ in range(n):
        _add_word(new_words)
    return new_words


def _add_word(words: List[str]) -> None:
    """Pick a random non-stopword, find a synonym, insert at a random position."""
    candidates = [w for w in words if w.lower() not in STOP_WORDS and w.isalpha()]
    if not candidates:
        return
    random.shuffle(candidates)
    for word in candidates:
        synonyms = _get_synonyms(word)
        if synonyms:
            words.insert(random.randint(0, len(words)), random.choice(synonyms))
            return


# ---------------------------------------------------------------------------
# 3. Random Swap
# ---------------------------------------------------------------------------
def random_swap(words: List[str], n: int) -> List[str]:
    """Randomly swap two words n times."""
    new_words = words.copy()
    if len(new_words) < 2:
        return new_words
    for _ in range(n):
        i, j = random.sample(range(len(new_words)), 2)
        new_words[i], new_words[j] = new_words[j], new_words[i]
    return new_words


# ---------------------------------------------------------------------------
# 4. Random Deletion
# ---------------------------------------------------------------------------
def random_deletion(words: List[str], p: float) -> List[str]:
    """Randomly delete each word with probability p."""
    if len(words) <= 1:
        return words
    new_words = [w for w in words if random.random() > p]
    # If everything got deleted, return one random word
    return new_words if new_words else [random.choice(words)]


# ---------------------------------------------------------------------------
# Main EDA function
# ---------------------------------------------------------------------------
def eda(
    text: str,
    alpha_sr: float = 0.1,
    alpha_ri: float = 0.1,
    alpha_rs: float = 0.1,
    p_rd: float = 0.1,
    num_aug: int = 4,
) -> List[str]:
    """
    Apply EDA to a single sentence and return `num_aug` augmented versions.

    Args:
        text: Input sentence.
        alpha_sr: Fraction of words to replace with synonyms.
        alpha_ri: Fraction of words to insert.
        alpha_rs: Fraction of words to swap.
        p_rd: Probability of deleting each word.
        num_aug: Total number of augmented sentences to generate.

    Returns:
        List of augmented sentence strings.
    """
    words = _tokenize(text)
    num_words = len(words)

    # Number of words to change for SR, RI, RS
    n_sr = max(1, int(alpha_sr * num_words))
    n_ri = max(1, int(alpha_ri * num_words))
    n_rs = max(1, int(alpha_rs * num_words))

    augmented = set()

    # Generate roughly num_aug / 4 from each technique, then fill remainder
    per_technique = max(1, num_aug // 4)

    for _ in range(per_technique):
        aug = synonym_replacement(words, n_sr)
        augmented.add(" ".join(aug))

    for _ in range(per_technique):
        aug = random_insertion(words, n_ri)
        augmented.add(" ".join(aug))

    for _ in range(per_technique):
        aug = random_swap(words, n_rs)
        augmented.add(" ".join(aug))

    for _ in range(per_technique):
        aug = random_deletion(words, p_rd)
        augmented.add(" ".join(aug))

    # Remove the original if it ended up in the set
    augmented.discard(text)

    # Trim to num_aug
    result = list(augmented)
    random.shuffle(result)
    return result[:num_aug]


def augment_dataset(
    texts: List[str],
    labels: List[str],
    num_aug: int = 4,
    alpha: float = 0.1,
) -> tuple:
    """
    Augment an entire dataset. Returns (augmented_texts, augmented_labels)
    containing ONLY the new augmented examples (not the originals).

    Args:
        texts: List of original text strings.
        labels: Corresponding labels.
        num_aug: Number of augmented versions per original sample.
        alpha: EDA intensity parameter (used for SR, RI, RS, RD).

    Returns:
        Tuple of (new_texts, new_labels).
    """
    aug_texts = []
    aug_labels = []
    for text, label in zip(texts, labels):
        augmented = eda(text, alpha_sr=alpha, alpha_ri=alpha, alpha_rs=alpha,
                        p_rd=alpha, num_aug=num_aug)
        for aug_text in augmented:
            aug_texts.append(aug_text)
            aug_labels.append(label)
    return aug_texts, aug_labels
