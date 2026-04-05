"""
EDA (Easy Data Augmentation) - synonym replacement, random insertion,
random swap, random deletion. Uses NLTK WordNet for synonyms.
"""

import random
from collections import Counter

import nltk
from nltk.corpus import wordnet, stopwords

for resource in ["wordnet", "omw-1.4", "stopwords", "averaged_perceptron_tagger",
                 "averaged_perceptron_tagger_eng", "punkt", "punkt_tab"]:
    nltk.download(resource, quiet=True)

STOP_WORDS = set(stopwords.words("english"))


def _get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            candidate = lemma.name().replace("_", " ").lower()
            if candidate != word.lower():
                synonyms.add(candidate)
    return list(synonyms)


def synonym_replacement(words, n):
    new_words = words.copy()
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


def random_insertion(words, n):
    new_words = words.copy()
    for _ in range(n):
        candidates = [w for w in new_words if w.lower() not in STOP_WORDS and w.isalpha()]
        if not candidates:
            break
        random.shuffle(candidates)
        for word in candidates:
            synonyms = _get_synonyms(word)
            if synonyms:
                new_words.insert(random.randint(0, len(new_words)), random.choice(synonyms))
                break
    return new_words


def random_swap(words, n):
    new_words = words.copy()
    if len(new_words) < 2:
        return new_words
    for _ in range(n):
        i, j = random.sample(range(len(new_words)), 2)
        new_words[i], new_words[j] = new_words[j], new_words[i]
    return new_words


def random_deletion(words, p):
    if len(words) <= 1:
        return words
    new_words = [w for w in words if random.random() > p]
    return new_words if new_words else [random.choice(words)]


def eda(text, alpha_sr=0.1, alpha_ri=0.1, alpha_rs=0.1, p_rd=0.1, num_aug=4):
    """Apply all 4 EDA ops and return num_aug augmented sentences."""
    words = text.split()
    num_words = len(words)

    n_sr = max(1, int(alpha_sr * num_words))
    n_ri = max(1, int(alpha_ri * num_words))
    n_rs = max(1, int(alpha_rs * num_words))

    augmented = set()
    per_technique = max(1, num_aug // 4)

    for _ in range(per_technique):
        augmented.add(" ".join(synonym_replacement(words, n_sr)))
    for _ in range(per_technique):
        augmented.add(" ".join(random_insertion(words, n_ri)))
    for _ in range(per_technique):
        augmented.add(" ".join(random_swap(words, n_rs)))
    for _ in range(per_technique):
        augmented.add(" ".join(random_deletion(words, p_rd)))

    augmented.discard(text)
    result = list(augmented)
    random.shuffle(result)
    return result[:num_aug]


def augment_dataset(texts, labels, num_aug=4, alpha=0.1):
    """Augment every sample. Returns only the new augmented examples."""
    aug_texts, aug_labels = [], []
    for text, label in zip(texts, labels):
        for aug_text in eda(text, alpha_sr=alpha, alpha_ri=alpha, alpha_rs=alpha,
                            p_rd=alpha, num_aug=num_aug):
            aug_texts.append(aug_text)
            aug_labels.append(label)
    return aug_texts, aug_labels


def augment_minority_classes(texts, labels, target_count=None, alpha=0.1):
    """Oversample minority classes with EDA until they reach target_count (default: majority count)."""
    class_counts = Counter(labels)
    if target_count is None:
        target_count = max(class_counts.values())

    # group texts by label
    class_texts = {}
    for text, label in zip(texts, labels):
        class_texts.setdefault(label, []).append(text)

    aug_texts, aug_labels = [], []

    for label, count in class_counts.items():
        if count >= target_count:
            continue

        needed = target_count - count
        source = class_texts[label]
        num_aug_per = max(1, needed // count + 1)

        generated = 0
        for text in source:
            if generated >= needed:
                break
            for aug_text in eda(text, alpha_sr=alpha, alpha_ri=alpha, alpha_rs=alpha,
                                p_rd=alpha, num_aug=num_aug_per):
                if generated >= needed:
                    break
                aug_texts.append(aug_text)
                aug_labels.append(label)
                generated += 1

    return aug_texts, aug_labels
