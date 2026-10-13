"""Pipeline dữ liệu CoNLL-2003 (Task 2): tải, kiểm tra, tạo vocab và xuất ra data/conll2003/.

Chạy từ thư mục gốc repo:  python src/conll_data.py
"""
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd
from datasets import load_dataset

DATASET_ID = "extraordinarylab/conll2003"
SPLITS = ["train", "val", "test"]
ENTITY_TYPES = ["PER", "ORG", "LOC", "MISC"]
BIO_TAGS = ["O"] + [f"{p}-{t}" for t in ENTITY_TYPES for p in ("B", "I")]
BIOES_TAGS = ["O"] + [f"{p}-{t}" for t in ENTITY_TYPES for p in ("B", "I", "E", "S")]
PAD, UNK = "<PAD>", "<UNK>"
PAD_LABEL_ID = -100  # CrossEntropyLoss(ignore_index=-100) bỏ qua vị trí padding
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "conll2003"


def normalize_word(word: str) -> str:
    return re.sub(r"\d", "0", word.lower())


def get_entities(tags: list[str]) -> list[tuple[str, int, int]]:
    """Trích các thực thể (type, start, end) từ chuỗi nhãn IOB2, end không bao gồm."""
    entities, start, ent_type = [], None, None
    for i, tag in enumerate(tags + ["O"]):
        if start is not None and not (tag.startswith("I-") and tag[2:] == ent_type):
            entities.append((ent_type, start, i))
            start = None
        if tag.startswith("B-") or (tag.startswith("I-") and start is None):
            start, ent_type = i, tag[2:]
    return entities


def bio_to_bioes(tags: list[str]) -> list[str]:
    bioes = ["O"] * len(tags)
    for ent_type, start, end in get_entities(tags):
        if end - start == 1:
            bioes[start] = f"S-{ent_type}"
        else:
            bioes[start] = f"B-{ent_type}"
            bioes[start + 1:end - 1] = [f"I-{ent_type}"] * (end - start - 2)
            bioes[end - 1] = f"E-{ent_type}"
    return bioes


def bioes_to_bio(tags: list[str]) -> list[str]:
    return [{"E": "I", "S": "B"}.get(t[0], t[0]) + t[1:] for t in tags]


def count_invalid_bio(tags: list[str]) -> int:
    # I-X hợp lệ khi đứng sau B-X hoặc I-X
    prev, invalid = "O", 0
    for tag in tags:
        if tag.startswith("I-") and prev[2:] != tag[2:]:
            invalid += 1
        prev = tag
    return invalid


def load_splits() -> dict[str, pd.DataFrame]:
    dataset = load_dataset(DATASET_ID)
    frames = {}
    for split, hf_split in zip(SPLITS, ["train", "validation", "test"]):
        frame = dataset[hf_split].to_pandas()
        frame["tokens"] = frame["tokens"].map(list)
        frame["ner_tags"] = frame["ner_tags"].map(list)
        frame = frame[frame["tokens"].map(lambda t: len(t) > 0 and t[0] != "-DOCSTART-")].reset_index(drop=True)

        assert (frame["tokens"].map(len) == frame["ner_tags"].map(len)).all(), f"{split}: token/tag length mismatch"
        assert set(t for tags in frame["ner_tags"] for t in tags) <= set(BIO_TAGS), f"{split}: unknown tag"
        frame["ner_tags_bioes"] = frame["ner_tags"].map(bio_to_bioes)
        frames[split] = frame
    return frames


def build_vocabs(train: pd.DataFrame) -> dict[str, dict[str, int]]:
    words = Counter(normalize_word(w) for tokens in train["tokens"] for w in tokens)
    chars = Counter(c for tokens in train["tokens"] for w in tokens for c in w)
    return {
        "word2idx": {PAD: 0, UNK: 1, **{w: i + 2 for i, (w, _) in enumerate(words.most_common())}},
        "char2idx": {PAD: 0, UNK: 1, **{c: i + 2 for i, (c, _) in enumerate(chars.most_common())}},
        "tag2idx": {t: i for i, t in enumerate(BIO_TAGS)},
        "tag2idx_bioes": {t: i for i, t in enumerate(BIOES_TAGS)},
    }


def oov_rates(train: pd.DataFrame, frame: pd.DataFrame) -> dict[str, float]:
    variants = {"raw": lambda w: w, "lower": str.lower, "lower_digit0": normalize_word}
    rates = {}
    for name, fn in variants.items():
        vocab = {fn(w) for tokens in train["tokens"] for w in tokens}
        all_words = [fn(w) for tokens in frame["tokens"] for w in tokens]
        entity_words = [fn(w) for tokens, tags in zip(frame["tokens"], frame["ner_tags"])
                        for w, t in zip(tokens, tags) if t != "O"]
        rates[f"{name}_all_tokens"] = round(sum(w not in vocab for w in all_words) / len(all_words), 4)
        rates[f"{name}_entity_tokens"] = round(sum(w not in vocab for w in entity_words) / len(entity_words), 4)
    return rates


def compute_stats(frames: dict[str, pd.DataFrame], vocabs: dict) -> dict:
    stats = {}
    for split, frame in frames.items():
        entities = Counter(e[0] for tags in frame["ner_tags"] for e in get_entities(tags))
        stats[split] = {
            "sentences": len(frame),
            "tokens": int(frame["tokens"].map(len).sum()),
            "max_sentence_len": int(frame["tokens"].map(len).max()),
            "entities": {t: entities[t] for t in ENTITY_TYPES},
            "invalid_bio_tags": int(frame["ner_tags"].map(count_invalid_bio).sum()),
        }
        if split != "train":
            stats[split]["oov"] = oov_rates(frames["train"], frame)
    return {
        "vocab_sizes": {k: len(v) for k, v in vocabs.items()},
        "splits": stats,
    }


def load_processed(output_dir: Path = OUTPUT_DIR) -> tuple[dict[str, pd.DataFrame], dict]:
    frames = {split: pd.read_json(output_dir / f"{split}.jsonl", lines=True) for split in SPLITS}
    vocabs = json.loads((output_dir / "vocab.json").read_text(encoding="utf-8"))
    return frames, vocabs


def main() -> None:
    frames = load_splits()
    vocabs = build_vocabs(frames["train"])
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for split, frame in frames.items():
        frame[["tokens", "ner_tags", "ner_tags_bioes"]].to_json(
            OUTPUT_DIR / f"{split}.jsonl", orient="records", lines=True, force_ascii=False)
    (OUTPUT_DIR / "vocab.json").write_text(json.dumps(vocabs, ensure_ascii=False), encoding="utf-8")

    summary = {
        "dataset": DATASET_ID,
        "entity_types": ENTITY_TYPES,
        "pad_label_id": PAD_LABEL_ID,
        **compute_stats(frames, vocabs),
    }
    (OUTPUT_DIR / "data_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"Saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
