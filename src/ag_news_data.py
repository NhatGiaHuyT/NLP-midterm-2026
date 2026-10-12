"""Pipeline dữ liệu AG News (Task 1): tải, làm sạch, chia tập và xuất ra data/ag_news/.

Chạy từ thư mục gốc repo:  python src/ag_news_data.py
"""
import html
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.model_selection import train_test_split

SEED = 42
VALIDATION_SIZE = 0.10
MIN_FREQ = 2
DATASET_ID = "fancyzhx/ag_news"
LABEL_NAMES = ["World", "Sports", "Business", "Sci/Tech"]
SPLITS = ["train", "val", "test"]
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "ag_news"
EXPORT_COLUMNS = ["source_index", "label", "label_name", "text", "text_clean", "text_nostop", "n_tokens"]

ENTITY_NAMES = "quot|amp|lt|gt|nbsp|hellip|rsquo|lsquo|rdquo|ldquo|mdash|ndash"
PUNCT_MAP = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "—": " - ", "–": " - "})
TOKEN_PATTERN = re.compile(r"(?:[a-z]\.){2,}|[a-z0-9]+(?:'[a-z]+)?")
STOP_WORDS = frozenset(ENGLISH_STOP_WORDS)


def clean_text(text: str) -> str:
    text = re.sub(r"(?<!&)#(\d+);", r"&#\1;", text)
    text = re.sub(rf"(?<![&\w])({ENTITY_NAMES});", r"&\1;", text)
    text = html.unescape(html.unescape(text))
    text = unicodedata.normalize("NFKC", text).translate(PUNCT_MAP).replace("''", '"')
    text = re.sub(r"<[^>]*>", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = text.replace("\\", " ")
    text = re.sub(r"(\w) '(s|t|re|ve|ll|d|m)\b", r"\1'\2", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text)


def remove_stopwords(text: str) -> str:
    return " ".join(t for t in tokenize(text) if t not in STOP_WORDS)


def load_splits() -> dict[str, pd.DataFrame]:
    """Trả về train/val/test thô; val là 10% train gốc, chia stratified."""
    dataset = load_dataset(DATASET_ID)
    assert dataset["train"].features["label"].names == LABEL_NAMES

    train_full = dataset["train"].to_pandas()
    test = dataset["test"].to_pandas()
    for frame in (train_full, test):
        frame["source_index"] = np.arange(len(frame))
        frame["label_name"] = frame["label"].map(dict(enumerate(LABEL_NAMES)))

    # Gọi y hệt notebook model để hai bên ra cùng một phép chia
    train_idx, val_idx = train_test_split(
        np.arange(len(train_full)), test_size=VALIDATION_SIZE,
        random_state=SEED, stratify=train_full["label"].to_numpy(),
    )
    return {
        "train": train_full.iloc[train_idx].reset_index(drop=True),
        "val": train_full.iloc[val_idx].reset_index(drop=True),
        "test": test,
    }


def preprocess(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["text_clean"] = frame["text"].map(clean_text)
    frame["text_nostop"] = frame["text_clean"].map(remove_stopwords)
    frame["n_tokens"] = frame["text_clean"].map(lambda t: len(tokenize(t)))
    return frame


def compute_stats(frames: dict[str, pd.DataFrame]) -> dict:
    train = frames["train"]
    token_counts = Counter(tok for text in train["text_clean"] for tok in tokenize(text))
    vocab = {tok for tok, c in token_counts.items() if c >= MIN_FREQ}

    def oov_rate(texts):
        tokens = [tok for text in texts for tok in tokenize(text)]
        return round(sum(tok not in vocab for tok in tokens) / len(tokens), 4)

    max_len = int(np.percentile(train["n_tokens"], 95))
    return {
        "split_sizes": {k: len(f) for k, f in frames.items()},
        "unique_tokens_train": len(token_counts),
        f"vocab_size_min_freq_{MIN_FREQ}_with_pad_unk": len(vocab) + 2,
        "oov_rate_val": oov_rate(frames["val"]["text_clean"]),
        "oov_rate_test": oov_rate(frames["test"]["text_clean"]),
        "tokens_per_text_mean": round(float(train["n_tokens"].mean()), 2),
        "tokens_per_text_max": int(train["n_tokens"].max()),
        "max_len_p95": max_len,
        "texts_truncated_at_max_len_pct": round(float((train["n_tokens"] > max_len).mean() * 100), 2),
    }


def load_processed(output_dir: Path = OUTPUT_DIR) -> dict[str, pd.DataFrame]:
    return {split: pd.read_csv(output_dir / f"{split}.csv", keep_default_na=False) for split in SPLITS}


def main() -> None:
    frames = {split: preprocess(frame) for split, frame in load_splits().items()}
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for split, frame in frames.items():
        frame[EXPORT_COLUMNS].to_csv(OUTPUT_DIR / f"{split}.csv", index=False)

    summary = {
        "dataset": DATASET_ID,
        "label_names": LABEL_NAMES,
        "seed": SEED,
        "validation_size": VALIDATION_SIZE,
        **compute_stats(frames),
    }
    (OUTPUT_DIR / "data_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"Saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
