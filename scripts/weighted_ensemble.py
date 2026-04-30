import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

X = train["TEXT"].fillna("")
y = train["LABEL"]
X_test = test["TEXT"].fillna("")


def make_model(
    C=5,
    word_ngram=(1, 3),
    char_ngram=(3, 5),
    max_word=120000,
    max_char=140000,
    min_df=2
):
    features = FeatureUnion([
        ("word_tfidf", TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            analyzer="word",
            ngram_range=word_ngram,
            max_features=max_word,
            min_df=min_df,
            sublinear_tf=True
        )),
        ("char_tfidf", TfidfVectorizer(
            lowercase=True,
            analyzer="char_wb",
            ngram_range=char_ngram,
            max_features=max_char,
            min_df=min_df,
            sublinear_tf=True
        ))
    ])

    return Pipeline([
        ("features", features),
        ("clf", LogisticRegression(
            C=C,
            max_iter=5000,
            class_weight="balanced",
            solver="liblinear",
            random_state=42
        ))
    ])


configs = [
    ("C5_word123_char35", 5.0, (1, 3), (3, 5), 120000, 140000, 2),
    ("C5_5_word123_char35", 5.5, (1, 3), (3, 5), 120000, 140000, 2),
    ("C4_word123_char46", 4.0, (1, 3), (4, 6), 120000, 140000, 2),
]

probs = []

for name, C, word_ngram, char_ngram, max_word, max_char, min_df in configs:
    print("Training:", name)

    model = make_model(
        C=C,
        word_ngram=word_ngram,
        char_ngram=char_ngram,
        max_word=max_word,
        max_char=max_char,
        min_df=min_df
    )

    model.fit(X, y)
    probs.append(model.predict_proba(X_test))


# Weighted blends.
# C5 gets the largest weight because it has your best public score and best CV.
blend_configs = {
    "weighted_70_20_10": [0.70, 0.20, 0.10],
    "weighted_80_10_10": [0.80, 0.10, 0.10],
    "weighted_60_25_15": [0.60, 0.25, 0.15],
    "weighted_75_15_10": [0.75, 0.15, 0.10],
}

base_preds = np.argmax(probs[0], axis=1)

for blend_name, weights in blend_configs.items():
    blended_probs = (
        weights[0] * probs[0]
        + weights[1] * probs[1]
        + weights[2] * probs[2]
    )

    preds = np.argmax(blended_probs, axis=1)

    changed = np.sum(preds != base_preds)

    submission = pd.DataFrame({
        "ID": test["ID"],
        "LABEL": preds
    })

    path = f"outputs/{blend_name}.csv"
    submission.to_csv(path, index=False)

    print("\nSaved:", path)
    print("Changed predictions vs C5:", changed)
    print("Prediction distribution:")
    print(submission["LABEL"].value_counts().sort_index())