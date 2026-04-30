import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score


train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

X = train["TEXT"].fillna("").reset_index(drop=True)
y = train["LABEL"].reset_index(drop=True)
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


model_configs = [
    ("C5_word123_char35", 5.0, (1, 3), (3, 5), 120000, 140000, 2),
    ("C5_5_word123_char35", 5.5, (1, 3), (3, 5), 120000, 140000, 2),
    ("C4_word123_char46", 4.0, (1, 3), (4, 6), 120000, 140000, 2),
]

blend_configs = {
    "weighted_80_10_10": [0.80, 0.10, 0.10],
    "weighted_70_20_10": [0.70, 0.20, 0.10],
    "weighted_75_15_10": [0.75, 0.15, 0.10],
    "weighted_60_25_15": [0.60, 0.25, 0.15],
}

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

blend_scores = {name: [] for name in blend_configs}
single_scores = {name: [] for name, *_ in model_configs}

for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), start=1):
    print("\n" + "=" * 70)
    print(f"Fold {fold}")

    X_train = X.iloc[train_idx]
    y_train = y.iloc[train_idx]
    X_val = X.iloc[val_idx]
    y_val = y.iloc[val_idx]

    val_probs = []

    for name, C, word_ngram, char_ngram, max_word, max_char, min_df in model_configs:
        print("Training:", name)

        model = make_model(
            C=C,
            word_ngram=word_ngram,
            char_ngram=char_ngram,
            max_word=max_word,
            max_char=max_char,
            min_df=min_df
        )

        model.fit(X_train, y_train)

        probs = model.predict_proba(X_val)
        preds = np.argmax(probs, axis=1)

        score = f1_score(y_val, preds, average="macro")
        single_scores[name].append(score)

        print(name, "fold macro F1:", score)

        val_probs.append(probs)

    for blend_name, weights in blend_configs.items():
        blended_probs = (
            weights[0] * val_probs[0]
            + weights[1] * val_probs[1]
            + weights[2] * val_probs[2]
        )

        blend_preds = np.argmax(blended_probs, axis=1)
        blend_score = f1_score(y_val, blend_preds, average="macro")

        blend_scores[blend_name].append(blend_score)

        print(blend_name, "fold macro F1:", blend_score)


print("\n" + "=" * 70)
print("Single model CV results:")

for name, scores in single_scores.items():
    scores = np.array(scores)
    print(name)
    print("Scores:", scores)
    print("Mean macro F1:", scores.mean())
    print("Std:", scores.std())
    print()

print("\n" + "=" * 70)
print("Weighted ensemble CV results:")

final_results = []

for name, scores in blend_scores.items():
    scores = np.array(scores)
    mean_score = scores.mean()
    std_score = scores.std()

    final_results.append((name, mean_score, std_score))

    print(name)
    print("Scores:", scores)
    print("Mean macro F1:", mean_score)
    print("Std:", std_score)
    print()

final_results = sorted(final_results, key=lambda x: x[1], reverse=True)

print("\nFinal weighted ensemble ranking:")
for name, mean_score, std_score in final_results:
    print(name, mean_score, std_score)