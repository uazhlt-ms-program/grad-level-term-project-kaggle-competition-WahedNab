# LING 539 Spring 2026 Class Competition

## Project Overview

This repository contains my final code for the LING 539 Spring 2026 class-wide Kaggle competition.

The task was to classify each document into one of three categories:

- `0`: Not a movie or TV show review
- `1`: Positive movie or TV show review
- `2`: Negative movie or TV show review

The Kaggle evaluation metric is **macro F1**, which gives equal weight to performance across all three classes.

## Best Kaggle Submission

My best public leaderboard submission was:

- **Submission name:** `weighted_60_25_15`
- **Public leaderboard score:** `0.93235`
- **Public leaderboard rank at time of submission:** 1st

## Final Approach

My final approach used a **weighted ensemble** of three TF-IDF + Logistic Regression classifiers.

Each model used both:

- word-level TF-IDF features
- character-level TF-IDF features

The final ensemble combined the predicted class probabilities from the three models using the following weights:

```text
60%  C5_word123_char35
25%  C5_5_word123_char35
15%  C4_word123_char46
```

The three models were:

### 1. C5_word123_char35

- Logistic Regression with `C = 5.0`
- word n-grams: `(1, 3)`
- character n-grams: `(3, 5)`

### 2. C5_5_word123_char35

- Logistic Regression with `C = 5.5`
- word n-grams: `(1, 3)`
- character n-grams: `(3, 5)`

### 3. C4_word123_char46

- Logistic Regression with `C = 4.0`
- word n-grams: `(1, 3)`
- character n-grams: `(4, 6)`

The final prediction was selected by averaging the predicted class probabilities from the three models and taking the class with the highest weighted probability.

## Local Evaluation

I evaluated the weighted ensemble using **5-fold stratified cross-validation** with macro F1.

The best ensemble result was:

```text
Model: weighted_60_25_15
Mean macro F1: 0.928345
Standard deviation: 0.001034
```

This was slightly better than the strongest individual model:

```text
Model: C5_word123_char35
Mean macro F1: 0.928193
Standard deviation: 0.001109
```

## Repository Files

### `scripts/weighted_ensemble.py`

Creates the final Kaggle submission files, including:

```text
outputs/weighted_60_25_15.csv
```

This script trains the final three TF-IDF + Logistic Regression models on the full training data, combines their predicted probabilities using weighted averaging, and writes the final Kaggle submission files to the `outputs/` folder.

### `scripts/weighted_ensemble_cv.py`

Runs 5-fold stratified cross-validation to evaluate the individual models and weighted ensembles. It reports the macro F1 score and standard deviation for each model and ensemble.

### `requirements.txt`

Contains the Python package dependencies needed to run the project.

## Data

The Kaggle data files are not included in this repository. To reproduce the result, download the following files from the Kaggle competition page:

```text
train.csv
test.csv
sample_submission.csv
```

Then place them in a folder named:

```text
data/
```

The expected structure is:

```text
grad-level-term-project-kaggle-competition-WahedNab/
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
├── outputs/
├── scripts/
│   ├── weighted_ensemble.py
│   └── weighted_ensemble_cv.py
├── README.md
└── requirements.txt
```

## Installation

Install the required Python packages:

```bash
pip install -r requirements.txt
```

## Reproducing the Kaggle Submission

After placing the Kaggle CSV files in the `data/` folder, run:

```bash
python scripts/weighted_ensemble.py
```

This will train the final models on the full training data and create submission files in the `outputs/` folder.

The final selected submission file is:

```text
outputs/weighted_60_25_15.csv
```

Submit this file to Kaggle.

## Reproducing the Local Cross-Validation Results

To reproduce the local macro F1 and standard deviation results, run:

```bash
python scripts/weighted_ensemble_cv.py
```

This script evaluates the individual models and weighted ensembles using 5-fold stratified cross-validation.

## Final Selected Kaggle Submissions

The two final selected Kaggle submissions were:

```text
weighted_60_25_15
boost_C5_word123_char35
```

The first was selected because it had the highest public leaderboard score and the best local weighted-ensemble cross-validation result. The second was selected as a strong single-model backup because it had a high public score and strong local cross-validation performance.

## Future Improvements

Future improvements could include:

- testing additional classifiers such as Linear SVM or Complement Naive Bayes
- using calibration methods to improve probability estimates before ensembling
- performing more detailed error analysis on ambiguous positive and negative reviews
- adding domain-specific features related to movie and TV review language
- experimenting with transformer-based models if allowed and computationally feasible