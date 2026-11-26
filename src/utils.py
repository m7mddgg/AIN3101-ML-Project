"""
Shared functions for the AIN3101 project.
Both notebooks (Dry Bean and Raisin) use these functions, so the same
pipelines, the same tuning setup and the same metrics are applied to both datasets.
"""
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, ConfusionMatrixDisplay)
from sklearn.inspection import permutation_importance

RANDOM_STATE = 42
FIG_DIR = "../results/figures/"
TABLE_DIR = "../results/tables/"


# ---------- Stage 2: quality check ----------
def quality_check(df):
    """Print shape, missing values, duplicates and data types."""
    print("Shape:", df.shape)
    print("Missing values:", df.isnull().sum().sum())
    print("Duplicate rows:", df.duplicated().sum())
    print("\nData types:")
    print(df.dtypes.value_counts())


# ---------- Stage 5 and 7: pipelines and grids ----------
def get_models():
    """Return the three models with their parameter grids.
    LR and SVM get a StandardScaler inside the Pipeline (no leakage).
    Random Forest does not need scaling (trees split on thresholds)."""
    models = {
        "Logistic Regression": (
            Pipeline([("scaler", StandardScaler()),
                      ("model", LogisticRegression(max_iter=5000))]),
            {"model__C": [0.01, 0.1, 1, 10, 100]},
        ),
        "Random Forest": (
            Pipeline([("model", RandomForestClassifier(random_state=RANDOM_STATE))]),
            {"model__n_estimators": [100, 200, 300],
             "model__max_depth": [None, 10, 20],
             "model__min_samples_split": [2, 5, 10]},
        ),
        "SVM": (
            Pipeline([("scaler", StandardScaler()),
                      ("model", SVC(random_state=RANDOM_STATE))]),
            # gamma is only used by the rbf kernel, so linear gets C only
            [{"model__kernel": ["linear"], "model__C": [0.1, 1, 10, 100]},
             {"model__kernel": ["rbf"], "model__C": [0.1, 1, 10, 100],
              "model__gamma": ["scale", 0.01, 0.1]}],
        ),
    }
    return models


def tune_model(pipe, grid, X_train, y_train, cv):
    """GridSearchCV on the training set only, scored with macro F1."""
    search = GridSearchCV(pipe, grid, cv=cv, scoring="f1_macro", n_jobs=-1)
    start = time.time()
    search.fit(X_train, y_train)
    tuning_time = time.time() - start
    return search, tuning_time


# ---------- Stage 8: evaluation ----------
def get_scores(y_true, y_pred):
    """Accuracy, macro Precision/Recall/F1 and weighted F1."""
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision (macro)": precision_score(y_true, y_pred, average="macro"),
        "Recall (macro)": recall_score(y_true, y_pred, average="macro"),
        "F1 (macro)": f1_score(y_true, y_pred, average="macro"),
        "F1 (weighted)": f1_score(y_true, y_pred, average="weighted"),
    }


def clean_params(params):
    """Turn {'model__C': 1} into 'C=1' for the results table."""
    return ", ".join(f"{k.replace('model__', '')}={v}" for k, v in params.items())


def plot_confusion(model, X_test, y_test, labels, title, filename):
    """Confusion matrix of one model on the test set, saved as a figure."""
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, display_labels=labels,
                                          cmap="Blues", ax=ax, colorbar=False)
    ax.set_title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(FIG_DIR + filename, dpi=150)
    plt.show()


# ---------- Stage 9: feature importance ----------
def get_permutation_importance(model, X_test, y_test):
    """Permutation importance on the test set, scored with macro F1."""
    result = permutation_importance(model, X_test, y_test, scoring="f1_macro",
                                    n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1)
    return pd.Series(result.importances_mean, index=X_test.columns).sort_values()


# ---------- EDA helpers ----------
def plot_histograms(X, title, filename):
    n = len(X.columns)
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(16, 3 * rows))
    axes = axes.flatten()
    for i, col in enumerate(X.columns):
        axes[i].hist(X[col], bins=40, color="steelblue")
        axes[i].set_title(col)
    for j in range(n, len(axes)):
        axes[j].axis("off")
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(FIG_DIR + filename, dpi=150)
    plt.show()


def plot_boxplots(X, y, title, filename):
    """One boxplot per feature, split by class."""
    n = len(X.columns)
    cols = 4
    rows = int(np.ceil(n / cols))
    classes = sorted(y.unique())
    fig, axes = plt.subplots(rows, cols, figsize=(16, 3.5 * rows))
    axes = axes.flatten()
    for i, col in enumerate(X.columns):
        data = [X.loc[y == c, col] for c in classes]
        axes[i].boxplot(data, flierprops={"markersize": 2})
        axes[i].set_xticks(range(1, len(classes) + 1))
        axes[i].set_xticklabels(classes, rotation=45, fontsize=8)
        axes[i].set_title(col)
    for j in range(n, len(axes)):
        axes[j].axis("off")
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(FIG_DIR + filename, dpi=150)
    plt.show()


def plot_correlation(X, title, filename):
    corr = X.corr()
    size = max(6, len(X.columns) * 0.6)
    fig, ax = plt.subplots(figsize=(size + 1, size))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)))
    ax.set_yticks(range(len(corr)))
    ax.set_xticklabels(corr.columns, rotation=90)
    ax.set_yticklabels(corr.columns)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
    plt.colorbar(im)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(FIG_DIR + filename, dpi=150)
    plt.show()
    return corr


def count_outliers(X):
    """Number of points outside 1.5*IQR for each feature."""
    q1 = X.quantile(0.25)
    q3 = X.quantile(0.75)
    iqr = q3 - q1
    outside = (X < q1 - 1.5 * iqr) | (X > q3 + 1.5 * iqr)
    return outside.sum().sort_values(ascending=False)
