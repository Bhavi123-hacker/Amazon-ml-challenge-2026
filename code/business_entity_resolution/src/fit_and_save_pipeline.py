"""Train and bundle the winning NeuralNet_MLP pipeline with StandardScaler.
"""

import json
import os
import pickle
import sys
import time

sys.path.insert(0, os.path.abspath("code/business_entity_resolution"))

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

from src.features import FEATURE_COLUMNS
from src.train_and_benchmark import build_training_dataset


def main():
    print("=== Training Final NeuralNet_MLP Pipeline ===", flush=True)
    df_pairs, gt_map = build_training_dataset("student_resource/dataset", n_s1_samples=5000, top_k_candidates=40)

    X = df_pairs[FEATURE_COLUMNS].values
    y = df_pairs["label"].values

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            alpha=0.01,
            learning_rate_init=0.003,
            max_iter=60,
            early_stopping=True,
            random_state=42,
            verbose=True,
        ))
    ])

    print(f"Fitting pipeline on {len(df_pairs):,} pairs across {len(FEATURE_COLUMNS)} features...", flush=True)
    t0 = time.time()
    pipeline.fit(X, y)
    print(f"Pipeline fitted in {time.time() - t0:.2f}s", flush=True)

    # Save to models/best_model_pipeline.pkl
    out_path = "models/best_model_pipeline.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"Saved pipeline to {out_path}", flush=True)


if __name__ == "__main__":
    main()
