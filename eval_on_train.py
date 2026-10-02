import os, sys, pickle, time
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath("code/business_entity_resolution"))
from src.features import FEATURE_COLUMNS, extract_pair_features
from src.normalize import apply_normalization_df
from src.evaluate import compute_macro_f05
from src.blocking import BlockingIndex

# 1. Load ground truth for 1000 entities
print("Loading ground truth...")
df_gt = pd.read_csv('student_resource/dataset/train/train_ground_truth.tsv', sep='\t', nrows=1000, keep_default_na=False)
gt_map = {}
for _, r in df_gt.iterrows():
    s1 = r['source1_entity_id'].strip()
    m = r['matched_entity_ids'].strip()
    gt_map[s1] = [x.strip() for x in m.split(',') if x.strip()] if m else []

target_s1 = set(gt_map.keys())

# 2. Load S1 records
print("Loading S1 records...")
s1_rows = []
for chunk in pd.read_csv('student_resource/dataset/train/train_source1.tsv', sep='\t', chunksize=50000, keep_default_na=False):
    sub = chunk[chunk['entity_id'].isin(target_s1)]
    if len(sub): s1_rows.append(sub)
    if sum(len(x) for x in s1_rows) >= len(target_s1): break
df_s1 = pd.concat(s1_rows, ignore_index=True)
df_s1 = apply_normalization_df(df_s1)

# 3. Load model
with open("models/best_model_pipeline.pkl", "rb") as f:
    model = pickle.load(f)

# Also test: What if we evaluate our blocking + model on these entities?
print("Evaluated setup ready.")
