import nbformat
import sys
from pathlib import Path

BASE_DIR = Path(r'c:\Users\PRARTHANA PN\OneDrive\Desktop\Geospatiial_modelling\geospatial-modelling-PFAS\notebooks')

# ================================
# 1. REBUILD MODEL_TRAINING.IPYNB (BASELINE)
# ================================
base_nb_path = BASE_DIR / 'model_training.ipynb'

if base_nb_path.exists():
    with open(base_nb_path, 'r', encoding='utf-8') as f:
        nb_base = nbformat.read(f, as_version=4)
else:
    nb_base = nbformat.v4.new_notebook()

source_base = """# ===============================================
# BASELINE CONTAMINATION MODELING (NO CUSTOM ALGORITHM)
# Target: Single compound value >= 10.0
# ===============================================
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
import joblib

print("Loading PFAS Data...")
BASE_DIR = Path.cwd()
if (BASE_DIR / 'pfas_major_compounds.csv').exists():
    csv_path = BASE_DIR / 'pfas_major_compounds.csv'
else:
    csv_path = BASE_DIR / 'notebooks' / 'pfas_major_compounds.csv'

df = pd.read_csv(csv_path)

# Define baseline target (Value >= 10.0)
target_col = 'value' 
THRESHOLD = 10.0 
df['is_contaminated'] = (df[target_col] >= THRESHOLD).astype(int)

print(f"Class Distribution:\\n{df['is_contaminated'].value_counts(normalize=True) * 100}")

# Base features
lat_col = 'lat'
lon_col = 'lon'
df = df.dropna(subset=[lat_col, lon_col])

# Dummy encode media
df = pd.get_dummies(df, columns=['type'], drop_first=True)
features = [lat_col, lon_col] + [c for c in df.columns if 'type_' in c]

model_df = df.dropna(subset=features + ['is_contaminated']).copy()
X = model_df[features]
y = model_df['is_contaminated']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\\n--- TRAINING 4 BASELINE MODELS ---")

# 1. Extra Trees
print("Training Extra Trees...")
base_et = ExtraTreesClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42, n_jobs=-1)
base_et.fit(X_train, y_train)

# 2. KNN
print("Training KNN...")
base_knn = KNeighborsClassifier(n_neighbors=5, weights='distance', n_jobs=-1)
base_knn.fit(X_train_scaled, y_train)

# 3. Random Forest
print("Training Random Forest...")
base_rf = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42, n_jobs=-1)
base_rf.fit(X_train, y_train)

# 4. XGBoost
print("Training XGBoost...")
pos_weight = (len(y_train) - y_train.sum()) / max(y_train.sum(), 1)
base_xgb = XGBClassifier(scale_pos_weight=pos_weight, n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, eval_metric='logloss')
base_xgb.fit(X_train, y_train)

print("\\n--- BASELINE RESULTS ---")
results = []
results.append(('Extra Trees', roc_auc_score(y_test, base_et.predict_proba(X_test)[:, 1]), accuracy_score(y_test, base_et.predict(X_test))))
results.append(('KNN', roc_auc_score(y_test, base_knn.predict_proba(X_test_scaled)[:, 1]), accuracy_score(y_test, base_knn.predict(X_test_scaled))))
results.append(('Random Forest', roc_auc_score(y_test, base_rf.predict_proba(X_test)[:, 1]), accuracy_score(y_test, base_rf.predict(X_test))))
results.append(('XGBoost', roc_auc_score(y_test, base_xgb.predict_proba(X_test)[:, 1]), accuracy_score(y_test, base_xgb.predict(X_test))))

results.sort(key=lambda x: x[1], reverse=True)
for rank, (name, auc, acc) in enumerate(results):
    print(f"#{rank+1} -> {name:20s} | ROC-AUC: {auc:.4f}  | Accuracy: {acc:.4f}")
"""
nb_base.cells = [nbformat.v4.new_code_cell(source_base)]
with open(base_nb_path, 'w', encoding='utf-8') as f:
    nbformat.write(nb_base, f)

# ================================
# 2. REBUILD MULTI_COMPOUND_OCCURANCE.IPYNB (CUSTOM ALGORITHM)
# ================================
mcri_nb_path = BASE_DIR / 'multi_compound_occurance.ipynb'
if mcri_nb_path.exists():
    with open(mcri_nb_path, 'r', encoding='utf-8') as f:
        nb_mcri = nbformat.read(f, as_version=4)
else:
    nb_mcri = nbformat.v4.new_notebook()

source_mcri = """# ===============================================
# MULTI-COMPOUND RISK INDEX (MCRI) SPATIAL MODELING
# OUR CUSTOM ALGORITHM APPROACH
# ===============================================
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import BallTree
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
import joblib

print("Loading Data & Creating MCRI Target...")
BASE_DIR = Path.cwd()
if (BASE_DIR / 'pfas_major_compounds.csv').exists():
    csv_path = BASE_DIR / 'pfas_major_compounds.csv'
else:
    csv_path = BASE_DIR / 'notebooks' / 'pfas_major_compounds.csv'

df = pd.read_csv(csv_path)

df_pivot = df.pivot_table(index=['lat', 'lon', 'type'], columns='substance', values='value', aggfunc='mean').fillna(0).reset_index()

compound_cols = df_pivot.columns[3:]  
df_pivot['total_pfas'] = df_pivot[compound_cols].sum(axis=1)
df_pivot['compound_count'] = (df_pivot[compound_cols] > 0).sum(axis=1)

weights = {'PFOS': 0.4, 'PFOA': 0.35, 'PFHxS': 0.25}
df_pivot['toxicity_score'] = sum(df_pivot.get(comp, 0) * w for comp, w in weights.items())

if 'PFOS' in df_pivot.columns and 'PFOA' in df_pivot.columns:
    df_pivot['PFOS_PFOA_interaction'] = df_pivot['PFOS'] * df_pivot['PFOA']
else:
    df_pivot['PFOS_PFOA_interaction'] = 0

df_pivot['MCRI'] = (0.3 * df_pivot['total_pfas'] + 0.3 * df_pivot['toxicity_score'] + 0.2 * df_pivot['compound_count'] + 0.2 * df_pivot['PFOS_PFOA_interaction'])

le = LabelEncoder()
df_pivot['media_encoded'] = le.fit_transform(df_pivot['type'])

hotspot_threshold = df_pivot['MCRI'].quantile(0.95)
hotspots_df = df_pivot[df_pivot['MCRI'] >= hotspot_threshold][['lat', 'lon']].copy()
tree = BallTree(np.deg2rad(hotspots_df[['lat', 'lon']].values), metric='haversine')
query_points = np.deg2rad(df_pivot[['lat', 'lon']].values)
dist, _ = tree.query(query_points, k=1)
df_pivot['dist_to_nearest_hotspot_km'] = dist.flatten() * 6371.0

mcri_threshold = df_pivot['MCRI'].quantile(0.75)
df_pivot['high_risk_label'] = (df_pivot['MCRI'] >= mcri_threshold).astype(int)

feature_columns = ['lat', 'lon', 'media_encoded', 'dist_to_nearest_hotspot_km']
X = df_pivot[feature_columns]
y = df_pivot['high_risk_label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\\n--- TRAINING 4 MODELS ON CUSTOM MCRI ALGORITHM ---")

# 1. Extra Trees
print("Training Extra Trees...")
mcri_et = ExtraTreesClassifier(n_estimators=300, max_depth=15, class_weight='balanced', random_state=42, n_jobs=-1)
mcri_et.fit(X_train, y_train)
joblib.dump(mcri_et, "mcri_extratrees_model.pkl")

# 2. KNN
print("Training KNN...")
mcri_knn = KNeighborsClassifier(n_neighbors=5, weights='distance', n_jobs=-1)
mcri_knn.fit(X_train_scaled, y_train)
joblib.dump(mcri_knn, "mcri_knn_model.pkl")

# 3. Random Forest
print("Training Random Forest...")
mcri_rf = RandomForestClassifier(n_estimators=300, max_depth=15, class_weight='balanced', random_state=42, n_jobs=-1)
mcri_rf.fit(X_train, y_train)
joblib.dump(mcri_rf, "mcri_randomforest_model.pkl")

# 4. XGBoost
print("Training XGBoost...")
pos_weight = (len(y_train) - y_train.sum()) / max(y_train.sum(), 1)
mcri_xgb = XGBClassifier(scale_pos_weight=pos_weight, n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42, eval_metric='logloss')
mcri_xgb.fit(X_train, y_train)
joblib.dump(mcri_xgb, "mcri_xgboost_model.pkl")

print("\\n--- ALL 4 MODELS COMPILED AND SAVED INDIVIDUALLY! ---")

print("\\n--- CUSTOM ALGORITHM LEADERBOARD ---")
results = []
results.append(('Extra Trees', roc_auc_score(y_test, mcri_et.predict_proba(X_test)[:, 1]), accuracy_score(y_test, mcri_et.predict(X_test))))
results.append(('KNN', roc_auc_score(y_test, mcri_knn.predict_proba(X_test_scaled)[:, 1]), accuracy_score(y_test, mcri_knn.predict(X_test_scaled))))
results.append(('Random Forest', roc_auc_score(y_test, mcri_rf.predict_proba(X_test)[:, 1]), accuracy_score(y_test, mcri_rf.predict(X_test))))
results.append(('XGBoost', roc_auc_score(y_test, mcri_xgb.predict_proba(X_test)[:, 1]), accuracy_score(y_test, mcri_xgb.predict(X_test))))

results.sort(key=lambda x: x[1], reverse=True)
for rank, (name, auc, acc) in enumerate(results):
    print(f"#{rank+1} -> {name:20s} | ROC-AUC: {auc:.4f}  | Accuracy: {acc:.4f}")

champion_name = results[0][0]
print(f"\\n🏆 GRAND CHAMPION USING OUR ALGORITHM: {champion_name} 🏆")
print(f"The champion model has been identified. (All four have been saved to disk as requested!)")
"""
nb_mcri.cells = [nbformat.v4.new_code_cell(source_mcri)]
with open(mcri_nb_path, 'w', encoding='utf-8') as f:
    nbformat.write(nb_mcri, f)

print("Successfully rebuilt both notebooks to exact user specifications.")
