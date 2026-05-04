import nbformat as nbf

nb = nbf.v4.new_notebook()

text_intro = """\
# Supervised Machine Learning Pipeline Otomatis
Notebook ini dirancang untuk memproses berbagai jenis data (kategorikal, numerik) secara otomatis, menangani *missing values*, *scaling*, dan mencari model Supervised Learning (Klasifikasi & Regresi) terbaik.

## Fitur:
1. **Pendeteksian Tipe Data**: Memisahkan fitur numerik dan kategorikal.
2. **Preprocessing Otomatis**:
   - Imputasi nilai kosong (Missing Values).
   - *Scaling* untuk data numerik (StandardScaler/RobustScaler).
   - *Encoding* untuk data kategorikal (OneHotEncoding).
3. **Multi-Model Training**:
   - Logistic Regression / Linear Regression
   - Random Forest
   - Gradient Boosting
   - XGBoost
   - LightGBM
   - CatBoost
4. **Evaluasi dan Pemilihan Model Terbaik**: Secara otomatis menampilkan metrik performa model dan memilih yang paling optimal.
"""

code_imports = """\
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Models - Classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Models - Regression
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# Metrics
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
"""

text_load_data = """\
## 1. Load Dataset
Ganti `df = pd.read_csv(...)` dengan path dataset Anda. Tentukan nama kolom target (label) di variabel `target_column`. Tentukan juga jenis tugasnya di `task_type` ('classification' atau 'regression').
"""

code_load_data = """\
# --- KONFIGURASI PENGGUNA ---
# CONTOH MENGGUNAKAN DATASET TITANIC (silakan ganti dataset di bawah ini)
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# Tentukan Target dan Tipe Task (klasifikasi/regresi)
target_column = 'Survived'
task_type = 'classification' # Pilih: 'classification' atau 'regression'

# Tampilkan data awal
display(df.head())
print("Informasi Dataset:")
print(df.info())
"""

text_preprocess = """\
## 2. Preprocessing Otomatis
Kita akan membuat `ColumnTransformer` untuk memproses kolom numerik dan kategorikal dengan cara yang berbeda.
- Numerik: Imputasi dengan Median -> Standarisasi
- Kategorikal: Imputasi dengan Modus -> One-Hot Encoding
"""

code_preprocess = """\
# Pisahkan Fitur (X) dan Target (y)
# Hapus kolom yang tidak relevan jika perlu (misalnya ID, Nama, dsb.)
columns_to_drop = [col for col in ['PassengerId', 'Name', 'Ticket', 'Cabin'] if col in df.columns]
df = df.drop(columns=columns_to_drop)

# Hapus baris di mana targetnya kosong (karena kita tidak bisa belajar dari target yang kosong)
df = df.dropna(subset=[target_column])

X = df.drop(columns=[target_column])
y = df[target_column]

# Deteksi tipe kolom
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

print(f"Fitur Numerik: {numeric_features}")
print(f"Fitur Kategorikal: {categorical_features}")

# Pipeline untuk fitur numerik
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Pipeline untuk fitur kategorikal
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Gabungkan dengan ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\\nJumlah data train: {len(X_train)}")
print(f"Jumlah data test: {len(X_test)}")
"""

text_modeling = """\
## 3. Modeling & Evaluasi
Kita akan melatih beberapa model sekaligus menggunakan Preprocessing Pipeline di atas. Kemudian, kita evaluasi skornya.
"""

code_modeling = """\
# Definisikan Model Berdasarkan Task Type
if task_type == 'classification':
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
        'LightGBM': LGBMClassifier(random_state=42, verbose=-1),
        'CatBoost': CatBoostClassifier(verbose=0, random_state=42)
    }
else:
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest Regressor': RandomForestRegressor(random_state=42),
        'Gradient Boosting Regressor': GradientBoostingRegressor(random_state=42),
        'XGBoost Regressor': XGBRegressor(random_state=42),
        'LightGBM Regressor': LGBMRegressor(random_state=42, verbose=-1),
        'CatBoost Regressor': CatBoostRegressor(verbose=0, random_state=42)
    }

results = []

# Proses training dan evaluasi
for name, model in models.items():
    # Buat Pipeline utuh: Preprocessor + Model
    clf = Pipeline(steps=[('preprocessor', preprocessor),
                          ('model', model)])

    # Train
    clf.fit(X_train, y_train)

    # Predict
    y_pred = clf.predict(X_test)

    # Evaluate
    if task_type == 'classification':
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        results.append({
            'Model': name,
            'Accuracy': acc,
            'F1 Score': f1
        })
    else:
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        results.append({
            'Model': name,
            'RMSE': rmse,
            'R2 Score': r2
        })

# Buat DataFrame hasil
results_df = pd.DataFrame(results)

# Tentukan metric untuk diurutkan
sort_metric = 'Accuracy' if task_type == 'classification' else 'R2 Score'
ascending = False if task_type == 'classification' else False # Untuk klasifikasi tinggi lebih baik, R2 tinggi lebih baik

results_df = results_df.sort_values(by=sort_metric, ascending=ascending).reset_index(drop=True)

display(results_df)

# Visualisasi Perbandingan Model
plt.figure(figsize=(10, 6))
sns.barplot(data=results_df, x='Model', y=sort_metric, palette='viridis')
plt.title(f"Perbandingan Model berdasarkan {sort_metric}")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
"""

text_best_model = """\
## 4. Model Terbaik
Berdasarkan hasil evaluasi, mari kita ambil model terbaik untuk digunakan lebih lanjut.
"""

code_best_model = """\
best_model_name = results_df.iloc[0]['Model']
best_score = results_df.iloc[0][sort_metric]

print(f"🎉 Model Terbaik adalah: **{best_model_name}** dengan {sort_metric} sebesar {best_score:.4f}")

# Simpan pipeline model terbaik
best_model = models[best_model_name]
best_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                ('model', best_model)])

# Fit ke seluruh data training
best_pipeline.fit(X_train, y_train)

print(f"Model {best_model_name} siap digunakan untuk memprediksi data baru!")
"""

nb['cells'] = [
    nbf.v4.new_markdown_cell(text_intro),
    nbf.v4.new_code_cell(code_imports),
    nbf.v4.new_markdown_cell(text_load_data),
    nbf.v4.new_code_cell(code_load_data),
    nbf.v4.new_markdown_cell(text_preprocess),
    nbf.v4.new_code_cell(code_preprocess),
    nbf.v4.new_markdown_cell(text_modeling),
    nbf.v4.new_code_cell(code_modeling),
    nbf.v4.new_markdown_cell(text_best_model),
    nbf.v4.new_code_cell(code_best_model)
]

with open('supervised/Supervised_ML_Pipeline.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("Notebook supervised/Supervised_ML_Pipeline.ipynb successfully created!")
