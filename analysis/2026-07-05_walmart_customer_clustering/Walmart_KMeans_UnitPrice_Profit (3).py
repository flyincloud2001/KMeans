# ── 套件匯入 ──────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import silhouette_score

# ── 參數設定 ──────────────────────────────────────────────────────────────────
FILE_PATH   = r'C:\Users\flyin\OneDrive\桌面\新代碼\KMeans\data\walmart Retail Data.xlsx'
FEATURES    = ['Unit Price', 'Profit']
N_CLUSTERS  = 3
MAX_ITER    = 10  # 疊代過程最多執行次數，若提前收斂會自動停止


# ── 資料讀取與前處理 ──────────────────────────────────────────────────────────
def load_data(path, features):
    # 讀取原始 Excel 檔案
    raw = pd.read_excel(path)
    # 只保留分析所需欄位
    data = raw[features].copy()
    # 刪除缺失值，避免影響分群結果
    data.dropna(inplace=True)
    return data


# ── 標準化 ────────────────────────────────────────────────────────────────────
def scale_features(data, features):
    # 零售數據常有極端值，RobustScaler 以中位數與四分位距縮放，較不受極端值影響
    scaler = RobustScaler()
    scaled = scaler.fit_transform(data[features])
    return scaled


# ── K-Means 建模 ──────────────────────────────────────────────────────────────
def fit_kmeans(scaled_data, n_clusters):
    model = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
    labels = model.fit_predict(scaled_data)
    return model, labels


# ── 視覺化 ────────────────────────────────────────────────────────────────────
def plot_clusters(data, features, labels):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(data[features[0]], data[features[1]],
               c=labels, cmap='viridis', alpha=0.5, s=8)
    ax.set_title('Walmart K-Means 分群結果')
    ax.set_xlabel(features[0])
    ax.set_ylabel(features[1])
    plt.tight_layout()
    plt.savefig('walmart_kmeans_clusters.png', dpi=150)
    plt.show()

data = load_data(FILE_PATH, FEATURES)
scaled = scale_features(data, FEATURES)
model, labels = fit_kmeans(scaled, N_CLUSTERS)
plot_clusters(data, FEATURES, labels)


# ------- 用來算出inertia的function ----------------
def inertia_calculation(data, )