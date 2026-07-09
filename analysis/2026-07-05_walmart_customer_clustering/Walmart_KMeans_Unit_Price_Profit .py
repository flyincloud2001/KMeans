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
MAX_CLUSTERS  = 10  # 疊代過程最多執行次數，若提前收斂會自動停止


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
def plot_clusters(data, features, labels, n):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(data[features[0]], data[features[1]],
            c=labels, cmap='viridis', alpha=0.5, s=8)
    ax.set_title(f'Walmart K-Means 分群結果 _for_k={n}')
    ax.set_xlabel(features[0])
    ax.set_ylabel(features[1])
    plt.tight_layout()
    plt.savefig(f'walmart_kmeans_clusters_for_k={n}.png', dpi=150)
    plt.show()

def inertias_line_plot(data):

    inertias = []
    clusters = []

    for n in range(1, MAX_CLUSTERS+1):
        model_temp, labels= fit_kmeans(scaled, n)
        inertia = model_temp.inertia_
        clusters.append(n)
        inertias.append(inertia)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(clusters, inertias, label='inertias')
    ax.set_xlabel("k values")
    ax.set_ylabel("inertia")
    ax.set_title("k values versus inertias")
    ax.axvline(3, color='r', alpha=0.5, linestyle='--', label='Elbow Point')
    plt.tight_layout()
    plt.savefig("k_values_versus_inertias.png", dpi=150)
    plt.show()
    
    return inertias

def silhouette_score_line_plot(scaled_data):

    silhouette_scores = []
    clusters = []
    
    for n in range(2, MAX_CLUSTERS+1):
        model_temp, labels= fit_kmeans(scaled_data, n)
        score = silhouette_score(scaled_data, labels)
        clusters.append(n)
        silhouette_scores.append(score)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(clusters, silhouette_scores, label='silhouette score')
    ax.set_xlabel("k values")
    ax.set_ylabel("silhouette score")
    ax.set_title("k values versus silhouette scores")
    ax.axvline(3, color='r', alpha=0.5, linestyle='--', label='Elbow Point')
    plt.tight_layout()
    plt.savefig("k_values_versus_silhouette_scores.png", dpi=150)
    plt.show()
    
    return silhouette_scores

def interpretation(labels, data):
    data['cluster'] = labels
    clusters = data.groupby('cluster')[FEATURES].mean()
    print(clusters)

def bar_plot(data):

    fig, ax = plt.subplots(figsize=(8, 5))
    data.groupby('cluster')[FEATURES].mean()['Profit'].plot(kind='bar', ax=ax)
    ax.set_title('各群平均獲利')
    ax.set_xlabel('cluster')
    ax.set_ylabel('Profit')
    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(figsize=(8, 5))
    data.groupby('cluster').size()['Profit'].plot(kind='bar', ax=ax)
    ax.set_title('各群平均獲利')
    ax.set_xlabel('cluster')
    ax.set_ylabel('Profit')
    plt.tight_layout()
    plt.show()

data = load_data(FILE_PATH, FEATURES)
scaled = scale_features(data, FEATURES)
inertias = inertias_line_plot(scaled)
silhouette_scores = silhouette_score_line_plot(scaled)
model, labels = fit_kmeans(scaled, 3)
plot_clusters(data, FEATURES, labels, 3)
interpretation(labels, data)
bar_plot(data)




