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
N_SELECT    = 3   # 從所有疊代中挑選展示的張數


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


# ── K-Means 疊代過程追蹤 ──────────────────────────────────────────────────────
def run_kmeans_iterations(scaled_data, n_clusters, max_iter, random_state=0):
    # 先用 KMeans++ 找出初始中心點，讓起點與正式模型一致
    init_model = KMeans(n_clusters=n_clusters, n_init=1, max_iter=1,
                         random_state=random_state)
    init_model.fit(scaled_data)
    centroids = init_model.cluster_centers_

    history = []
    for iteration in range(1, max_iter + 1):
        # 將每個點指派給距離最近的中心點
        distances = np.linalg.norm(
            scaled_data[:, None, :] - centroids[None, :, :], axis=2)
        labels = distances.argmin(axis=1)

        # 計算該次疊代的分群品質
        score = silhouette_score(scaled_data, labels)
        history.append({'iteration': iteration, 'labels': labels,
                         'centroids': centroids, 'score': score})

        # 依照目前分群結果重新計算中心點
        new_centroids = np.array([
            scaled_data[labels == k].mean(axis=0) for k in range(n_clusters)
        ])

        # 中心點不再變化即代表收斂，提前結束疊代
        if np.allclose(new_centroids, centroids):
            break
        centroids = new_centroids

    return history


# ── 挑選代表性疊代 ────────────────────────────────────────────────────────────
def select_representative_iterations(history, n_select):
    # 依 Silhouette Score 由低到高排序，平均取樣出最具代表性的幾次疊代
    ranked = sorted(history, key=lambda record: record['score'])
    indices = np.linspace(0, len(ranked) - 1, n_select).astype(int)
    return [ranked[i] for i in indices]


# ── 疊代過程視覺化 ────────────────────────────────────────────────────────────
def plot_iteration_progress(scaled_data, records):
    fig, axes = plt.subplots(1, len(records), figsize=(6 * len(records), 5))
    for ax, record in zip(axes, records):
        ax.scatter(scaled_data[:, 0], scaled_data[:, 1],
                   c=record['labels'], cmap='viridis', alpha=0.5, s=8)
        ax.scatter(record['centroids'][:, 0], record['centroids'][:, 1],
                   c='black', s=100)
        ax.set_title(f"Iteration: {record['iteration']}")
        ax.set_xlabel(f'{FEATURES[0]}（標準化後）')
        ax.set_ylabel(f'{FEATURES[1]}（標準化後）')
    plt.tight_layout()
    plt.savefig('walmart_kmeans_iteration_progress.png', dpi=150)
    plt.show()


# ── 主流程 ────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    data = load_data(FILE_PATH, FEATURES)
    scaled_data = scale_features(data, FEATURES)
    model, labels = fit_kmeans(scaled_data, N_CLUSTERS)
    data['cluster'] = labels

    # Silhouette Score 為分群品質的核心診斷指標，用來判斷群與群之間是否有明顯區隔
    score = silhouette_score(scaled_data, labels)
    print(f'Silhouette Score：{score:.4f}')

    plot_clusters(data, FEATURES, labels)

    # ── 疊代過程分析 ──────────────────────────────────────────────────────
    history = run_kmeans_iterations(scaled_data, N_CLUSTERS, MAX_ITER)

    for record in history:
        print(f"Iteration {record['iteration']} Silhouette Score："
              f"{record['score']:.4f}")

    selected_records = select_representative_iterations(history, N_SELECT)
    plot_iteration_progress(scaled_data, selected_records)
