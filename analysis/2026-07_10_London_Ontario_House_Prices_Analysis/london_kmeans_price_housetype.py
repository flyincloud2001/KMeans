# ── Import Modules ──────────────────────────────────────────────────────────────────
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import silhouette_score

# ── Parameters Setup ──────────────────────────────────────────────────────────────────
FILE_PATH = r'C:\Users\flyin\OneDrive\桌面\新代碼\KMeans\data\london_zolo_house_prices.csv'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 圖表輸出路徑，統一存到本分析資料夾內
MAX_CLUSTERS = 10  # Elbow / Silhouette 分析時，K 值掃描的上限


# ── Read Data and Preprocess Data ──────────────────────────────────────────────────────────
def load_data(path):
    # 讀取原始 CSV 檔案
    raw = pd.read_csv(path)
    # 只保留分析所需欄位：price 與 property_type
    data = raw[['price', 'property_type']].copy()
    # 刪除缺失值，避免影響分群結果
    data.dropna(inplace=True)
    return data


# ── One Hot Encoding ────────────────────────────────────────────────────────────────────
def one_hot_encode_property_type(data):
    # 手動建立 One Hot Encoding，只產生 is_townhouse 與 is_condo 兩欄（不建立 is_house）
    # 這是標準的 dummy variable 編碼方式：n 個類別只需要 n-1 欄即可完整表示，
    # 避免多重共線性（dummy variable trap）。三種房型的對應組合如下：
    #   House     -> (is_townhouse=0, is_condo=0)　← 兩欄皆為 0 時代表基準類別 House
    #   Townhouse -> (is_townhouse=1, is_condo=0)
    #   Condo     -> (is_townhouse=0, is_condo=1)
    data = data.copy()
    data['is_townhouse'] = (data['property_type'] == 'Townhouse').astype(int)
    data['is_condo'] = (data['property_type'] == 'Condo').astype(int)
    return data


# ── Standardize the Data and Build Feature Matrix ────────────────────────────────────────
def build_feature_matrix(data):
    # price 數值範圍大且可能有極端值，用 RobustScaler（以中位數與四分位距縮放）較不受極端值影響
    scaler = RobustScaler()
    price_scaled = scaler.fit_transform(data[['price']])

    # is_townhouse 與 is_condo 已經是 0 / 1，不需要標準化，直接沿用原始數值
    dummy_features = data[['is_townhouse', 'is_condo']].to_numpy()

    # 合併標準化後的 price 與兩個 dummy 欄位，組成最終要丟入 K-Means 的特徵矩陣
    feature_matrix = np.hstack([price_scaled, dummy_features])
    return feature_matrix


# ── K-Means Model ──────────────────────────────────────────────────────────────
def fit_kmeans(feature_matrix, n_clusters):
    model = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
    labels = model.fit_predict(feature_matrix)
    return model, labels


# ── Elbow Method ────────────────────────────────────────────────────────────────────
def plot_elbow_method(feature_matrix):
    # K 從 1 跑到 10，每個 K 都重新訓練一次 K-Means，記錄 inertia_（群內平方誤差總和）
    k_values = []
    inertias = []

    for k in range(1, MAX_CLUSTERS + 1):
        model_temp, _ = fit_kmeans(feature_matrix, k)
        k_values.append(k)
        inertias.append(model_temp.inertia_)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(k_values, inertias, marker='o', label='inertia')
    ax.set_xlabel('K 值')
    ax.set_ylabel('inertia')
    ax.set_title('K 值對 inertia 折線圖（Elbow Method）')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, 'k_values_versus_inertias.png'), dpi=150)
    plt.show()

    return inertias


# ── Silhouette Score ────────────────────────────────────────────────────────────────────
def plot_silhouette_score(feature_matrix):
    # K 從 2 跑到 10（Silhouette Score 至少需要 2 群才能計算），記錄每個 K 的 silhouette_score
    k_values = []
    silhouette_scores = []

    for k in range(2, MAX_CLUSTERS + 1):
        model_temp, labels = fit_kmeans(feature_matrix, k)
        score = silhouette_score(feature_matrix, labels)
        k_values.append(k)
        silhouette_scores.append(score)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(k_values, silhouette_scores, marker='o', color='darkorange', label='silhouette score')
    ax.set_xlabel('K 值')
    ax.set_ylabel('silhouette score')
    ax.set_title('K 值對 silhouette score 折線圖')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, 'k_values_versus_silhouette_scores.png'), dpi=150)
    plt.show()

    return silhouette_scores


# ── Interpretation（尚未呼叫，待確認 K 值後再手動執行）───────────────────────────────────
def interpret_clusters(feature_matrix, data, k):
    # 輸入最終決定的 K 值後，訓練 K-Means 模型
    model, labels = fit_kmeans(feature_matrix, k)

    clustered_data = data.copy()
    clustered_data['cluster'] = labels

    # 計算每一群在原始 price（未標準化）上的平均值
    price_mean_by_cluster = clustered_data.groupby('cluster')['price'].mean()
    # 計算每一群在 is_townhouse 與 is_condo 上的平均值，可反映該群中 townhouse 與 condo 各自的比例
    house_type_ratio_by_cluster = clustered_data.groupby('cluster')[['is_townhouse', 'is_condo']].mean()

    print(f'===== K={k} 分群結果：各群原始 price 平均值 =====')
    print(price_mean_by_cluster)
    print(f'===== K={k} 分群結果：各群 is_townhouse / is_condo 平均值（房型比例） =====')
    print(house_type_ratio_by_cluster)

    return clustered_data


# ── Visualization（尚未呼叫，待確認 K 值後再手動執行）─────────────────────────────────────
def plot_average_price_by_cluster(clustered_data):
    # 畫出每一群的平均價格長條圖，並在長條上標示數值
    price_mean_by_cluster = clustered_data.groupby('cluster')['price'].mean()

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(price_mean_by_cluster.index, price_mean_by_cluster, color='steelblue')
    ax.bar_label(bars, fmt='%.0f')
    ax.set_title('各群平均 price')
    ax.set_xlabel('cluster')
    ax.set_ylabel('average price')
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, 'average_price_by_cluster.png'), dpi=150)
    plt.show()


# ── Main：目前只執行到 Elbow 與 Silhouette，K 值確認後再另外呼叫 interpret_clusters 與 plot_average_price_by_cluster ──
data = load_data(FILE_PATH)
data = one_hot_encode_property_type(data)
feature_matrix = build_feature_matrix(data)

inertias = plot_elbow_method(feature_matrix)
print('===== 各 K 值對應的 inertia =====')
print(inertias)

silhouette_scores = plot_silhouette_score(feature_matrix)
print('===== 各 K 值對應的 silhouette score =====')
print(silhouette_scores)
