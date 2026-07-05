# ══════════════════════════════════════════════════════════════════════════
# Walmart Retail Data 顧客分群分析
# 使用 K Means 聚類演算法，依照顧客的消費行為特徵進行分群
# ══════════════════════════════════════════════════════════════════════════
#
# 資料來源說明：
# 本程式使用 Kaggle 上的 Walmart Retail Data 數據集
# 下載網址：https://www.kaggle.com/datasets/saadabdurrazzaq/walmart-retail-data
# 請先手動下載該數據集的 xlsx 檔案，並將檔名與路徑對應到下方 FILE_PATH 變數
#
# 資料結構說明：
# 這份數據是「訂單交易紀錄」，不是「每位顧客一筆」的彙總資料
# 因此需要先依照顧客姓名做群組彙總，計算出每位顧客的消費特徵
# 才能拿去做 K Means 分群
#
# ══════════════════════════════════════════════════════════════════════════

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# 設定中文字型，避免圖表中文字顯示成方框
# 這裡明確載入系統內的 Noto Sans CJK 字型檔，並註冊給 matplotlib 使用
# 若在自己的電腦上執行且沒有這個字型，圖表中文字仍可能顯示成方框
# 屆時可以換成電腦上已安裝的中文字型名稱，例如 Windows 上的 'Microsoft JhengHei'
import matplotlib.font_manager as fm
import os

_cjk_font_path = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
if os.path.exists(_cjk_font_path):
    fm.fontManager.addfont(_cjk_font_path)
    plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP']  # 此字型檔內含繁體中文字形
else:
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei']

plt.rcParams['axes.unicode_minus'] = False

# ── 參數設定 ──────────────────────────────────────────────────────────────
FILE_PATH = 'walmart Retail Data.xlsx'  # 請改成你實際存放的檔案路徑
RANDOM_STATE = 42  # 固定亂數種子，確保每次執行結果一致

# ══════════════════════════════════════════════════════════════════════════
# 第一部分：讀取資料
# ══════════════════════════════════════════════════════════════════════════

raw = pd.read_excel(FILE_PATH)

print('=== 原始資料概況 ===')
print(f'總筆數：{len(raw)}')
print(f'欄位數：{raw.shape[1]}')
print(f'不重複顧客數：{raw["Customer Name"].nunique()}')
print()

# ══════════════════════════════════════════════════════════════════════════
# 第二部分：特徵工程
# 概念：把交易層級的資料，彙總成顧客層級的資料
# 這個做法稱為 RFM 分析的變體，分別對應消費金額、消費頻率、消費習慣
# ══════════════════════════════════════════════════════════════════════════

# 依照顧客姓名分組，計算每位顧客的彙總特徵
customer_data = raw.groupby('Customer Name').agg(
    total_sales=('Sales', 'sum'),           # 總消費金額（Monetary）
    order_count=('Order ID', 'nunique'),    # 不重複訂單數（Frequency）
    avg_discount=('Discount', 'mean'),      # 平均折扣率
    total_profit=('Profit', 'sum'),         # 帶給公司的總利潤
    avg_order_qty=('Order Quantity', 'mean')  # 平均每筆訂單購買數量
).reset_index()

print('=== 彙總後的顧客層級資料（前5筆）===')
print(customer_data.head())
print()

print('=== 顧客層級資料統計摘要 ===')
print(customer_data[['total_sales', 'order_count', 'avg_discount',
                      'total_profit', 'avg_order_qty']].describe().round(2))
print()

# 選定用於分群的特徵
# 選擇總消費金額、訂單數、平均折扣率這三個最能代表消費行為的欄位
cluster_cols = ['total_sales', 'order_count', 'avg_discount']

# ══════════════════════════════════════════════════════════════════════════
# 第三部分：標準化
# 概念：total_sales 數值範圍遠大於 avg_discount，若不標準化
# K Means 會被數值尺度大的欄位主導，導致分群結果失真
# ══════════════════════════════════════════════════════════════════════════

scaler = StandardScaler()
X_scaled = scaler.fit_transform(customer_data[cluster_cols])

# ══════════════════════════════════════════════════════════════════════════
# 第四部分：用 Elbow Method 與 Silhouette Score 決定最適合的分群數
# 概念：Elbow Method 看群內平方和下降的轉折點
# Silhouette Score 則量化每個點與自己群和其他群的相對距離，越接近1越好
# ══════════════════════════════════════════════════════════════════════════

inertia_list = []
silhouette_list = []
k_range = range(2, 9)  # 測試分成2到8群

for k in k_range:
    model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    labels = model.fit_predict(X_scaled)
    inertia_list.append(model.inertia_)
    silhouette_list.append(silhouette_score(X_scaled, labels))

print('=== 不同群數的 Silhouette Score ===')
for k, score in zip(k_range, silhouette_list):
    print(f'  k={k}：{score:.4f}')

# 選擇 Silhouette Score 最高的群數作為最終分群依據
best_k = list(k_range)[int(np.argmax(silhouette_list))]
print(f'\n選定最適合的群數：k={best_k}')
print()

# 繪製 Elbow 圖與 Silhouette 圖
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(list(k_range), inertia_list, marker='o')
axes[0].set_title('Elbow Method：群內平方和 vs 群數')
axes[0].set_xlabel('群數 k')
axes[0].set_ylabel('群內平方和（Inertia）')

axes[1].plot(list(k_range), silhouette_list, marker='o', color='orange')
axes[1].axvline(best_k, color='red', linestyle='--', label=f'最佳群數 k={best_k}')
axes[1].set_title('Silhouette Score vs 群數')
axes[1].set_xlabel('群數 k')
axes[1].set_ylabel('Silhouette Score')
axes[1].legend()

plt.tight_layout()
plt.savefig('walmart_kmeans_selection.png', dpi=150)
plt.show()

# ══════════════════════════════════════════════════════════════════════════
# 第五部分：用選定的群數執行最終分群
# ══════════════════════════════════════════════════════════════════════════

final_model = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10)
customer_data['cluster'] = final_model.fit_predict(X_scaled)

# ══════════════════════════════════════════════════════════════════════════
# 第六部分：分群結果解讀
# 概念：分群完成後，需要看每一群在原始特徵上的平均值
# 才能判斷這一群顧客實際代表什麼樣的消費行為
# ══════════════════════════════════════════════════════════════════════════

cluster_summary = customer_data.groupby('cluster')[cluster_cols + ['total_profit']].mean().round(2)
cluster_summary['顧客人數'] = customer_data['cluster'].value_counts().sort_index()

print('=== 各群顧客的平均消費特徵 ===')
print(cluster_summary)
print()

# 自動判斷每一群的商業意義標籤
# 依照總消費金額的相對高低，標記為高價值、中價值、低價值顧客
sales_rank = cluster_summary['total_sales'].rank(ascending=False)
labels_map = {}
for cluster_id, rank in sales_rank.items():
    if rank == 1:
        labels_map[cluster_id] = '高價值顧客'
    elif rank == len(sales_rank):
        labels_map[cluster_id] = '低價值顧客'
    else:
        labels_map[cluster_id] = '中價值顧客'

print('=== 各群的商業意義標籤 ===')
for cluster_id, label in labels_map.items():
    print(f'  群 {cluster_id}：{label}')
print()

# ══════════════════════════════════════════════════════════════════════════
# 第七部分：視覺化分群結果
# ══════════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(9, 7))
scatter = ax.scatter(
    customer_data['total_sales'],
    customer_data['order_count'],
    c=customer_data['cluster'],
    cmap='viridis',
    alpha=0.7,
    s=60
)
ax.set_xlabel('總消費金額（Total Sales）')
ax.set_ylabel('訂單數量（Order Count）')
ax.set_title(f'Walmart 顧客分群結果（K Means，k={best_k}）')
legend = ax.legend(*scatter.legend_elements(), title='群別')
ax.add_artist(legend)
plt.tight_layout()
plt.savefig('walmart_kmeans_clusters.png', dpi=150)
plt.show()

# 各群平均消費金額長條圖
fig, ax = plt.subplots(figsize=(8, 5))
cluster_summary['total_sales'].plot(kind='bar', ax=ax, color='steelblue')
ax.set_title('各群平均總消費金額比較')
ax.set_xlabel('群別')
ax.set_ylabel('平均總消費金額')
plt.tight_layout()
plt.savefig('walmart_cluster_sales_comparison.png', dpi=150)
plt.show()

print('=== 分析完成 ===')
print('圖表已儲存：walmart_kmeans_selection.png')
print('圖表已儲存：walmart_kmeans_clusters.png')
print('圖表已儲存：walmart_cluster_sales_comparison.png')
