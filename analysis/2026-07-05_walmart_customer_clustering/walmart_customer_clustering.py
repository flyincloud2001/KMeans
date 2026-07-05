# ══════════════════════════════════════════════════════════════════════════
# Walmart Retail Data 顧客分群分析（修改版：改用 RFM 特徵）
# 使用 K Means 聚類演算法，依照顧客的消費行為特徵進行分群
# ══════════════════════════════════════════════════════════════════════════
#
# 資料來源說明：
# 本程式使用 Kaggle 上的 Walmart Retail Data 數據集
# 下載網址：https://www.kaggle.com/datasets/saadabdurrazzaq/walmart-retail-data
# 請先手動下載該數據集的 xlsx 檔案，並將檔名與路徑對應到下方 FILE_PATH 變數
#
# 修改說明（相較於第一版）：
# 第一版用總消費金額、訂單數、平均折扣率當特徵，但總消費金額和訂單數
# 相關係數高達 0.65，本質上是同一個資訊講兩次，導致 K Means 只在單一維度
# 上把顧客切成大小兩群，沒有展現多維度分群的效果
# 這一版改用真正的 RFM 分析：
#   R（Recency）：最近一次購買距離資料最後日期的天數
#   F（Frequency）：訂單數量
#   M（Monetary）：改用「平均每筆訂單金額」，而非「總消費金額」
#     這樣可以把「單筆消費金額大小」和「購買頻率」分開來看
#     避免兩個特徵本質上重複
# 這三個特徵彼此的相關係數都在 0.42 以下，才是真正各自獨立的資訊維度
#
# ══════════════════════════════════════════════════════════════════════════

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ── 中文字型設定 ──────────────────────────────────────────────────────────
# 避免圖表中文字顯示成方框
# 若在自己的電腦上執行且沒有 Noto Sans CJK 字型，會改用系統內建中文字型
# 例如 Windows 上的 Microsoft JhengHei，需自行確認電腦上實際安裝的字型名稱
_cjk_font_path = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
if os.path.exists(_cjk_font_path):
    fm.fontManager.addfont(_cjk_font_path)
    plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP']  # 此字型檔內含繁體中文字形
else:
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ── 參數設定 ──────────────────────────────────────────────────────────────
FILE_PATH = r'C:\Users\flyin\OneDrive\桌面\新代碼\KMeans\data\walmart Retail Data.xlsx'  # 請改成你實際存放的檔案路徑
RANDOM_STATE = 42  # 固定亂數種子，確保每次執行結果一致

# ══════════════════════════════════════════════════════════════════════════
# 第一部分：讀取資料
# ══════════════════════════════════════════════════════════════════════════

raw = pd.read_excel(FILE_PATH)
raw['Order Date'] = pd.to_datetime(raw['Order Date'])  # 轉換成日期格式才能計算天數差

print('=== 原始資料概況 ===')
print(f'總筆數：{len(raw)}')
print(f'欄位數：{raw.shape[1]}')
print(f'不重複顧客數：{raw["Customer Name"].nunique()}')
print()

# ══════════════════════════════════════════════════════════════════════════
# 第二部分：特徵工程（RFM 分析）
# 概念：把交易層級的資料，彙總成顧客層級的資料
# ══════════════════════════════════════════════════════════════════════════

# 資料集裡最後一筆訂單的日期，作為計算 Recency 的基準點
max_date = raw['Order Date'].max()
print(f'資料集最後日期：{max_date.date()}（作為計算 Recency 的基準點）')
print()

# 依照顧客姓名分組，計算每位顧客的彙總特徵
customer_data = raw.groupby('Customer Name').agg(
    total_sales=('Sales', 'sum'),           # 總消費金額，僅供輔助參考，不放入分群特徵
    order_count=('Order ID', 'nunique'),    # 不重複訂單數 → Frequency
    last_order=('Order Date', 'max'),       # 最後一次購買日期，用來計算 Recency
    total_profit=('Profit', 'sum')          # 帶給公司的總利潤，僅供解讀分群結果用
).reset_index()

# 計算平均每筆訂單金額 → Monetary（改良版，避免和訂單數重複）
customer_data['avg_order_value'] = customer_data['total_sales'] / customer_data['order_count']

# 計算 Recency：最後一次購買距離資料集最後日期的天數
customer_data['recency_days'] = (max_date - customer_data['last_order']).dt.days

print('=== 彙總後的顧客層級資料（前5筆）===')
print(customer_data[['Customer Name', 'avg_order_value', 'order_count', 'recency_days']].head())
print()

# 選定用於分群的特徵：真正的 RFM 三個維度
cluster_cols = ['avg_order_value', 'order_count', 'recency_days']

print('=== 顧客層級資料統計摘要 ===')
print(customer_data[cluster_cols].describe().round(2))
print()

# 確認三個特徵彼此的相關係數，驗證沒有重複資訊
print('=== 特徵間相關係數（用來確認沒有重複資訊）===')
print(customer_data[cluster_cols].corr().round(3))
print()

# ══════════════════════════════════════════════════════════════════════════
# 第三部分：標準化
# 概念：avg_order_value 數值範圍遠大於 order_count 和 recency_days
# 若不標準化，K Means 會被數值尺度大的欄位主導，導致分群結果失真
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

cluster_summary = customer_data.groupby('cluster')[cluster_cols + ['total_profit']].mean().round(1)
cluster_summary['顧客人數'] = customer_data['cluster'].value_counts().sort_index()

print('=== 各群顧客的平均消費特徵 ===')
print(cluster_summary)
print()

# ── 自動判斷每一群的商業意義標籤 ──────────────────────────────────────────
# 概念：第一版規則把「是否很久沒買」放在最優先判斷，
# 導致平均訂單金額最高、利潤貢獻最高的大額顧客，
# 只因為recency偏高就被誤判成流失顧客，這是規則優先順序錯誤
# 修正後，改以「利潤貢獻的相對排名」作為第一優先判斷依據
# 因為利潤才是最終衡量顧客商業價值的指標，recency只用來當作補充說明
overall_median = customer_data[cluster_cols].median()
profit_rank = cluster_summary['total_profit'].rank(ascending=False)  # 排名1代表利潤最高

labels_map = {}
for cluster_id, row in cluster_summary.iterrows():
    high_freq = row['order_count'] > overall_median['order_count']
    recent = row['recency_days'] < overall_median['recency_days']  # 天數越小代表越近期
    recency_note = '，且為近期購買' if recent else '，但已較久未購買'

    if profit_rank[cluster_id] == 1:
        # 利潤貢獻最高的一群，優先判斷為高價值顧客
        # 不論訂單頻率高低，都先標記為高價值，再用recency補充說明活躍程度
        if high_freq:
            label = '高價值活躍顧客（利潤貢獻最高、購買頻繁）'
        else:
            label = f'高價值大額顧客（利潤貢獻最高、平均單筆金額高{recency_note}）'
    elif profit_rank[cluster_id] == len(profit_rank):
        # 利潤貢獻最低的一群
        label = f'低價值顧客（利潤貢獻最低{recency_note}）'
    elif high_freq and recent:
        label = '活躍高頻顧客（訂單頻繁、近期仍持續購買）'
    else:
        label = f'一般顧客（消費特徵中等{recency_note}）'

    labels_map[cluster_id] = label

print('=== 各群的商業意義標籤 ===')
for cluster_id, label in labels_map.items():
    print(f'  群 {cluster_id}：{label}')
print()

# ══════════════════════════════════════════════════════════════════════════
# 第七部分：視覺化分群結果
# ══════════════════════════════════════════════════════════════════════════

# 用平均訂單金額與訂單數兩個維度做散點圖，顏色代表 Recency
fig, ax = plt.subplots(figsize=(9, 7))
scatter = ax.scatter(
    customer_data['avg_order_value'],
    customer_data['order_count'],
    c=customer_data['cluster'],
    cmap='viridis',
    alpha=0.7,
    s=60
)
ax.set_xlabel('平均每筆訂單金額（Avg Order Value）')
ax.set_ylabel('訂單數量（Order Count）')
ax.set_title(f'Walmart 顧客分群結果（K Means，k={best_k}）')
legend = ax.legend(*scatter.legend_elements(), title='群別')
ax.add_artist(legend)
plt.tight_layout()
plt.savefig('walmart_kmeans_clusters.png', dpi=150)
plt.show()

# 各群在三個 RFM 維度上的比較（用標準化後的數值比較，才能放在同一張圖上）
cluster_scaled_summary = pd.DataFrame(
    X_scaled, columns=cluster_cols
)
cluster_scaled_summary['cluster'] = customer_data['cluster']
cluster_scaled_mean = cluster_scaled_summary.groupby('cluster').mean()

fig, ax = plt.subplots(figsize=(9, 5))
cluster_scaled_mean.plot(kind='bar', ax=ax)
ax.set_title('各群在 RFM 三維度上的標準化平均值比較')
ax.set_xlabel('群別')
ax.set_ylabel('標準化後的平均值')
ax.axhline(0, color='gray', linewidth=0.8)
ax.legend(['平均訂單金額', '訂單數', 'Recency天數'])
plt.tight_layout()
plt.savefig('walmart_cluster_rfm_comparison.png', dpi=150)
plt.show()

print('=== 分析完成 ===')
print('圖表已儲存：walmart_kmeans_selection.png')
print('圖表已儲存：walmart_kmeans_clusters.png')
print('圖表已儲存：walmart_cluster_rfm_comparison.png')