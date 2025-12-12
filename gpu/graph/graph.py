import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re

# 1. 設定檔案路徑
PROFILES_DIR = './profiles'
CSV_PATTERN = re.compile(r'time_distribution-\((.*?)\)\.csv')

# 2. 核心：解析 CSV 檔案並提取數據
def process_csv_file(filepath):
    """
    讀取單個 CSV 檔案，提取兩個主要 Kernel 的時間，並解析檔案名以獲取實作方法和框數。
    """
    try:
        # 讀取 CSV 檔案
        # 由於 CSV 包含了額外的 NVPROF 輸出，我們需要跳過開頭的非表格行
        # df = pd.read_csv(filepath, skiprows=3, header=0) 
        df = pd.read_csv(filepath, skiprows=3, header=0)
        
        # 找出檔名中的實作方法和框數
        filename = os.path.basename(filepath)
        match = CSV_PATTERN.search(filename)
        if not match:
            print(f"Skipping file due to unexpected name format: {filename}")
            return None
        
        # 實作名可能包含多個部分 (e.g., nms-opt-2000-10000, nms-torch-solvebank64-8000)
        full_label = match.group(1) 
        
        # 嘗試從標籤中解析框數和實作類型
        parts = full_label.split('-')
        
        # 假設最後一個數字是主要的框數 (e.g., 8000 in 'nms-opt-8000' or 'nms-opt-2000-10000')
        # 如果是 '2000-10000' 這種範圍，我們使用最大值 10000 作為 X 軸的排序參考
        box_count_str = parts[-1] 
        try:
            box_count = int(box_count_str)
        except ValueError:
            # 處理像 '2000-10000' 這樣的範圍，取最後一個數字作為主要框數
            if '-' in box_count_str:
                box_count = int(box_count_str.split('-')[-1])
            else:
                box_count = 0 # 無法解析，設為 0

        # 實作名稱是不包含數字的部分
        # e.g., 'nms-opt-8000' -> 'nms-opt'
        # e.g., 'nms-torch-solvebank64-8000' -> 'nms-torch-solvebank64'
        implementation = '-'.join(parts[:-1]) if parts[-1].isdigit() else full_label
        # 進一步處理 'nms-opt-2000-10000' 的情況，保留完整的標籤作為唯一 ID
        if implementation.endswith('-' + parts[-2]) and parts[-2].isdigit():
             implementation = '-'.join(parts[:-2]) 
        
        # 確保 `Time` 欄位是數字類型，並清理名稱
        df['Time'] = pd.to_numeric(df['Time'], errors='coerce')
        df['Name'] = df['Name'].astype(str).str.strip()

        # 提取兩個目標 Kernel 的時間 (Time 是以 ms 為單位)
        
        # Kernel 1: NMS 核心計算
        nms_core_name = df[df['Name'].str.contains(r'nms_kernel_impl')]['Name'].iloc[0]
        nms_core_time = df[df['Name'] == nms_core_name]['Time'].sum()
        
        # Kernel 2: 結果收集
        gather_name = df[df['Name'].str.contains(r'gather_keep_from_mask')]['Name'].iloc[0]
        gather_time = df[df['Name'] == gather_name]['Time'].sum()

        # 計算總和
        total_time = nms_core_time + gather_time
        
        return {
            'Implementation': implementation,
            'Box_Count_Label': full_label, # 用於顯示的完整標籤
            'Box_Count': box_count,        # 用於排序的數值
            'Kernel_NMS_Core_Time': nms_core_time,
            'Kernel_Gather_Time': gather_time,
            'Total_Kernel_Time': total_time
        }
        
    except Exception as e:
        print(f"Error processing file {filename}: {e}")
        return None

# 3. 遍歷所有 CSV 檔案
all_data = []
for filename in os.listdir(PROFILES_DIR):
    if filename.endswith('.csv') and filename.startswith('time_distribution-'):
        filepath = os.path.join(PROFILES_DIR, filename)
        data = process_csv_file(filepath)
        if data:
            all_data.append(data)

if not all_data:
    print(f"在 '{PROFILES_DIR}' 目錄中未找到有效的 time_distribution CSV 檔案。")
    exit()

df_results = pd.DataFrame(all_data)

# 4. 繪製圖表
sns.set_theme(style="whitegrid")

# 4.1. 總執行時間 vs. 框數 (X軸為 Box_Count 進行排序)
plt.figure(figsize=(12, 6))
# 為了讓 X 軸排序正確，使用 Box_Count 進行排序
df_results_sorted = df_results.sort_values(by=['Box_Count', 'Implementation'])

sns.lineplot(
    data=df_results_sorted, 
    x='Box_Count', 
    y='Total_Kernel_Time', 
    hue='Implementation', 
    marker='o',
    errorbar=None, # 假設每個配置只有一個數據點
    palette='Spectral'
)

plt.title('NMS 不同實作方法的核心 Kernel 總執行時間比較', fontsize=16)
plt.xlabel('輸入框數量 (Box Count)', fontsize=14)
plt.ylabel('總 Kernel 執行時間 (ms)', fontsize=14)
plt.legend(title='NMS 實作方法', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xscale('log') # 使用對數尺度可以更好地觀察大範圍的輸入
plt.xticks(df_results_sorted['Box_Count'].unique(), labels=df_results_sorted['Box_Count'].unique())
plt.grid(True, which="both", ls="--")
plt.tight_layout()
plt.savefig('nms_total_kernel_time_comparison.png')
print("✅ 圖表 1: nms_total_kernel_time_comparison.png 已儲存。")
plt.show()


# 4.2. 堆疊長條圖：顯示 Kernel 耗時結構
# 只選擇幾個有代表性的框數來繪製，避免圖表過於擁擠
# 找出最大的框數
max_box_count = df_results['Box_Count'].max()
representative_box_counts = [1000, 5000, max_box_count]
# 包含有範圍的數據，例如 '2000-10000'
representative_labels = df_results[
    (df_results['Box_Count'].isin(representative_box_counts)) | (df_results['Box_Count_Label'].str.contains('-'))
]['Box_Count_Label'].unique()

# 整理繪製堆疊圖的數據
df_stack = df_results[df_results['Box_Count_Label'].isin(representative_labels)].copy()

# 將寬格式轉換為長格式，以便繪製堆疊長條圖
df_melted = df_stack.melt(
    id_vars=['Box_Count_Label', 'Box_Count'],
    value_vars=['Kernel_NMS_Core_Time', 'Kernel_Gather_Time'],
    var_name='Kernel_Type',
    value_name='Time_ms'
)

# 再次排序以確保圖表的可讀性
df_melted_sorted = df_melted.sort_values(by=['Box_Count', 'Box_Count_Label'])

plt.figure(figsize=(14, 8))
sns.barplot(
    data=df_melted_sorted,
    x='Box_Count_Label', 
    y='Time_ms', 
    hue='Kernel_Type', 
    palette={"Kernel_NMS_Core_Time": "tab:blue", "Kernel_Gather_Time": "tab:red"},
    dodge=False # 這是關鍵，讓長條圖堆疊起來
)

plt.title('NMS 實作方法的核心 Kernel 時間結構 (部分輸入框數)', fontsize=16)
plt.xlabel('NMS 實作方法與輸入框數量', fontsize=14)
plt.ylabel('Kernel 執行時間 (ms)', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.legend(title='Kernel 功能', labels=['NMS 核心計算', '結果收集 (Gather)'])
plt.tight_layout()
plt.savefig('nms_kernel_time_breakdown_stacked.png')
print("✅ 圖表 2: nms_kernel_time_breakdown_stacked.png 已儲存。")
plt.show()

print("\n---")
print("數據處理完成，請檢查程式碼執行目錄中生成的 PNG 圖表文件。")