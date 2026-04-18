import pandas as pd
import numpy as np
import os
import warnings

warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import OneHotEncoder

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体（解决绘图中文乱码）
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
# 定义股票列表
stocks = ['AAPL', 'GOOGL', 'MSFT', 'NFLX', 'NVDA']

# 定义训练集和测试集的时间范围
train_start = '2013-01-01'
train_end = '2016-12-31'
test_start = '2017-01-01'
test_end = '2018-12-31'

# 创建输出目录
output_dir = 'split_data'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("=" * 80)
print("股票数据划分工具")
print("=" * 80)
print(f"训练集时间范围: {train_start} 至 {train_end}")
print(f"测试集时间范围: {test_start} 至 {test_end}")
print("=" * 80)

for stock in stocks:
    # 构建文件路径
    file_path = f'{stock}_data.csv'

    try:
        # 读取CSV文件
        df = pd.read_csv(file_path)

        # 将date列转换为datetime类型
        df['date'] = pd.to_datetime(df['date'])

        # 划分训练集和测试集
        train_data = df[(df['date'] >= train_start) & (df['date'] <= train_end)].copy()
        test_data = df[(df['date'] >= test_start) & (df['date'] <= test_end)].copy()

        # 保存训练集
        train_file = os.path.join(output_dir, f'{stock}_train.csv')
        train_data.to_csv(train_file, index=False)

        # 保存测试集
        test_file = os.path.join(output_dir, f'{stock}_test.csv')
        test_data.to_csv(test_file, index=False)

        # 打印统计信息
        print(f"\n{stock}:")
        print(f"  原始数据行数: {len(df)}")
        print(
            f"  训练集行数: {len(train_data)} ({train_data['date'].min().strftime('%Y-%m-%d')} 至 {train_data['date'].max().strftime('%Y-%m-%d')})")
        print(
            f"  测试集行数: {len(test_data)} ({test_data['date'].min().strftime('%Y-%m-%d')} 至 {test_data['date'].max().strftime('%Y-%m-%d')})")
        print(f"  训练集已保存至: {train_file}")
        print(f"  测试集已保存至: {test_file}")

    except FileNotFoundError:
        print(f"\n错误: 找不到文件 {file_path}")
    except Exception as e:
        print(f"\n处理 {stock} 时出错: {str(e)}")

print("\n" + "=" * 80)
print("数据划分完成！")
print("=" * 80)


# ==============================
# 特征工程：分类任务（预测 t+1 涨跌 / t+5 涨跌）
# ==============================
def feature_engineering_classification(df):
    """
    分类任务特征工程（同时生成 t+1 和 t+5 标签）
    统一处理，独热编码放在外部统一做
    """
    df = df.copy()

    # 1. 价格基础特征
    df['return'] = df['close'].pct_change()
    df['range'] = df['high'] - df['low']
    df['volatility'] = (df['high'] - df['low']) / df['close']

    # 滞后价格
    df['close_lag1'] = df['close'].shift(1)
    df['close_lag3'] = df['close'].shift(3)
    df['close_lag5'] = df['close'].shift(5)

    # 2. 技术指标
    df['sma5'] = df['close'].rolling(5).mean()
    df['sma10'] = df['close'].rolling(10).mean()
    df['sma20'] = df['close'].rolling(20).mean()
    df['trend'] = df['close'] / df['sma20']

    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    # ATR
    df['tr'] = np.maximum(
        df['high'] - df['low'],
        abs(df['high'] - df['close'].shift(1)),
        abs(df['low'] - df['close'].shift(1))
    )
    df['atr'] = df['tr'].rolling(14).mean()

    # 3. 量价特征
    df['volume_change'] = df['volume'].pct_change()

    # 4. 日期特征
    df['weekday'] = df['date'].dt.weekday
    df['month'] = df['date'].dt.month

    # 5. 滚动波动率
    df['return_roll5'] = df['return'].rolling(5).mean()
    df['std_roll5'] = df['return'].rolling(5).std()

    # ----------------------------------------------------
    # 【关键】生成两个分类标签：t+1 涨跌 + t+5 涨跌
    # 1=涨，0=跌
    # ----------------------------------------------------
    df['target_t1'] = (df['close'].shift(-1) > df['close']).astype(int)
    df['target_t5'] = (df['close'].shift(-5) > df['close']).astype(int)

    # 去掉缺失值（必须放在标签生成之后）
    df = df.dropna()

    return df


# ==============================
# 对划分后的数据进行特征工程
# ==============================
print("\n" + "=" * 80)
print("步骤2: 特征工程 - 分类任务")
print("=" * 80)

all_train_data = []
all_test_data = []

for stock in stocks:
    train_file = os.path.join(output_dir, f'{stock}_train.csv')
    test_file = os.path.join(output_dir, f'{stock}_test.csv')

    try:
        # 读取训练集
        train_df = pd.read_csv(train_file)
        train_df['date'] = pd.to_datetime(train_df['date'])
        train_df['stock_code'] = stock

        # 读取测试集
        test_df = pd.read_csv(test_file)
        test_df['date'] = pd.to_datetime(test_df['date'])
        test_df['stock_code'] = stock

        all_train_data.append(train_df)
        all_test_data.append(test_df)

        print(f"\n{stock} 数据加载成功")
        print(f"  训练集: {len(train_df)} 行")
        print(f"  测试集: {len(test_df)} 行")

    except FileNotFoundError as e:
        print(f"\n错误: 找不到文件 {e.filename}")
    except Exception as e:
        print(f"\n处理 {stock} 时出错: {str(e)}")

if all_train_data and all_test_data:
    # 合并所有股票数据
    train_combined = pd.concat(all_train_data, ignore_index=True)
    test_combined = pd.concat(all_test_data, ignore_index=True)

    print(f"\n合并后 - 训练集: {len(train_combined)} 行, 测试集: {len(test_combined)} 行")

    # 应用特征工程
    print("\n正在进行特征工程...")
    train_featured = feature_engineering_classification(train_combined)
    test_featured = feature_engineering_classification(test_combined)

    # ==============================================
    # 【修复】统一对 训练集+测试集 做独热编码
    # ==============================================

    encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')

    # 用训练集拟合编码器
    encoder.fit(train_featured[['stock_code', 'weekday', 'month']])

    # 生成编码列名
    cat_cols = ['stock_code', 'weekday', 'month']
    encoded_cols = []
    for i, col in enumerate(cat_cols):
        categories = encoder.categories_[i][1:]  # drop first
        encoded_cols.extend([f"{col}_{c}" for c in categories])

    # 编码训练集
    train_encoded = encoder.transform(train_featured[cat_cols])
    train_encoded_df = pd.DataFrame(train_encoded, columns=encoded_cols, index=train_featured.index)

    # 编码测试集
    test_encoded = encoder.transform(test_featured[cat_cols])
    test_encoded_df = pd.DataFrame(test_encoded, columns=encoded_cols, index=test_featured.index)

    # 合并
    train_featured = pd.concat([train_featured.drop(columns=cat_cols), train_encoded_df], axis=1)
    test_featured = pd.concat([test_featured.drop(columns=cat_cols), test_encoded_df], axis=1)

    # 强制列完全一致
    test_featured = test_featured.reindex(columns=train_featured.columns, fill_value=0)

    # 删除字符串列 Name,防止StandardScaler遇到字符报错
    train_featured = train_featured.drop(columns=['Name'])
    test_featured = test_featured.drop(columns=['Name'])

    # 保存特征工程后的数据
    train_featured_file = os.path.join(output_dir, 'train_featured.csv')
    test_featured_file = os.path.join(output_dir, 'test_featured.csv')

    train_featured.to_csv(train_featured_file, index=False)
    test_featured.to_csv(test_featured_file, index=False)

    print(f"\n特征工程完成！")
    print(f"  训练集特征数: {train_featured.shape[1] - 2}")  # 减去2个target
    print(f"  训练集样本数: {len(train_featured)}")
    print(f"  测试集样本数: {len(test_featured)}")
    print(f"  已保存至: {train_featured_file}")
    print(f"  已保存至: {test_featured_file}")

    # 显示目标变量分布
    print(f"\ntarget_t1 分布（训练集）:")
    print(train_featured['target_t1'].value_counts())
    print(f"\ntarget_t5 分布（训练集）:")
    print(train_featured['target_t5'].value_counts())

    # ==============================================
    # 【验证】直接打印你关心的 stock_code 独热列
    # ==============================================
    print("\n" + "=" * 50)
    print("✅ 已生成的股票独热编码列：")
    stock_cols = [c for c in train_featured.columns if 'stock_code' in c]
    print(stock_cols)
    print("=" * 50)

else:
    print("\n错误: 没有成功加载任何数据，无法进行特征工程")

print("\n" + "=" * 80)
print("全部完成！")
print("=" * 80)

# ==============================
# 步骤3: 模型训练与评估
# 包含：逻辑回归、随机森林、LSTM(TimeSeriesSplit)
# 预测目标：t+1涨跌 / t+5涨跌
# ==============================


print("\n" + "=" * 80)
print("步骤3: 模型训练与评估")
print("=" * 80)

# ----------------------
# 1. 加载特征工程后的数据
# ----------------------
train_df = pd.read_csv(os.path.join(output_dir, 'train_featured.csv'))
test_df = pd.read_csv(os.path.join(output_dir, 'test_featured.csv'))

# 分离特征与标签
targets = ['target_t1', 'target_t5']
drop_cols = ['date'] + targets

X_train = train_df.drop(columns=drop_cols)
y_train_t1 = train_df['target_t1']
y_train_t5 = train_df['target_t5']

X_test = test_df.drop(columns=drop_cols)
y_test_t1 = test_df['target_t1']
y_test_t5 = test_df['target_t5']

print(f"训练集特征维度: {X_train.shape}")
print(f"测试集特征维度: {X_test.shape}")
print(f"t+1 训练标签分布:\n{y_train_t1.value_counts()}")
print(f"t+5 训练标签分布:\n{y_train_t5.value_counts()}")
print("DataFrame info:")
print(X_train.info())
print("\nDataFrame dtypes:")
print(X_train.dtypes)
print("\nFirst few rows:")
print(X_train.head())
# 特征标准化（逻辑回归、LSTM需要）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ----------------------
# 通用评估函数
# ----------------------
def evaluate_model(y_true, y_pred, model_name, target_name):
    print(f"\n📊 {model_name} - {target_name} 评估结果")
    print("-" * 50)
    print(f"准确率: {accuracy_score(y_true, y_pred):.4f}")
    print("\n分类报告:")
    print(classification_report(y_true, y_pred, zero_division=0))

    # 混淆矩阵热力图
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{model_name} - {target_name} 混淆矩阵')
    plt.xlabel('预测值')
    plt.ylabel('真实值')
    plt.tight_layout()


# ==============================================================================
# 模型1：逻辑回归 - t+1 涨跌预测
# ==============================================================================
print("\n" + "=" * 50)
print("模型1: 逻辑回归 - t+1 涨跌预测")
print("=" * 50)
lr_t1 = LogisticRegression(max_iter=1000, random_state=42)
lr_t1.fit(X_train_scaled, y_train_t1)
y_pred_lr_t1 = lr_t1.predict(X_test_scaled)
evaluate_model(y_test_t1, y_pred_lr_t1, "逻辑回归", "t+1涨跌")

# ==============================================================================
# 模型2：逻辑回归 - t+5 涨跌预测
# ==============================================================================
print("\n" + "=" * 50)
print("模型2: 逻辑回归 - t+5 涨跌预测")
print("=" * 50)
lr_t5 = LogisticRegression(max_iter=1000, random_state=42)
lr_t5.fit(X_train_scaled, y_train_t5)
y_pred_lr_t5 = lr_t5.predict(X_test_scaled)
evaluate_model(y_test_t5, y_pred_lr_t5, "逻辑回归", "t+5涨跌")

# ==============================================================================
# 模型3：随机森林 - t+1 涨跌预测
# ==============================================================================
print("\n" + "=" * 50)
print("模型3: 随机森林 - t+1 涨跌预测")
print("=" * 50)
rf_t1 = RandomForestClassifier(n_estimators=100, random_state=42)
rf_t1.fit(X_train, y_train_t1)
y_pred_rf_t1 = rf_t1.predict(X_test)
evaluate_model(y_test_t1, y_pred_rf_t1, "随机森林", "t+1涨跌")

# 特征重要性
plt.figure(figsize=(12, 6))
imp = pd.Series(rf_t1.feature_importances_, index=X_train.columns).sort_values(ascending=False).head(15)
imp.plot(kind='bar')
plt.title('随机森林 t+1 特征重要性 Top15')
plt.tight_layout()

# ==============================================================================
# 模型4：随机森林 - t+5 涨跌预测
# ==============================================================================
print("\n" + "=" * 50)
print("模型4: 随机森林 - t+5 涨跌预测")
print("=" * 50)
rf_t5 = RandomForestClassifier(n_estimators=100, random_state=42)
rf_t5.fit(X_train, y_train_t5)
y_pred_rf_t5 = rf_t5.predict(X_test)
evaluate_model(y_test_t5, y_pred_rf_t5, "随机森林", "t+5涨跌")

# 特征重要性
plt.figure(figsize=(12, 6))
imp = pd.Series(rf_t5.feature_importances_, index=X_train.columns).sort_values(ascending=False).head(15)
imp.plot(kind='bar')
plt.title('随机森林 t+5 特征重要性 Top15')
plt.tight_layout()


# ==============================================================================
# 【LSTM专用】时序数据构造函数
# ==============================================================================
def create_sequences(X, y, time_steps=10):
    """将二维数据转为LSTM需要的3维数据 (samples, time_steps, features)"""
    Xs, ys = [], []
    for i in range(time_steps, len(X)):
        Xs.append(X[i - time_steps:i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)


# LSTM时序步长
TIME_STEPS = 10

# ==============================================================================
# 模型5：LSTM + TimeSeriesSplit - t+1 涨跌预测
# ==============================================================================
print("\n" + "=" * 50)
print("模型5: LSTM + TimeSeriesSplit - t+1 涨跌预测")
print("=" * 50)

# 时序交叉验证
tscv = TimeSeriesSplit(n_splits=5)
val_accs = []

for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train_scaled)):
    print(f"\n🔁 LSTM t+1 第 {fold + 1} 折训练...")

    # 训练/验证划分
    X_t, X_v = X_train_scaled[train_idx], X_train_scaled[val_idx]
    y_t, y_v = y_train_t1.values[train_idx], y_train_t1.values[val_idx]

    # 构造时序序列
    X_t_seq, y_t_seq = create_sequences(X_t, y_t, TIME_STEPS)
    X_v_seq, y_v_seq = create_sequences(X_v, y_v, TIME_STEPS)

    # 构建LSTM
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(TIME_STEPS, X_train.shape[1])),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # 早停
    es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # 训练
    model.fit(
        X_t_seq, y_t_seq,
        validation_data=(X_v_seq, y_v_seq),
        epochs=30, batch_size=32,
        callbacks=[es], verbose=1
    )

    # 验证评估
    val_pred = (model.predict(X_v_seq, verbose=0) > 0.5).astype(int)
    val_acc = accuracy_score(y_v_seq, val_pred)
    val_accs.append(val_acc)
    print(f"第{fold + 1}折验证准确率: {val_acc:.4f}")

print(f"\n✅ LSTM t+1 5折平均验证准确率: {np.mean(val_accs):.4f}")

# 最终测试
X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_t1.values, TIME_STEPS)
X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_t1.values, TIME_STEPS)

# 重新训练最终模型
lstm_t1 = Sequential([
    LSTM(64, return_sequences=True, input_shape=(TIME_STEPS, X_train.shape[1])),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])
lstm_t1.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
lstm_t1.fit(X_train_seq, y_train_seq, epochs=20, batch_size=32, verbose=1)

# 测试预测
y_pred_lstm_t1 = (lstm_t1.predict(X_test_seq, verbose=0) > 0.5).astype(int)
# 对齐真实标签长度
y_test_aligned = y_test_seq[-len(y_pred_lstm_t1):]
evaluate_model(y_test_aligned, y_pred_lstm_t1, "LSTM", "t+1涨跌")

# ==============================================================================
# 模型6：LSTM + TimeSeriesSplit - t+5 涨跌预测
# ==============================================================================
print("\n" + "=" * 50)
print("模型6: LSTM + TimeSeriesSplit - t+5 涨跌预测")
print("=" * 50)

tscv = TimeSeriesSplit(n_splits=5)
val_accs_t5 = []

for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train_scaled)):
    print(f"\n🔁 LSTM t+5 第 {fold + 1} 折训练...")

    X_t, X_v = X_train_scaled[train_idx], X_train_scaled[val_idx]
    y_t, y_v = y_train_t5.values[train_idx], y_train_t5.values[val_idx]

    X_t_seq, y_t_seq = create_sequences(X_t, y_t, TIME_STEPS)
    X_v_seq, y_v_seq = create_sequences(X_v, y_v, TIME_STEPS)

    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(TIME_STEPS, X_train.shape[1])),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    model.fit(
        X_t_seq, y_t_seq,
        validation_data=(X_v_seq, y_v_seq),
        epochs=30, batch_size=32,
        callbacks=[es], verbose=1
    )

    val_pred = (model.predict(X_v_seq, verbose=0) > 0.5).astype(int)
    val_acc = accuracy_score(y_v_seq, val_pred)
    val_accs_t5.append(val_acc)
    print(f"第{fold + 1}折验证准确率: {val_acc:.4f}")

print(f"\n✅ LSTM t+5 5折平均验证准确率: {np.mean(val_accs_t5):.4f}")

# 最终测试
X_train_seq_t5, y_train_seq_t5 = create_sequences(X_train_scaled, y_train_t5.values, TIME_STEPS)
X_test_seq_t5, y_test_seq_t5 = create_sequences(X_test_scaled, y_test_t5.values, TIME_STEPS)

lstm_t5 = Sequential([
    LSTM(64, return_sequences=True, input_shape=(TIME_STEPS, X_train.shape[1])),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])
lstm_t5.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
lstm_t5.fit(X_train_seq_t5, y_train_seq_t5, epochs=20, batch_size=32, verbose=1)

y_pred_lstm_t5 = (lstm_t5.predict(X_test_seq_t5, verbose=0) > 0.5).astype(int)
y_test_aligned_t5 = y_test_seq_t5[-len(y_pred_lstm_t5):]
evaluate_model(y_test_aligned_t5, y_pred_lstm_t5, "LSTM", "t+5涨跌")

# ==============================
# 最终总结
# ==============================
print("\n" + "=" * 80)
print("🎉 全部6个模型训练完成！")
print("包含：逻辑回归(t+1/t+5)、随机森林(t+1/t+5)、LSTM时序交叉验证(t+1/t+5)")
print("=" * 80)

# ==============================
# 步骤4：统一对比 t+1 与 t+5 性能
# 指标：准确率、F1、AUC-ROC
# 图表保存到 result/ 文件夹
# ==============================


# 创建结果保存文件夹
result_dir = 'result'
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

# 中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ======================
# 收集所有模型结果
# ======================
results = []

# 1. 逻辑回归 t+1
acc_lr_t1 = accuracy_score(y_test_t1, y_pred_lr_t1)
f1_lr_t1 = f1_score(y_test_t1, y_pred_lr_t1)
auc_lr_t1 = roc_auc_score(y_test_t1, lr_t1.predict_proba(X_test_scaled)[:,1])
results.append(['逻辑回归', 't+1', acc_lr_t1, f1_lr_t1, auc_lr_t1])

# 2. 逻辑回归 t+5
acc_lr_t5 = accuracy_score(y_test_t5, y_pred_lr_t5)
f1_lr_t5 = f1_score(y_test_t5, y_pred_lr_t5)
auc_lr_t5 = roc_auc_score(y_test_t5, lr_t5.predict_proba(X_test_scaled)[:,1])
results.append(['逻辑回归', 't+5', acc_lr_t5, f1_lr_t5, auc_lr_t5])

# 3. 随机森林 t+1
acc_rf_t1 = accuracy_score(y_test_t1, y_pred_rf_t1)
f1_rf_t1 = f1_score(y_test_t1, y_pred_rf_t1)
auc_rf_t1 = roc_auc_score(y_test_t1, rf_t1.predict_proba(X_test)[:,1])
results.append(['随机森林', 't+1', acc_rf_t1, f1_rf_t1, auc_rf_t1])

# 4. 随机森林 t+5
acc_rf_t5 = accuracy_score(y_test_t5, y_pred_rf_t5)
f1_rf_t5 = f1_score(y_test_t5, y_pred_rf_t5)
auc_rf_t5 = roc_auc_score(y_test_t5, rf_t5.predict_proba(X_test)[:,1])
results.append(['随机森林', 't+5', acc_rf_t5, f1_rf_t5, auc_rf_t5])

# 5. LSTM t+1
acc_lstm_t1 = accuracy_score(y_test_aligned, y_pred_lstm_t1)
f1_lstm_t1 = f1_score(y_test_aligned, y_pred_lstm_t1)
auc_lstm_t1 = roc_auc_score(y_test_aligned, lstm_t1.predict(X_test_seq, verbose=0))
results.append(['LSTM', 't+1', acc_lstm_t1, f1_lstm_t1, auc_lstm_t1])

# 6. LSTM t+5
acc_lstm_t5 = accuracy_score(y_test_aligned_t5, y_pred_lstm_t5)
f1_lstm_t5 = f1_score(y_test_aligned_t5, y_pred_lstm_t5)
auc_lstm_t5 = roc_auc_score(y_test_aligned_t5, lstm_t5.predict(X_test_seq_t5, verbose=0))
results.append(['LSTM', 't+5', acc_lstm_t5, f1_lstm_t5, auc_lstm_t5])

# 转DataFrame
df_res = pd.DataFrame(results, columns=['模型', '预测周期', '准确率', 'F1分数', 'AUC'])
print("\n" + "="*80)
print("📊 所有模型 t+1 vs t+5 性能对比")
print("="*80)
print(df_res.round(4))

# ======================
# 可视化 1：准确率对比
# ======================
plt.figure(figsize=(10,5))
models = ['逻辑回归', '随机森林', 'LSTM']
t1_acc = [acc_lr_t1, acc_rf_t1, acc_lstm_t1]
t5_acc = [acc_lr_t5, acc_rf_t5, acc_lstm_t5]
x = np.arange(len(models))
width = 0.35

plt.bar(x - width/2, t1_acc, width, label='t+1 (短期)')
plt.bar(x + width/2, t5_acc, width, label='t+5 (中长期)')
plt.xlabel('模型')
plt.ylabel('准确率')
plt.title('准确率对比：t+1 短期预测 vs t+5 中长期预测')
plt.xticks(x, models)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(result_dir, 'accuracy_compare.png'), dpi=300)
plt.close()

# ======================
# 可视化 2：F1分数对比
# ======================
plt.figure(figsize=(10,5))
t1_f1 = [f1_lr_t1, f1_rf_t1, f1_lstm_t1]
t5_f1 = [f1_lr_t5, f1_rf_t5, f1_lstm_t5]

plt.bar(x - width/2, t1_f1, width, label='t+1 (短期)')
plt.bar(x + width/2, t5_f1, width, label='t+5 (中长期)')
plt.xlabel('模型')
plt.ylabel('F1分数')
plt.title('F1分数对比：t+1 短期预测 vs t+5 中长期预测')
plt.xticks(x, models)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(result_dir, 'f1_compare.png'), dpi=300)
plt.close()

# ======================
# 可视化 3：AUC-ROC 曲线汇总
# ======================
plt.figure(figsize=(10,8))

# LR t+1
fpr, tpr, _ = roc_curve(y_test_t1, lr_t1.predict_proba(X_test_scaled)[:,1])
plt.plot(fpr, tpr, label=f'LR t+1 (AUC={auc_lr_t1:.3f})')

# LR t+5
fpr, tpr, _ = roc_curve(y_test_t5, lr_t5.predict_proba(X_test_scaled)[:,1])
plt.plot(fpr, tpr, label=f'LR t+5 (AUC={auc_lr_t5:.3f})')

# RF t+1
fpr, tpr, _ = roc_curve(y_test_t1, rf_t1.predict_proba(X_test)[:,1])
plt.plot(fpr, tpr, label=f'RF t+1 (AUC={auc_rf_t1:.3f})')

# RF t+5
fpr, tpr, _ = roc_curve(y_test_t5, rf_t5.predict_proba(X_test)[:,1])
plt.plot(fpr, tpr, label=f'RF t+5 (AUC={auc_rf_t5:.3f})')

# LSTM t+1
fpr, tpr, _ = roc_curve(y_test_aligned, lstm_t1.predict(X_test_seq, verbose=0))
plt.plot(fpr, tpr, label=f'LSTM t+1 (AUC={auc_lstm_t1:.3f})')

# LSTM t+5
fpr, tpr, _ = roc_curve(y_test_aligned_t5, lstm_t5.predict(X_test_seq_t5, verbose=0))
plt.plot(fpr, tpr, label=f'LSTM t+5 (AUC={auc_lstm_t5:.3f})')

plt.plot([0,1],[0,1], 'k--')
plt.xlabel('假正率 FPR')
plt.ylabel('真正率 TPR')
plt.title('AUC-ROC 曲线对比：t+1 vs t+5')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(result_dir, 'auc_roc_compare.png'), dpi=300)
plt.close()

# ======================
# 保存结果表格
# ======================
df_res.round(4).to_csv(os.path.join(result_dir, 'model_performance.csv'), index=False, encoding='utf-8-sig')

print("\n✅ 所有图表已保存到：result/ 文件夹")
print("📁 包含：")
print("   - accuracy_compare.png   准确率对比")
print("   - f1_compare.png         F1分数对比")
print("   - auc_roc_compare.png    ROC曲线")
print("   - model_performance.csv  详细结果表格")