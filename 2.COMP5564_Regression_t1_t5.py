
import os
import warnings

warnings.filterwarnings('ignore')
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import TimeSeriesSplit
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
# 特征工程：回归任务（预测 t+1 收盘价 / t+5 收盘价）
# ==============================
def feature_engineering_regression(df):
    """
    回归任务特征工程（同时生成 t+1 和 t+5 标签）
    统一处理，独热编码放在外部统一做
    """
    df = df.copy()

    # 1. 价格基础特征
    df['return'] = df['close'].pct_change()
    df['range'] = df['high'] - df['low']
    df['volatility'] = (df['high'] - df['low']) / df['close']

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
    # 【关键】生成两个分类标签：t+1 收盘价 + t+5 收盘价
    # ----------------------------------------------------
    df['target_t1'] = df['close'].shift(-1)   # 预测下一日收盘价
    df['target_t5'] = df['close'].shift(-5)   # 预测第5日收盘价

    # 6. 去掉缺失值（必须放在标签生成之后）
    df = df.dropna()

    return df


# ==============================
# 对划分后的数据进行特征工程
# ==============================
print("\n" + "=" * 80)
print("步骤2: 特征工程 - 回归任务")
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
    train_featured = feature_engineering_regression(train_combined)
    test_featured = feature_engineering_regression(test_combined)

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
    train_featured_file = os.path.join(output_dir, 'train_featured_reg.csv')
    test_featured_file = os.path.join(output_dir, 'test_featured_reg.csv')

    train_featured.to_csv(train_featured_file, index=False)
    test_featured.to_csv(test_featured_file, index=False)

    print(f"\n回归特征工程完成！")
    print(f"  训练集特征数: {train_featured.shape[1] - 2}")  # 减去2个target
    print(f"  训练集样本数: {len(train_featured)}")
    print(f"  测试集样本数: {len(test_featured)}")
    print(f"  已保存至: {train_featured_file}")
    print(f"  已保存至: {test_featured_file}")

    # 查看标签范围（回归是连续价格）
    print("\n【回归标签统计 - 训练集】")
    print("target_t1（t+1收盘价）范围：")
    print(train_featured['target_t1'].describe())
    print("\ntarget_t5（t+5收盘价）范围：")
    print(train_featured['target_t5'].describe())

else:
    print("\n错误: 没有成功加载任何数据，无法进行特征工程")

print("\n" + "=" * 80)
print("回归任务特征工程全部完成！")
print("=" * 80)




# ==============================
# 步骤3: 模型训练与评估（6大模型）
# ==============================


# 绘图风格设置
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 加载特征工程后的数据
train_df = pd.read_csv(os.path.join(output_dir, 'train_featured_reg.csv'))
test_df = pd.read_csv(os.path.join(output_dir, 'test_featured_reg.csv'))

# 分离特征和标签（通用）
X_train = train_df.drop(['date', 'target_t1', 'target_t5'], axis=1)
X_test = test_df.drop(['date', 'target_t1', 'target_t5'], axis=1)
y_train_t1 = train_df['target_t1']
y_test_t1 = test_df['target_t1']
y_train_t5 = train_df['target_t5']
y_test_t5 = test_df['target_t5']

# 精简快速检查 X_train
print("X_train 形状：", X_train.shape)
print("\n缺失值：\n", X_train.isnull().sum()[X_train.isnull().sum() > 0])
print("\n数据类型：\n", X_train.dtypes)
print("\n前5行：\n", X_train.head())

# 评估函数
def evaluate_model(y_true, y_pred, model_name, target_name):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"\n{model_name} - {target_name} 评估结果：")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R²: {r2:.4f}")
    return {'RMSE': rmse, 'MAE': mae, 'R2': r2}

# 绘图函数
def plot_predictions(y_true, y_pred, model_name, target_name):
    plt.figure(figsize=(12, 5))
    plt.plot(y_true, label='真实值', alpha=0.7)
    plt.plot(y_pred, label='预测值', alpha=0.7)
    plt.title(f'{model_name} - {target_name} 预测效果')
    plt.xlabel('样本序号')
    plt.ylabel('收盘价')
    plt.legend()
    plt.grid(alpha=0.3)

# 创建模型保存目录
model_dir = 'models'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

print("\n" + "=" * 80)
print("步骤3: 开始训练6个预测模型")
print("=" * 80)

# ==============================================
# 1. 线性回归 - t+1
# ==============================================
print("\n【1/6】训练 线性回归 - t+1 价格预测")
lr_t1 = LinearRegression()
lr_t1.fit(X_train, y_train_t1)
y_pred_lr_t1 = lr_t1.predict(X_test)
eval_lr_t1 = evaluate_model(y_test_t1, y_pred_lr_t1, "线性回归", "t+1")
plot_predictions(y_test_t1, y_pred_lr_t1, "线性回归", "t+1")

# ==============================================
# 2. 线性回归 - t+5
# ==============================================
print("\n【2/6】训练 线性回归 - t+5 价格预测")
lr_t5 = LinearRegression()
lr_t5.fit(X_train, y_train_t5)
y_pred_lr_t5 = lr_t5.predict(X_test)
eval_lr_t5 = evaluate_model(y_test_t5, y_pred_lr_t5, "线性回归", "t+5")
plot_predictions(y_test_t5, y_pred_lr_t5, "线性回归", "t+5")

# ==============================================
# 3. XGBoost - t+1
# ==============================================
print("\n【3/6】训练 XGBoost - t+1 价格预测")
xgb_t1 = xgb.XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)
xgb_t1.fit(X_train, y_train_t1)
y_pred_xgb_t1 = xgb_t1.predict(X_test)
eval_xgb_t1 = evaluate_model(y_test_t1, y_pred_xgb_t1, "XGBoost", "t+1")
plot_predictions(y_test_t1, y_pred_xgb_t1, "XGBoost", "t+1")

# ==============================================
# 4. XGBoost - t+5
# ==============================================
print("\n【4/6】训练 XGBoost - t+5 价格预测")
xgb_t5 = xgb.XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)
xgb_t5.fit(X_train, y_train_t5)
y_pred_xgb_t5 = xgb_t5.predict(X_test)
eval_xgb_t5 = evaluate_model(y_test_t5, y_pred_xgb_t5, "XGBoost", "t+5")
plot_predictions(y_test_t5, y_pred_xgb_t5, "XGBoost", "t+5")

# ==============================================
# LSTM 数据预处理
# ==============================================
def create_sequences(X, y, time_steps=10):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X.iloc[i:(i + time_steps)].values)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)

time_steps = 10
# LSTM 输入形状：[样本数, 时间步, 特征数]
X_train_lstm, y_train_lstm_t1 = create_sequences(X_train, y_train_t1, time_steps)
X_test_lstm, y_test_lstm_t1 = create_sequences(X_test, y_test_t1, time_steps)
_, y_train_lstm_t5 = create_sequences(X_train, y_train_t5, time_steps)
_, y_test_lstm_t5 = create_sequences(X_test, y_test_t5, time_steps)

# ==============================================
# 5. LSTM + TimeSeriesSplit - t+1
# ==============================================
print("\n【5/6】训练 LSTM + 时间序列交叉验证 - t+1")
tscv = TimeSeriesSplit(n_splits=5)
fold_scores = []

for train_idx, val_idx in tscv.split(X_train_lstm):
    X_t, X_v = X_train_lstm[train_idx], X_train_lstm[val_idx]
    y_t, y_v = y_train_lstm_t1[train_idx], y_train_lstm_t1[val_idx]

    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(time_steps, X_train.shape[1])),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_t, y_t, epochs=10, batch_size=32, validation_data=(X_v, y_v), verbose=0)
    score = model.evaluate(X_v, y_v, verbose=0)
    fold_scores.append(score)

# 最终训练
lstm_t1 = Sequential([
    LSTM(64, return_sequences=True, input_shape=(time_steps, X_train.shape[1])),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(1)
])
lstm_t1.compile(optimizer='adam', loss='mse')
lstm_t1.fit(X_train_lstm, y_train_lstm_t1, epochs=15, batch_size=32, verbose=1)
y_pred_lstm_t1 = lstm_t1.predict(X_test_lstm, verbose=0).flatten()
eval_lstm_t1 = evaluate_model(y_test_lstm_t1, y_pred_lstm_t1, "LSTM", "t+1")
plot_predictions(y_test_lstm_t1, y_pred_lstm_t1, "LSTM", "t+1")

# ==============================================
# 6. LSTM + TimeSeriesSplit - t+5
# ==============================================
print("\n【6/6】训练 LSTM + 时间序列交叉验证 - t+5")
fold_scores = []

for train_idx, val_idx in tscv.split(X_train_lstm):
    X_t, X_v = X_train_lstm[train_idx], X_train_lstm[val_idx]
    y_t, y_v = y_train_lstm_t5[train_idx], y_train_lstm_t5[val_idx]

    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(time_steps, X_train.shape[1])),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_t, y_t, epochs=10, batch_size=32, validation_data=(X_v, y_v), verbose=0)
    score = model.evaluate(X_v, y_v, verbose=0)
    fold_scores.append(score)

lstm_t5 = Sequential([
    LSTM(64, return_sequences=True, input_shape=(time_steps, X_train.shape[1])),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(1)
])
lstm_t5.compile(optimizer='adam', loss='mse')
lstm_t5.fit(X_train_lstm, y_train_lstm_t5, epochs=15, batch_size=32, verbose=1)
y_pred_lstm_t5 = lstm_t5.predict(X_test_lstm, verbose=0).flatten()
eval_lstm_t5 = evaluate_model(y_test_lstm_t5, y_pred_lstm_t5, "LSTM", "t+5")
plot_predictions(y_test_lstm_t5, y_pred_lstm_t5, "LSTM", "t+5")

# ==============================================
# 最终汇总表
# ==============================================
print("\n" + "=" * 80)
print("所有模型训练完成！最终评估汇总")
print("=" * 80)

results = pd.DataFrame({
    '模型': ['线性回归_t1', '线性回归_t5', 'XGBoost_t1', 'XGBoost_t5', 'LSTM_t1', 'LSTM_t5'],
    'RMSE': [eval_lr_t1['RMSE'], eval_lr_t5['RMSE'],
             eval_xgb_t1['RMSE'], eval_xgb_t5['RMSE'],
             eval_lstm_t1['RMSE'], eval_lstm_t5['RMSE']],
    'MAE': [eval_lr_t1['MAE'], eval_lr_t5['MAE'],
            eval_xgb_t1['MAE'], eval_xgb_t5['MAE'],
            eval_lstm_t1['MAE'], eval_lstm_t5['MAE']],
    'R2': [eval_lr_t1['R2'], eval_lr_t5['R2'],
           eval_xgb_t1['R2'], eval_xgb_t5['R2'],
           eval_lstm_t1['R2'], eval_lstm_t5['R2']]
})

print(results.round(4))
print("\n所有模型已训练完成！")

# ==============================
# 最终对比：同一模型 t+1 vs t+5 性能差异
# 可视化 + 自动保存到 result 文件夹
# ==============================
import matplotlib.pyplot as plt
import pandas as pd
import os

# 创建结果保存文件夹
result_dir = 'result'
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

# 绘图风格（中文正常显示）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150  # 高清图

# ======================
# 构造对比数据
# ======================
comparison = pd.DataFrame({
    'model': ['Linear', 'Linear', 'XGBoost', 'XGBoost', 'LSTM', 'LSTM'],
    'target': ['t+1', 't+5', 't+1', 't+5', 't+1', 't+5'],
    'RMSE': [
        eval_lr_t1['RMSE'], eval_lr_t5['RMSE'],
        eval_xgb_t1['RMSE'], eval_xgb_t5['RMSE'],
        eval_lstm_t1['RMSE'], eval_lstm_t5['RMSE']
    ],
    'MAE': [
        eval_lr_t1['MAE'], eval_lr_t5['MAE'],
        eval_xgb_t1['MAE'], eval_xgb_t5['MAE'],
        eval_lstm_t1['MAE'], eval_lstm_t5['MAE']
    ],
    'R2': [
        eval_lr_t1['R2'], eval_lr_t5['R2'],
        eval_xgb_t1['R2'], eval_xgb_t5['R2'],
        eval_lstm_t1['R2'], eval_lstm_t5['R2']
    ]
})

print("\n" + "=" * 80)
print("同一模型 t+1（短期） vs t+5（长期） 性能对比表")
print("=" * 80)
print(comparison.round(4))

# ======================
# 对比图1：RMSE 对比
# ======================
plt.figure(figsize=(10, 5))
for idx, m in enumerate(['Linear', 'XGBoost', 'LSTM']):
    sub = comparison[comparison['model'] == m]
    plt.bar([x + idx * 0.25 for x in [0, 1]], sub['RMSE'], width=0.25, label=m)

plt.xticks([0.25, 1.25], ['t+1 短期预测', 't+5 长期预测'])
plt.title('各模型 短期(t+1) vs 长期(t+5) —— RMSE 对比（越低越好）')
plt.ylabel('RMSE')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(result_dir, 'rmse_compare.png'), dpi=150)
plt.close()

# ======================
# 对比图2：MAE 对比
# ======================
plt.figure(figsize=(10, 5))
for idx, m in enumerate(['Linear', 'XGBoost', 'LSTM']):
    sub = comparison[comparison['model'] == m]
    plt.bar([x + idx * 0.25 for x in [0, 1]], sub['MAE'], width=0.25, label=m)

plt.xticks([0.25, 1.25], ['t+1 短期预测', 't+5 长期预测'])
plt.title('各模型 短期(t+1) vs 长期(t+5) —— MAE 对比（越低越好）')
plt.ylabel('MAE')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(result_dir, 'mae_compare.png'), dpi=150)
plt.close()

# ======================
# 对比图3：R² 对比
# ======================
plt.figure(figsize=(10, 5))
for idx, m in enumerate(['Linear', 'XGBoost', 'LSTM']):
    sub = comparison[comparison['model'] == m]
    plt.bar([x + idx * 0.25 for x in [0, 1]], sub['R2'], width=0.25, label=m)

plt.xticks([0.25, 1.25], ['t+1 短期预测', 't+5 长期预测'])
plt.title('各模型 短期(t+1) vs 长期(t+5) —— R² 对比（越高越好）')
plt.ylabel('R²')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(result_dir, 'r2_compare.png'), dpi=150)
plt.close()

# ======================
# 每个模型单独对比图（最关键！）
# ======================
for model_name in ['Linear', 'XGBoost', 'LSTM']:
    sub = comparison[comparison['model'] == model_name]

    plt.figure(figsize=(12, 4))

    # RMSE
    plt.subplot(1, 3, 1)
    plt.bar(sub['target'], sub['RMSE'], color=['#3498db', '#e74c3c'])
    plt.title(f'{model_name} - RMSE')
    plt.ylim(0, max(sub['RMSE']) * 1.2)

    # MAE
    plt.subplot(1, 3, 2)
    plt.bar(sub['target'], sub['MAE'], color=['#3498db', '#e74c3c'])
    plt.title(f'{model_name} - MAE')
    plt.ylim(0, max(sub['MAE']) * 1.2)

    # R2
    plt.subplot(1, 3, 3)
    plt.bar(sub['target'], sub['R2'], color=['#3498db', '#e74c3c'])
    plt.title(f'{model_name} - R²')
    plt.ylim(0, 1)

    plt.suptitle(f'{model_name} 模型：短期 t+1 vs 长期 t+5 预测性能对比')
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, f'{model_name}_t1_vs_t5.png'), dpi=150)
    plt.close()

print("\n✅ 所有对比图表已保存到文件夹：", result_dir)
# print("\n📌 核心结论：")
# print("1. 所有模型在 t+1 短期预测上都比 t+5 长期更准确")
# print("2. 预测周期越长（t+5），误差越大，R² 越低，预测难度越高")
# print("3. XGBoost 和 LSTM 明显优于线性回归，适合股票价格预测")