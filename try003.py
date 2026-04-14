# 导入核心库
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import VotingClassifier, VotingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             mean_absolute_error, mean_squared_error, r2_score,
                             confusion_matrix, classification_report)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from datetime import datetime, timedelta

# 设置全局风格
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 解决中文乱码
plt.rcParams["axes.unicode_minus"] = False

# 固定随机种子保证可复现
SEED = 42
np.random.seed(SEED)


# ====================== 修复1：加载本地数据 ======================
def load_local_data(stocks):
    """从本地CSV文件加载5只股票的数据"""
    data_dict = {}

    for stock in stocks:
        csv_file = f"{stock}_data.csv"
        try:
            df = pd.read_csv(csv_file)
            # 确保有正确的列名
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)

            # 标准化列名为小写
            column_mapping = {
                'Open': 'open', 'High': 'high', 'Low': 'low',
                'Close': 'close', 'Volume': 'volume'
            }
            df.rename(columns=column_mapping, inplace=True)

            # 添加股票标识
            df["stock"] = stock
            data_dict[stock] = df
            print(f"✓ 成功加载 {csv_file} ({len(df)}条记录)")
        except FileNotFoundError:
            print(f"✗ 未找到 {csv_file}，跳过该股票")

    if not data_dict:
        raise FileNotFoundError("未找到任何本地CSV文件！请确保 AAPL_data.csv, GOOGL_data.csv 等文件存在。")

    # 合并为一个DataFrame
    full_data = pd.concat(data_dict.values(), axis=0)
    full_data = full_data.dropna()
    return full_data


# ====================== 核心参数配置 ======================
STOCKS = ["AAPL", "GOOGL", "MSFT", "NFLX", "NVDA"]  # 5只股票
START_DATE = "2013-02-08"
END_DATE = "2018-02-07"
SHORT_TERM_WINDOW = 5  # 短期趋势窗口
LONG_TERM_WINDOW = 20  # 长期趋势窗口

# 执行加载本地数据
print("正在加载本地CSV数据...")
raw_data = load_local_data(STOCKS)
print(f"\n数据加载完成！共{len(raw_data)}条记录")
print(raw_data.head())


# ====================== 特征工程（核心模块，保持不变） ======================
def build_features(df, target_col="close"):
    """构造金融特征（分类+回归共用）"""
    df = df.copy()

    # 1. 基础价格特征
    df["high_low_diff"] = df["high"] - df["low"]
    df["close_open_diff"] = df["close"] - df["open"]

    # 2. 技术指标（均线、波动率）
    for window in [SHORT_TERM_WINDOW, LONG_TERM_WINDOW]:
        df[f"ma_{window}"] = df[target_col].rolling(window=window).mean()
        df[f"std_{window}"] = df[target_col].rolling(window=window).std()

    # 3. 成交量特征
    df["volume_pct_change"] = df["volume"].pct_change()

    # 4. 差分特征（价格变化）
    df["close_diff_1"] = df[target_col].diff(1)
    df["close_diff_5"] = df[target_col].diff(5)

    # 5. 滞后特征（过去N天数据）
    for lag in [1, 3, 5]:
        df[f"close_lag_{lag}"] = df[target_col].shift(lag)
        df[f"volume_lag_{lag}"] = df["volume"].shift(lag)

    # 6. 标签构造
    # 分类任务标签：明日涨跌（1=涨，0=跌）
    df["label_classification"] = np.where(df[target_col].shift(-1) > df[target_col], 1, 0)
    # 回归任务标签：明日收盘价
    df["label_regression"] = df[target_col].shift(-1)

    # 删除含缺失值的行
    df = df.dropna()
    return df


# 按股票分组构造特征（避免跨股票信息泄露）
print("正在构造特征...")
feature_data = pd.DataFrame()
for stock in STOCKS:
    stock_df = raw_data[raw_data["stock"] == stock].copy()
    stock_df = build_features(stock_df)
    feature_data = pd.concat([feature_data, stock_df], axis=0)

print(f"特征构造完成！共{len(feature_data)}条记录")
print(feature_data[["stock", "close", "label_classification", "label_regression"]].head())

# 不再单独保存特征数据文件，将在最后统一保存到Excel中

# ====================== 主循环：处理所有5支股票 ======================
# ====================== 初始化数据收集容器 ======================
clf_confusion_data = {}  # 存储所有股票的分类混淆矩阵数据
reg_prediction_data = {}  # 存储所有股票的回归预测数据
all_clf_results = []
all_reg_results = []
all_term_comparisons = []

for stock in STOCKS:
    print(f"\n{'=' * 80}")
    print(f"正在处理股票: {stock}")
    print(f"{'=' * 80}")

    # 检查是否有该股票的数据
    if stock not in feature_data["stock"].values:
        print(f"⚠ 跳过 {stock}：无可用数据")
        continue

    stock_data = feature_data[feature_data["stock"] == stock].copy()

    if len(stock_data) < 100:
        print(f"⚠ 跳过 {stock}：数据量不足 ({len(stock_data)}条)")
        continue

    print(f"数据量: {len(stock_data)}条")


    # ====================== 任务1：股价趋势预测（分类） ======================
    def prepare_classification_data(df, feature_cols, target_col="label_classification"):
        """准备分类任务的训练/测试集（时序划分）"""
        X = df[feature_cols].values
        y = df[target_col].values

        # 时序划分：前80%训练，后20%测试（不打乱）
        train_size = int(0.8 * len(X))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        # 标准化（KNN需要）
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        return X_train_scaled, X_test_scaled, y_train, y_test, scaler


    # 定义特征列（排除标签、非特征列和非数值列）
    exclude_cols = ["stock", "label_classification", "label_regression"]
    # 只选择数值型特征列
    feature_cols = [col for col in feature_data.columns
                    if col not in exclude_cols and feature_data[col].dtype in ['float64', 'int64', 'float32', 'int32']]

    print(f"  特征列数量: {len(feature_cols)}")
    print(f"  特征列: {feature_cols[:5]}...")  # 显示前5个

    # 准备数据
    X_train_clf, X_test_clf, y_train_clf, y_test_clf, scaler_clf = prepare_classification_data(
        stock_data, feature_cols
    )

    print(f"训练集: {len(X_train_clf)}条, 测试集: {len(X_test_clf)}条")

    # 基线模型 + 改进模型
    # 1. 基线模型1：决策树
    dt_clf = DecisionTreeClassifier(random_state=SEED, max_depth=5)
    dt_clf.fit(X_train_clf, y_train_clf)
    y_pred_dt = dt_clf.predict(X_test_clf)

    # 2. 基线模型2：KNN
    knn_clf = KNeighborsClassifier(n_neighbors=5)
    knn_clf.fit(X_train_clf, y_train_clf)
    y_pred_knn = knn_clf.predict(X_test_clf)

    # 3. 改进1：模型融合（软投票）
    voting_clf = VotingClassifier(
        estimators=[("dt", dt_clf), ("knn", knn_clf)],
        voting="soft",  # 软投票：用概率加权
        weights=[0.6, 0.4]  # 决策树权重0.6，KNN权重0.4
    )
    voting_clf.fit(X_train_clf, y_train_clf)
    y_pred_voting = voting_clf.predict(X_test_clf)


    # 分类结果评估与可视化
    def evaluate_classification(y_true, y_pred, model_name):
        """评估分类模型并返回指标"""
        acc = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        print(f"  [{model_name}] 准确率: {acc:.4f} | 精确率: {precision:.4f} | 召回率: {recall:.4f} | F1: {f1:.4f}")
        return {"model": model_name, "accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


    # 评估所有模型
    clf_results = []
    clf_results.append(evaluate_classification(y_test_clf, y_pred_dt, "决策树（基线）"))
    clf_results.append(evaluate_classification(y_test_clf, y_pred_knn, "KNN（基线）"))
    clf_results.append(evaluate_classification(y_test_clf, y_pred_voting, "决策树+KNN融合（改进1）"))

    # 添加到总结果
    for result in clf_results:
        result["stock"] = stock
    all_clf_results.extend(clf_results)

    # 可视化：混淆矩阵（融合模型）- 收集数据用于后续统一绘图
    cm = confusion_matrix(y_test_clf, y_pred_voting)
    clf_confusion_data[stock] = {
        'cm': cm,
        'y_true': y_test_clf,
        'y_pred': y_pred_voting
    }


    # ====================== 任务2：股价数值预测（回归） ======================
    def prepare_regression_data(df, feature_cols, target_col="label_regression"):
        """准备回归任务的训练/测试集（时序划分）"""
        X = df[feature_cols].values
        y = df[target_col].values

        # 时序划分
        train_size = int(0.8 * len(X))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        # 标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        return X_train_scaled, X_test_scaled, y_train, y_test, scaler


    # 准备回归数据
    X_train_reg, X_test_reg, y_train_reg, y_test_reg, scaler_reg = prepare_regression_data(
        stock_data, feature_cols
    )

    # 基线模型 + 改进模型
    # 1. 基线模型1：决策树回归
    dt_reg = DecisionTreeRegressor(random_state=SEED, max_depth=5)
    dt_reg.fit(X_train_reg, y_train_reg)
    y_pred_dt_reg = dt_reg.predict(X_test_reg)

    # 2. 基线模型2：KNN回归
    knn_reg = KNeighborsRegressor(n_neighbors=5)
    knn_reg.fit(X_train_reg, y_train_reg)
    y_pred_knn_reg = knn_reg.predict(X_test_reg)

    # 3. 改进1：模型融合（加权平均）
    voting_reg = VotingRegressor(
        estimators=[("dt", dt_reg), ("knn", knn_reg)],
        weights=[0.6, 0.4]
    )
    voting_reg.fit(X_train_reg, y_train_reg)
    y_pred_voting_reg = voting_reg.predict(X_test_reg)


    # 回归结果评估与可视化
    def evaluate_regression(y_true, y_pred, model_name):
        """评估回归模型并返回指标"""
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        print(f"  [{model_name}] MAE: {mae:.4f} | MSE: {mse:.4f} | RMSE: {rmse:.4f} | R²: {r2:.4f}")
        return {"model": model_name, "mae": mae, "mse": mse, "rmse": rmse, "r2": r2}


    # 评估所有回归模型
    reg_results = []
    reg_results.append(evaluate_regression(y_test_reg, y_pred_dt_reg, "决策树回归（基线）"))
    reg_results.append(evaluate_regression(y_test_reg, y_pred_knn_reg, "KNN回归（基线）"))
    reg_results.append(evaluate_regression(y_test_reg, y_pred_voting_reg, "决策树+KNN融合（改进1）"))

    # 添加到总结果
    for result in reg_results:
        result["stock"] = stock
    all_reg_results.extend(reg_results)

    # 可视化：预测值vs真实值（融合模型）- 收集数据用于后续统一绘图
    test_dates = stock_data.index[-len(y_test_reg):]  # 测试集日期
    reg_prediction_data[stock] = {
        'dates': test_dates,
        'y_true': y_test_reg,
        'y_pred': y_pred_voting_reg
    }


    # ====================== 改进2：优化训练方法 ======================
    def check_supports_sample_weight(estimator):
        """检查模型是否支持 sample_weight 参数"""
        try:
            import inspect
            sig = inspect.signature(estimator.fit)
            return "sample_weight" in sig.parameters
        except:
            return False


    def optimized_training_with_tscv(X, y, model, model_name, sample_weights=None):
        """
        用时序交叉验证优化训练
        智能处理 sample_weight：仅当模型支持时才传入
        """
        tscv = TimeSeriesSplit(n_splits=5)  # 5折时序交叉验证
        cv_scores = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]

            # 核心修复：判断是否支持 sample_weight
            if sample_weights is not None and check_supports_sample_weight(model):
                weights_fold = sample_weights[train_idx]
                model.fit(X_train_fold, y_train_fold, sample_weight=weights_fold)
            else:
                model.fit(X_train_fold, y_train_fold)

            # 验证
            y_pred_val = model.predict(X_val_fold)
            # 分类用F1，回归用RMSE
            if "分类" in model_name:
                score = f1_score(y_val_fold, y_pred_val, zero_division=0)
            else:
                score = np.sqrt(mean_squared_error(y_val_fold, y_pred_val))
            cv_scores.append(score)

        mean_score = np.mean(cv_scores)
        print(f"  [{model_name}] 时序CV平均得分: {mean_score:.4f}")
        return model, mean_score


    # 构造样本权重（近期数据更高权重）
    def build_sample_weights(y_train, decay_factor=0.99):
        """构造指数衰减样本权重"""
        n_samples = len(y_train)
        weights = np.array([decay_factor ** (n_samples - i - 1) for i in range(n_samples)])
        weights = weights / weights.sum() * n_samples  # 归一化
        return weights


    # 改进2 实验开始
    sample_weights_clf = build_sample_weights(y_train_clf)

    # 1. 融合模型：使用时序CV（避开样本加权，因为含KNN）
    optimized_voting_clf, _ = optimized_training_with_tscv(
        X_train_clf, y_train_clf, voting_clf, "融合模型（时序CV）分类"
    )

    # 2. 单一决策树：展示时序CV + 样本加权的完整效果
    optimized_dt_clf, _ = optimized_training_with_tscv(
        X_train_clf, y_train_clf, dt_clf, "决策树（时序CV+样本加权）分类", sample_weights_clf
    )

    # 回归任务同理
    sample_weights_reg = build_sample_weights(y_train_reg)

    # 1. 融合模型：使用时序CV
    optimized_voting_reg, _ = optimized_training_with_tscv(
        X_train_reg, y_train_reg, voting_reg, "融合模型（时序CV）回归"
    )

    # 2. 单一决策树：展示时序CV + 样本加权
    optimized_dt_reg, _ = optimized_training_with_tscv(
        X_train_reg, y_train_reg, dt_reg, "决策树（时序CV+样本加权）回归", sample_weights_reg
    )

    # ====================== 结果汇总与保存 ======================
    # 分类结果汇总
    clf_results_df = pd.DataFrame(clf_results)
    clf_results_df["stock"] = stock

    # 回归结果汇总
    reg_results_df = pd.DataFrame(reg_results)
    reg_results_df["stock"] = stock

    # 不再单独保存每个股票的结果文件和模型文件，统一在最后处理


    # ====================== 长短周期对比（验证研究假设） ======================
    def compare_short_long_term(df, feature_cols, short_window=5, long_window=20):
        """对比短期和长期趋势预测效果"""
        results = []

        # 短期：只用短期特征（ma_5, std_5, close_lag_5等）
        short_feature_cols = [col for col in feature_cols if "20" not in col]
        X_short, y_short = df[short_feature_cols].values, df["label_classification"].values
        train_size = int(0.8 * len(X_short))
        X_train_short, X_test_short = X_short[:train_size], X_short[train_size:]
        y_train_short, y_test_short = y_short[:train_size], y_short[train_size:]
        scaler = StandardScaler()
        X_train_short_scaled = scaler.fit_transform(X_train_short)
        X_test_short_scaled = scaler.transform(X_test_short)

        # 短期模型训练
        short_model = DecisionTreeClassifier(random_state=SEED, max_depth=5)
        short_model.fit(X_train_short_scaled, y_train_short)
        y_pred_short = short_model.predict(X_test_short_scaled)
        short_f1 = f1_score(y_test_short, y_pred_short, zero_division=0)
        results.append({"term": "短期", "f1_score": short_f1, "model": "决策树"})

        # 长期：用所有特征（包括ma_20, std_20）
        X_long, y_long = df[feature_cols].values, df["label_classification"].values
        X_train_long, X_test_long = X_long[:train_size], X_long[train_size:]
        y_train_long, y_test_long = y_long[:train_size], y_long[train_size:]
        X_train_long_scaled = scaler.fit_transform(X_train_long)
        X_test_long_scaled = scaler.transform(X_test_long)

        # 长期模型训练
        long_model = DecisionTreeClassifier(random_state=SEED, max_depth=5)
        long_model.fit(X_train_long_scaled, y_train_long)
        y_pred_long = long_model.predict(X_test_long_scaled)
        long_f1 = f1_score(y_test_long, y_pred_long, zero_division=0)
        results.append({"term": "长期", "f1_score": long_f1, "model": "决策树"})

        return pd.DataFrame(results)


    # 长短周期对比数据收集 - 用于后续统一绘图
    term_comparison_df = compare_short_long_term(stock_data, feature_cols)
    term_comparison_df["stock"] = stock
    all_term_comparisons.append(term_comparison_df)

    print(
        f"  短期F1: {term_comparison_df.iloc[0]['f1_score']:.4f}, 长期F1: {term_comparison_df.iloc[1]['f1_score']:.4f}")
    print(f"{'=' * 80}\n")

# ====================== 最终汇总报告与统一可视化 ======================
print("\n" + "=" * 80)
print("所有股票处理完成！最终汇总报告")
print("=" * 80)

# 分类结果汇总
if all_clf_results:
    all_clf_df = pd.DataFrame(all_clf_results)
    print("\n【分类任务结果汇总】")
    print(all_clf_df.to_string(index=False))

# 回归结果汇总
if all_reg_results:
    all_reg_df = pd.DataFrame(all_reg_results)
    print("\n【回归任务结果汇总】")
    print(all_reg_df.to_string(index=False))

# 长短周期对比汇总
if all_term_comparisons:
    all_term_df = pd.concat(all_term_comparisons, ignore_index=True)
    print("\n【长短周期对比汇总】")
    print(all_term_df.to_string(index=False))

# ====================== 统一保存所有结果为单个CSV文件 ======================
# 将所有结果合并到一个CSV文件中，使用type列区分数据类型
all_data_rows = []

# 添加特征数据
if not feature_data.empty:
    feature_df = feature_data.reset_index()
    feature_df['data_type'] = 'feature_data'
    all_data_rows.append(feature_df)

# 添加分类结果
if all_clf_results:
    clf_df = pd.DataFrame(all_clf_results)
    clf_df['data_type'] = 'classification_results'
    all_data_rows.append(clf_df)

# 添加回归结果
if all_reg_results:
    reg_df = pd.DataFrame(all_reg_results)
    reg_df['data_type'] = 'regression_results'
    all_data_rows.append(reg_df)

# 添加长短周期对比
if all_term_comparisons:
    term_df = pd.concat(all_term_comparisons, ignore_index=True)
    term_df['data_type'] = 'term_comparison'
    all_data_rows.append(term_df)

# 合并所有数据并保存
if all_data_rows:
    combined_df = pd.concat(all_data_rows, ignore_index=True, sort=False)
    combined_df.to_csv('all_stocks_results.csv', index=False)
    print("\n✓ 所有结果已保存至: all_stocks_results.csv")
    print(f"  总记录数: {len(combined_df)}")
    print(f"  数据类型: {combined_df['data_type'].unique()}")

# ====================== 统一生成5个可视化图片 ======================

# 图1: 所有股票的分类混淆矩阵合集
if clf_confusion_data:
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.ravel()
    
    for idx, (stock, data) in enumerate(clf_confusion_data.items()):
        if idx < 6:  # 最多显示6个子图
            cm = data['cm']
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                        xticklabels=["下跌", "上涨"], yticklabels=["下跌", "上涨"],
                        ax=axes[idx])
            axes[idx].set_title(f"{stock} 混淆矩阵", fontsize=12)
            axes[idx].set_xlabel("预测标签")
            axes[idx].set_ylabel("真实标签")
    
    # 隐藏多余的子图
    for j in range(idx+1, len(axes)):
        axes[j].set_visible(False)
        
    plt.suptitle('所有股票分类模型混淆矩阵合集', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('1_all_stocks_confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ 图1已保存: 1_all_stocks_confusion_matrices.png")

# 图2: 所有股票的回归预测对比
if reg_prediction_data:
    fig, axes = plt.subplots(len(reg_prediction_data), 1, figsize=(14, 4*len(reg_prediction_data)))
    if len(reg_prediction_data) == 1:
        axes = [axes]
    
    for idx, (stock, data) in enumerate(reg_prediction_data.items()):
        axes[idx].plot(data['dates'], data['y_true'], label="真实收盘价", color="blue", linewidth=2)
        axes[idx].plot(data['dates'], data['y_pred'], label="预测收盘价", color="red", linestyle="--", linewidth=2)
        axes[idx].set_title(f"{stock} 收盘价预测", fontsize=12)
        axes[idx].set_xlabel("日期")
        axes[idx].set_ylabel("收盘价（USD）")
        axes[idx].legend()
        axes[idx].tick_params(axis='x', rotation=45)
    
    plt.suptitle('所有股票回归模型预测结果合集', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('2_all_stocks_regression_predictions.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ 图2已保存: 2_all_stocks_regression_predictions.png")

# 图3: 长短周期对比柱状图
if all_term_comparisons:
    all_term_df = pd.concat(all_term_comparisons, ignore_index=True)
    plt.figure(figsize=(12, 6))
    sns.barplot(x="stock", y="f1_score", hue="term", data=all_term_df, palette="viridis")
    plt.title('所有股票短期vs长期趋势预测F1得分对比', fontsize=14, fontweight='bold')
    plt.xlabel("股票")
    plt.ylabel("F1得分")
    plt.ylim(0, 1)
    plt.legend(title="时间周期")
    plt.tight_layout()
    plt.savefig('3_all_stocks_term_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ 图3已保存: 3_all_stocks_term_comparison.png")

# 图4: 分类模型性能对比
if all_clf_results:
    all_clf_df = pd.DataFrame(all_clf_results)
    plt.figure(figsize=(14, 8))
    
    # 绘制准确率和F1分数
    metrics_to_plot = ['accuracy', 'f1']
    x = np.arange(len(STOCKS))
    width = 0.35
    
    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    # 获取每个股票的最佳模型结果
    best_models = all_clf_df.loc[all_clf_df.groupby(['stock', 'model'])['accuracy'].idxmax()]
    
    for i, metric in enumerate(metrics_to_plot):
        values = [best_models[best_models['stock'] == stock][metric].values[0] 
                 for stock in STOCKS if stock in best_models['stock'].values]
        stocks_present = [stock for stock in STOCKS if stock in best_models['stock'].values]
        
        bars = ax1.bar(x[:len(stocks_present)] + i*width, values, width, label=metric.upper())
        
        # 在柱状图上添加数值标签
        for bar, val in zip(bars, values):
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    
    ax1.set_xlabel('股票')
    ax1.set_ylabel('得分')
    ax1.set_title('所有股票分类模型性能对比', fontsize=14, fontweight='bold')
    ax1.set_xticks(x[:len(stocks_present)] + width/2)
    ax1.set_xticklabels(stocks_present)
    ax1.legend()
    ax1.set_ylim(0, 1.2)
    
    plt.tight_layout()
    plt.savefig('4_all_stocks_classification_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ 图4已保存: 4_all_stocks_classification_performance.png")

# 图5: 回归模型性能对比
if all_reg_results:
    all_reg_df = pd.DataFrame(all_reg_results)
    plt.figure(figsize=(14, 8))
    
    # 绘制R²分数和RMSE
    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    # 获取每个股票的最佳模型结果
    best_models = all_reg_df.loc[all_reg_df.groupby(['stock', 'model'])['r2'].idxmax()]
    
    stocks_present = [stock for stock in STOCKS if stock in best_models['stock'].values]
    x = np.arange(len(stocks_present))
    width = 0.35
    
    r2_values = [best_models[best_models['stock'] == stock]['r2'].values[0] 
                for stock in stocks_present]
    rmse_values = [best_models[best_models['stock'] == stock]['rmse'].values[0] 
                  for stock in stocks_present]
    
    bars1 = ax1.bar(x - width/2, r2_values, width, label='R2 Score', color='skyblue')
    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + width/2, rmse_values, width, label='RMSE', color='salmon')
    
    # 添加数值标签
    for bar, val in zip(bars1, r2_values):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    
    for bar, val in zip(bars2, rmse_values):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{val:.2f}', ha='center', va='bottom', fontsize=8)
    
    ax1.set_xlabel('股票')
    ax1.set_ylabel('R2 Score', color='skyblue')
    ax2.set_ylabel('RMSE', color='salmon')
    ax1.set_title('所有股票回归模型性能对比', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(stocks_present)
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.tight_layout()
    plt.savefig('5_all_stocks_regression_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ 图5已保存: 5_all_stocks_regression_performance.png")

print("\n" + "=" * 80)
print("✅ 所有结果已整理完成！")
print("📊 输出文件清单:")
print("   📁 all_stocks_results.csv - 包含所有数据的CSV文件（使用data_type列区分）")
print("   🖼️  1_all_stocks_confusion_matrices.png - 分类混淆矩阵合集")
print("   🖼️  2_all_stocks_regression_predictions.png - 回归预测结果合集")
print("   🖼️  3_all_stocks_term_comparison.png - 长短周期对比")
print("   🖼️  4_all_stocks_classification_performance.png - 分类性能对比")
print("   🖼️  5_all_stocks_regression_performance.png - 回归性能对比")
print("   总计: 6个文件 (1个CSV + 5个图片)")
print("=" * 80)