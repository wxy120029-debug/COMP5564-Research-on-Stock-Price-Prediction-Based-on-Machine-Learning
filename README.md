# COMP5564 ML in Finance Data Project 

## Research Report on Stock Price Prediction Based on Machine Learning

### Project Overview

This project conducts research on stock price prediction using machine learning techniques, employing various machine learning models (LSTM, Linear Regression, XGBoost) to perform classification and regression analysis on stock data across different time windows (t1 and t5).

### Project Structure

```
My5564/
├── .idea/                    # IDE configuration files
├── archive/                  # Historical data archive
│   ├── individual_stocks_5yr/    # Individual stock 5-year data
│   ├── all_stocks_5yr.csv        # All stocks 5-year summary data
│   ├── getSandP.py              # Script to fetch S&P 500 data
│   └── merge.sh                 # Data merging script
├── models/                   # Directory for storing trained models
├── result/                   # Experimental results and visualization charts
│   ├── LSTM_t1_vs_t5.png         # LSTM model t1 vs t5 comparison chart
│   ├── Linear_t1_vs_t5.png       # Linear Regression model t1 vs t5 comparison chart
│   ├── XGBoost_t1_vs_t5.png      # XGBoost model t1 vs t5 comparison chart
│   ├── accuracy_compare.png      # Accuracy comparison chart
│   ├── auc_roc_compare.png       # AUC-ROC comparison chart
│   ├── f1_compare.png            # F1 score comparison chart
│   ├── mae_compare.png           # MAE comparison chart
│   ├── model_performance.csv     # Model performance metrics summary
│   ├── r2_compare.png            # R² comparison chart
│   └── rmse_compare.png          # RMSE comparison chart
├── split_data/               # Split training/testing data
│   ├── AAPL_test.csv             # Apple test data
│   ├── AAPL_train.csv            # Apple training data
│   ├── GOOGL_test.csv            # Google test data
│   ├── GOOGL_train.csv           # Google training data
│   ├── MSFT_test.csv             # Microsoft test data
│   ├── MSFT_train.csv            # Microsoft training data
│   ├── NFLX_test.csv             # Netflix test data
│   ├── NFLX_train.csv            # Netflix training data
│   ├── NVDA_test.csv             # NVIDIA test data
│   ├── NVDA_train.csv            # NVIDIA training data
│   ├── test_featured.csv         # Test feature data
│   ├── test_featured_reg.csv     # Test regression feature data
│   ├── train_featured.csv        # Training feature data
│   └── train_featured_reg.csv    # Training regression feature data
├── .gitignore                # Git ignore file configuration
├── 1.COMP5564_Classification_t1_t5.py  # Classification task main program (t1 vs t5)
├── 2.COMP5564_Regression_t1_t5.py      # Regression task main program (t1 vs t5)
├── AAPL_data.csv               # Apple raw data
├── GOOGL_data.csv              # Google raw data
├── MSFT_data.csv               # Microsoft raw data
├── NFLX_data.csv               # Netflix raw data
└── NVDA_data.csv               # NVIDIA raw data
```

### Main File Descriptions

#### Core Code Files
- `1.COMP5564_Classification_t1_t5.py`: Implements classification prediction of stock price movements, comparing model performance across different time windows (t1, t5)
- `2.COMP5564_Regression_t1_t5.py`: Implements regression prediction of stock prices, comparing model performance across different time windows (t1, t5)

#### Data Files
- `*_data.csv`: Raw historical data files for each stock
- `split_data/`: Training/testing datasets after preprocessing and feature engineering

#### Result Files
- `result/`: Contains visualization charts of various evaluation metrics and performance summary tables
- `models/`: Stores trained machine learning models

### Technology Stack

- Python programming language
- Machine Learning Libraries: scikit-learn, XGBoost, TensorFlow/Keras (LSTM)
- Data Processing: pandas, numpy
- Visualization: matplotlib, seaborn

### Research Content

This project investigates the following aspects:
1. Performance of different machine learning models in stock price prediction
2. Comparison of short-term (t1) vs medium-term (t5) prediction windows
3. Performance differences between classification tasks (price movement prediction) and regression tasks (price prediction)
4. Prediction effectiveness analysis across multiple tech stocks (AAPL, GOOGL, MSFT, NFLX, NVDA)

### How to Run

1. Ensure required dependencies are installed
2. Run classification task: `python 1.COMP5564_Classification_t1_t5.py`
3. Run regression task: `python 2.COMP5564_Regression_t1_t5.py`

### Notes

- The project requires sufficient memory to process large-scale stock data
- Deep learning models (LSTM) may require GPU acceleration for training
- Results may vary slightly due to random seeds and data splits