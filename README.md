# COMP5564 - Research on Stock Price Prediction Based on Machine Learning

## Project Overview

This project focuses on predicting stock prices using various machine learning techniques. It analyzes historical stock data from multiple companies and implements both classification and regression models to forecast stock price movements and values.

## Repository Structure

```
COMP5564-Research-on-Stock-Price-Prediction-Based-on-Machine-Learning/
├── try003.py                          # Main Python script for stock price prediction
├── feature_data.csv                   # Extracted features dataset
├── all_stocks_results.csv             # Comprehensive results from all stocks analysis
├── AAPL_data.csv                      # Apple Inc. stock data
├── GOOGL_data.csv                     # Alphabet Inc. (Google) stock data
├── MSFT_data.csv                      # Microsoft Corporation stock data
├── NFLX_data.csv                      # Netflix Inc. stock data
├── NVDA_data.csv                      # NVIDIA Corporation stock data
├── archive/                           # Historical data archive
│   ├── individual_stocks_5yr/         # 5-year individual stock data for S&P 500 companies
│   │   └── individual_stocks_5yr/     # CSV files for each stock ticker
│   ├── all_stocks_5yr.csv            # Combined 5-year data for all stocks
│   ├── getSandP.py                    # Script to fetch S&P 500 data
│   └── merge.sh                       # Shell script to merge stock data
├── *.png                              # Visualization outputs
│   ├── 1_all_stocks_confusion_matrices.png
│   ├── 2_all_stocks_regression_predictions.png
│   ├── 3_all_stocks_term_comparison.png
│   ├── 4_all_stocks_classification_performance.png
│   └── 5_all_stocks_regression_performance.png
└── 6_result.txt                       # Text summary of results
```

## Key Components

### Main Script
- **try003.py**: The core Python script that implements machine learning models for stock price prediction, including data preprocessing, feature engineering, model training, and evaluation.

### Data Files
- **Individual Stock Data**: CSV files containing historical price data for major tech stocks (AAPL, GOOGL, MSFT, NFLX, NVDA)
- **Feature Dataset**: Preprocessed features extracted from raw stock data for model training
- **Results**: Comprehensive analysis results stored in CSV format

### Archive
- Contains 5-year historical data for S&P 500 companies
- Includes scripts for data acquisition and preprocessing
- Provides a comprehensive dataset for robust model training

### Visualizations
The project generates several visualization outputs:
1. Confusion matrices for classification performance
2. Regression prediction plots
3. Term comparison charts
4. Classification performance metrics
5. Regression performance metrics

## Features

- Multi-stock analysis across S&P 500 companies
- Both classification (direction prediction) and regression (price prediction) approaches
- Comprehensive feature engineering from historical stock data
- Performance evaluation with multiple metrics
- Visual representation of model performance

## Requirements

To run this project, you'll need:
- Python 3.x
- Common data science libraries (pandas, numpy, scikit-learn, matplotlib, etc.)

## Usage

1. Ensure all required Python packages are installed
2. Run the main script:
   ```bash
   python try003.py
   ```
3. Review the generated results and visualizations

## Results

The project produces:
- Prediction accuracy metrics for classification tasks
- Error metrics (MSE, RMSE, MAE) for regression tasks
- Visual comparisons of predicted vs actual values
- Performance analysis across different stocks and time periods

## Author

This project was developed as part of COMP5564 coursework focusing on machine learning applications in financial markets.

## License

This project is for educational and research purposes.
