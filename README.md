# SP_2025_ML_Predictor
Predicting stocks' ranks by expected performance in 2025 via scikit-learn's random forest classifier 

## Data
The data used in this project is from two csv files: 
1. sp500_index.csv: the index against which the performance of each stock is measured
2. sp500_stocks.csv: the historical performance of 500+ stocks spanning 14 years

source: https://www.kaggle.com/datasets/andrewmvd/sp-500-stocks?select=sp500_stocks.csv

## Features
Right now, only the past five years of data (2020-2025) are used for training. Each year is considered as an earnings season. Later improvements will include using the entire datasets for training and considering an yearly quarter as one season. 

Stock and market returns are calculated before filtering and splitting into five seasonal dataframes. This ensures that the correct final closing values from the last day of one season are used when calculating returns for the first day of the next season. 

For each stock in a season, 50-day and 200-day moving averages are calculated. From these, the moving average spread is derived. The “Golden Cross” principle is applied: if the short-term moving average is greater than the long-term moving average, the stock is considered bullish; otherwise bearish. To make the model less sensitive to noisy fluctuations in the spread, a binary indicator is also included so the model can decide which signal is more useful. For each stock, moving averages across the season can now be claculated. 

This leads to seven performance statistics for each stock:
1. Average Volume - indicates how actively the stock traded during the season
2. Moving Average Mean Spread - captures the stock’s overall trend across the season, less affected by short-term fluctuations
3. Moving Average Binary - binary bullish/bearish trend indicator
4. Volatility - measures fluctuations in the stock price
5. Beta - captures this volatility w.r.t to the overall market
6. Annual Return - % return over the season
7. Relative Annual - stock return relative to the market return

The ranks of stocks in each season are calculated based on annual return (descending), volatility (ascending), beta (ascending), and moving average spread (descending).

## Process
For training data, each stock's performance statistics from season 'n' are used to predict its rank in season 'n + 1'. Stocks missing in a previous year are assigned default feature values representing the average of the bottom 3 stocks from the previous season.

Moreover, as the 2025 season hasn't concluded yet, features for the 2024 season are used as the most recent set of training features.

A scikit-learn pipeline is then created that scales features and trains a Random Forest. The RandomForestCLassifier outputs a probability distribution over possible ranks (1-50). The expected rank of each stock is then obatined by multiplying each probability with its corresponding rank. Finally, the stocks are sorted based on expected ranks (ascending) and are assigned integer ranks.

## Results
Predicted Stock Ranking 2025 table (1 = best):
1. NVDA (expected rank 53.23)
2. TSLA (expected rank 65.22)
3. SO (expected rank 73.33)
4. CME (expected rank 73.60)
5. LRCX (expected rank 73.70)
6. MO (expected rank 75.03)
7. PPL (expected rank 76.43)
8. WEC (expected rank 76.44)
9. TSN (expected rank 76.74)
10. ANET (expected rank 76.96)
.....
40. NDAQ (expected rank 84.55)
41. UDR (expected rank 84.59)
42. EIX (expected rank 84.64)
43. IR (expected rank 84.70)
44. DOV (expected rank 84.95)
45. CRM (expected rank 84.96)
46. HLT (expected rank 85.01)
47. KMX (expected rank 85.09)
48. TDG (expected rank 85.17)
49. IQV (expected rank 85.19)
50. PNW (expected rank 85.20)
