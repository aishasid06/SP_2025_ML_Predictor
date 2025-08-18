import os
from typing import List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def summarise_season(seasonal_df):
    annual_index_return = seasonal_df.drop_duplicates('Date')
    annual_index_return = annual_index_return['S&P500'].iloc[-1] / annual_index_return['S&P500'].iloc[0] - 1

    seasonal_df['MA 50']  = seasonal_df.groupby('Symbol')['Adj Close'].transform(lambda s: s.rolling(50,  min_periods=1).mean())
    seasonal_df['MA 200'] = seasonal_df.groupby('Symbol')['Adj Close'].transform(lambda s: s.rolling(200, min_periods=1).mean())
    
    #take average of both and then compare and stuff 
    seasonal_df['MA Spread'] = seasonal_df['MA 50'] - seasonal_df['MA 200']
    seasonal_df['MA Binary'] = (seasonal_df['MA 50'] > seasonal_df['MA 200']).astype(int)

    stock_factors = seasonal_df.groupby('Symbol').agg(
        first_price = ('Adj Close', 'first'),
        last_price = ('Adj Close', 'last'),
        avg_volume = ('Volume', 'mean'),
        ma_avg_50 = ('MA 50', 'mean'),
        ma_avg_200 = ('MA 200', 'mean')
    )

    stock_factors = stock_factors.rename(columns={
    'first_price': 'Start Price',
    'last_price': 'End Price',
    'avg_volume': 'Avg Vol',
    'ma_avg_50': 'MAM 50',
    'ma_avg_200': 'MAM 200'
    })

    stock_factors['MAM Spread'] = stock_factors['MAM 50'] - stock_factors['MAM 200']
    stock_factors['MAM Binary'] = (stock_factors['MAM 50'] > stock_factors['MAM 200']).astype(int)

    vols = seasonal_df.groupby('Symbol')['Stock Return'].std().rename('Volatility')

    def stock_beta(s):
        s = s[['Stock Return','Market Return']].dropna()
        if len(s) < 2:
            return np.nan
        var_m = s['Market Return'].var()
        if not var_m or np.isnan(var_m):
            return np.nan
        cov_sm = s[['Stock Return','Market Return']].cov().iloc[0,1]
        return cov_sm / var_m

    betas = seasonal_df.groupby('Symbol').apply(stock_beta).rename('Beta')

    summarised_seasonal_df = pd.concat([stock_factors, vols, betas], axis=1).reset_index()
    summarised_seasonal_df['Annual Return'] = summarised_seasonal_df['End Price'] / summarised_seasonal_df['Start Price'] - 1
    summarised_seasonal_df['Relative Annual Return'] = summarised_seasonal_df['Annual Return'] - annual_index_return
    summarised_seasonal_df = summarised_seasonal_df[['Symbol', 'Avg Vol', 'MAM Spread', 'MAM Binary', 'Volatility', 'Beta', 'Annual Return', 'Relative Annual Return']]
    # summarised_seasonal_df.sort_values(['Symbol'], ascending=True)
    # print(summarised_seasonal_df['Symbol'].to_string(index=False))
    summarised_seasonal_df = summarised_seasonal_df.sort_values(
        ['Annual Return', 'Volatility', 'Beta', 'MAM Spread'], ascending=[False, True, True, False]
    ).reset_index(drop=True)
    summarised_seasonal_df['Rank'] = summarised_seasonal_df.index + 1
    return summarised_seasonal_df


def prepare_seasonal_data(sp500_files: List[str]):
    companies_df = pd.read_csv(sp500_files[0])
    index_df = pd.read_csv(sp500_files[1])
    stocks_df = pd.read_csv(sp500_files[2])

    stocks_df = stocks_df.dropna(how='any',axis=0)
    stocks_df['Date'] = pd.to_datetime(stocks_df['Date'])
    stocks_df.sort_values(['Symbol', 'Date'], inplace=True)
    stocks_df['Stock Return'] = stocks_df.groupby('Symbol')['Adj Close'].pct_change()
    stocks_df = stocks_df[(stocks_df['Date'] >= "2020-01-01") & (stocks_df['Date'] <= "2024-12-20")]
    stocks_df.reset_index(drop=True, inplace=True)

    index_df['Date'] = pd.to_datetime(index_df['Date'])
    index_df['Market Return'] = index_df['S&P500'].pct_change()
    index_df = index_df[(index_df['Date'] >= "2020-01-01") & (index_df['Date'] <= "2024-12-20")]
    index_df.reset_index(drop=True, inplace=True)

    stocks_df = pd.merge(stocks_df, index_df, how='inner', on='Date')

    seasons = {year: data for year, data in stocks_df.groupby(stocks_df['Date'].dt.year)}

    for season in seasons.keys():
        seasons[season].reset_index(drop=True, inplace=True)
        seasons[season] = summarise_season(seasons[season])
    
    return seasons


def prepare_training_data(season_files: List[str]):
    season_summaries = prepare_seasonal_data(season_files)

    feature_rows = []
    target_rows = []
    seasons = list(season_summaries.keys())
    for i in range(len(seasons) - 1):
        prev_summary = season_summaries[seasons[i]].copy().set_index('Symbol')
        curr_summary = season_summaries[seasons[i + 1]].copy().set_index('Symbol')

        bottom_three = prev_summary.sort_values(['Annual Return', 'Volatility', 'Beta', 'MAM Spread'], ascending=[True, True, True, True]).head(3)
        default_features = bottom_three.mean().to_dict()

        for symbol, row in curr_summary.iterrows():
            if symbol in prev_summary.index:
                feats = prev_summary.loc[symbol][['Avg Vol', 'MAM Spread', 'MAM Binary', 'Volatility', 'Beta', 'Annual Return', 'Relative Annual Return']].to_dict()
            else:
                feats = {k: default_features[k] for k in ['Avg Vol', 'MAM Spread', 'MAM Binary', 'Volatility', 'Beta', 'Annual Return', 'Relative Annual Return']}
            feature_rows.append(feats)
            target_rows.append(row['Rank'])
    
    X_train = pd.DataFrame(feature_rows)
    y_train = pd.Series(target_rows)

    last_summary = season_summaries[seasons[-1]].copy().set_index('Symbol')

    bottom_three_last = last_summary.sort_values(['Annual Return', 'Volatility', 'Beta', 'MAM Spread'], ascending=[True, True, True, True]).head(3)
    default_features_last = bottom_three_last.mean().to_dict()

    latest_features_rows = []
    latest_symbols = last_summary.index.tolist()
    promoted = []   # maybe add stocks that were made available on the market in 2025

    for symbol in latest_symbols:
        feats = last_summary.loc[symbol][['Avg Vol', 'MAM Spread', 'MAM Binary', 'Volatility', 'Beta', 'Annual Return', 'Relative Annual Return']].to_dict()
        latest_features_rows.append((symbol, feats))

    for symbol in promoted:
        if symbol not in latest_symbols:
            feats = {k: default_features_last[k] for k in ['Avg Vol', 'MAM Spread', 'MAM Binary', 'Volatility', 'Beta', 'Annual Return', 'Relative Annual Return']}
        latest_features_rows.append((symbol, feats))
    latest_features_df = pd.DataFrame([feats for _, feats in latest_features_rows],
    index=[t for t, _ in latest_features_rows])

    return X_train, y_train, latest_features_df


def build_and_train_model(X: pd.DataFrame, y: pd.Series) -> Pipeline:
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            random_state=42,
            class_weight="balanced"
        ))
    ])
    model.fit(X, y)
    return model


def predict_stock_rank(model: Pipeline, features: pd.DataFrame) -> pd.DataFrame:
    probas = model.predict_proba(features)
    classes = model.named_steps["rf"].classes_
    exp_ranks = probas.dot(classes)
    prediction_df = pd.DataFrame({
        'Symbol': features.index,
        'Expected Rank': exp_ranks
    })

    prediction_df = prediction_df.sort_values('Expected Rank').reset_index(drop=True)
    prediction_df['Predicted Rank'] = prediction_df.index + 1

    return prediction_df[['Predicted Rank', 'Symbol', 'Expected Rank']]


def main():
    sp500_files = [
        os.path.join(os.path.dirname(__file__), "sp500_companies.csv"),
        os.path.join(os.path.dirname(__file__), "sp500_index.csv"),
        os.path.join(os.path.dirname(__file__), "sp500_stocks.csv")
    ]

    X_train, y_train, latest_features = prepare_training_data(sp500_files)
    
    model = build_and_train_model(X_train, y_train)

    predictions = predict_stock_rank(model, latest_features)
    predictions = predictions.iloc[:50].copy()

    print("Predicted Stock Ranking 2025 table (1 = best):")
    for _, row in predictions.iterrows():
        print(
            f"{int(row['Predicted Rank'])}. {row['Symbol']} "
            f"(expected rank {row['Expected Rank']:.2f})"
        )


if __name__ == "__main__":
    main()