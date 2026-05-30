# Trade Classification of ExxonMobil

<img width="2000" height="1050" alt="exxonmobil-cut-jobs" src="https://github.com/user-attachments/assets/198897cb-8c8c-4927-a405-5e17fd51371f" />

## 1. Introduction 
In quantitative finance, understanding order flow - identifying whether buyers or sellers initiate trades - is critical for analyzing liquidity, price discovery, and short-term market momentum. Because standard historical datasets typically omit explicit trade direction labels, researchers rely on market microstructure rules to infer them. One of the most seminal frameworks for this is the Lee-Ready Algorithm, which dynamically combines the Quote Rule and the Tick Test to classify trades based on their positioning relative to prevailing bid-ask spreads and historical price changes. 

This micro-project applies the Lee-Ready framework to historical daily data for ExxonMobil (XOM) spanning from 2023 to 2025 to generate target classification labels (Buy vs. Sell). Using engineered features derived from price fluctuations and volume indicators, we train a Random Forest Classifier to predict these trade directions. The model's predictive performance is evaluated sequentially on the training and testing sets to assess its robustness, alongside an analysis of feature importances to identify the key drivers of order-flow classification in a major energy equity. 

## 2. Objectives
- Apply the combined logic of the Quote Rule and Tick-Test to ExxonMobil historical data to construct a robust, rule-based target label for classification.
- Building predictive features from underlying market data to provide the machine learning model with adequate structural context.
- Execute an 80/20 chronological train-test split on the 2023-2025 dataset to preserve time series integrity and prevent future data leakage.
- Assess and compare model performance metrics across both the training and testing sets to detect potential overfitting or baseline drift.
- Extract and interpret the Random Forest feature importance scores to isolate which market metrics exert the strongest influence on trade classification boundaries. 


## 3. Trade Labelling - Lee and Ready Algorithm 
The Lee-Ready Algorithm is a hybrid classification rule that combines the Quote Rule and the Tick Test to decide whether a trade was buyer-initiated or seller-initiated. 

```` python
def tick_test_algo(row):
  if row['price_diff'] > 0:
    return 1
  elif row['price_diff'] < 0:
    return 0
  else:
    return np.nan

def quote_rule_algo(row):
  if row['close'] > row['midpoint']:
    return 1
  elif row['close'] < row['midpoint']:
    return 0
  else:
    return tick_test_algo(row)

df2['target'] = df2.apply(quote_rule_algo, axis = 1)

````

### 3.1 The Quote Rule
The algorithm starts inside quote_rule_algo(row) by looking at where the closing price sits relative to the day's midpoint boundary. 

- If close > midpoint: The trade is classified immediately as Buy (1) because it is closer to the seller's asking price.
- If close < midpoint: The trade is classified immediately as Sell (0) because it is closer to the buyer's bid price.

### 3.2 The Tick Test
If the closing price lands exactly on the midpoint, the Quote Rule becomes ambiguous. The Tick Test ignores the midpoint and instead looks backward at momentum. 
- If price diff > 0: The price rose since the last period, so it outputs a Buy (1).
- If price diff < 0: The price dropped since the last period, so it outputs a Sell (0).
- If price diff = 0: The price remained completely unchanged, resulting in a missing value, which would typically be filled forward using historical context. 


## 4. Feature Engineering 
Feature engineering transforms raw historical market data into structured signals that expose the underlying dynamics of price volatility, intraday momentum, and trading intensity. By isolating these specific relationships, we provide the Random Forest classifier with contextual patterns that help it map market behaviour directly to the Lee-Ready trade classification targets. 

1. `open_close_diff`: Captures intraday price direction and momentum by measuring the net change between the opening and closing prices.
2. `high_low_diff`: Measures daily market volatility and trading range width by calculating the distance between the highest and lowest prices.
3. `volume_change`: Tracks shifting market intensity and liquidity acceleration by calculating the daily percentage change in traded share volume.

````python
df2['open_close_diff'] = df2['open'] - df2['close']

df2['high_low_diff'] = df2['high'] - df2['low']

df2['volume_change'] = df2['volume'].pct_change()

df2 = df2.dropna()

````

## 5. Split Data 
This step partitions the engineered features and Lee-Ready target labels into independent datasets for sequential model training and validation.

1. `X = df2[['open_close_diff', 'high_low_diff', 'volume_change']]`: Isolates the independent variables into a feature matrix used by the model to find predictive patterns.
2. `y = df2['target']`: Extracts the dependent variable vector containing the Lee-Ready trade classification labels the model aims to predict.
3. `X_train, X_test, y_train, y_test = train_test_split(...)`: allocates exactly 20% of the data for model evaluation, leaving the remaining 80% for training; preserves chronological order to prevent data leakage, ensuring the model trains on the past to predict the future.

````python
X = df2[['open_close_diff', 'high_low_diff', 'volume_change']]
y = df2['target']

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size = 0.2,
                                                    shuffle = False)

````

## 6. Random Forest Classifier
The Random Forest Classifier was selected because it is effective in modeling complex nonlinear relationships and interactions between engineered features without requiring strong statistical assumptions. Its ensemble structure combines multiple decision trees, improving prediction stability and reducing overfitting compared to a single decision tree.

### 6.1 Hyperparameter Tuning and Model Training
To improve model performance, hyperparameter tuning was performed using GridSearchCV with TimeSeriesSplit cross-validation, which preserves the chronological order of time-series and prevents future data leakage. The tuning process evaluated different combinations of key parameters, including the number of trees (n_estimators = 50 to 100), maximum tree depth (max_depth = 8 to 10), minimum samples required at leaf nodes (min_samples_leaf = 1 to 5), and minimum samples required for node splitting (min_samples_split = 2 to 5). Model performance was evaluated using negative mean squared error, and the parameter combination with the best validation performance was selected as the final optimized Random Forest model.

````python
# initialize the base random forest regressor
rfr = RandomForestClassifier(random_state = 42)

# define the hyperparameter grid
param_grid = {'n_estimators': [50, 60, 70, 80, 90, 100],
              'max_depth': [8, 9, 10],
              'min_samples_leaf': [1, 2, 3, 4, 5],
              'min_samples_split': [2, 3, 4, 5]}

# set up time series split
tscv = TimeSeriesSplit(n_splits = 5)

# set up the grid search cross-validation configuration
grid_search = GridSearchCV(estimator = rfr,
                           param_grid = param_grid,
                           cv = tscv,
                           n_jobs = 1,
                           verbose = 3,
                           scoring = 'accuracy',
                           return_train_score = True,
                           refit = True)

# execute the search over all parameter combination
grid_search.fit(X_train, y_train)


````
The best estimators are `{'max_depth': 8, 'min_samples_leaf': 4, 'min_samples_split': 2, 'n_estimators': 70}`.

The best score for the best estimators is `0.8099999999999999`.

## 7. Model Evaluation 
Based on the model evaluation metrics, here is a summary of the performance for the trade classification model:

- The model performs well on both datasets, achieving 84% accuracy on the unseen test set compared to 91% accuract on the training set.
- The minor drop in accuracy and ROC score from train to test indicates that the Random Forest model generalizes well and has not heavily overfit the historical data.
- The exceptionally high ROC scores prove that the model is highly effective at separating the underlying probabilities of a Buy versus a Sell signal.


## 8. Future Work 
Future work will integrate advanced deep learning models such as LSTMs or Transformers, alongside sentiment analysis of financial news, to capture non-linear temporal relationships and improve classification robustness in volatile markets. 


## 9. Conclusion 
The model successfully predicts ExxonMobil trade direction with 84% test accuracy using Lee-Ready labels and Random Forest classification. 


## 10. Libraries 
- pandas
- numpy
- sklearn
- matplotlib
- seaborn
- yfinance



## References
[1] Menaldo, S. (2026, March 10). Trade Classification Algorithms. Medium. https://medium.com/@simomenaldo/trade-classification-algorithms-6a2fede1e4f5

‌
