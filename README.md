# Trade Classification of ExxonMobil

<img width="2000" height="1050" alt="exxonmobil-cut-jobs" src="https://github.com/user-attachments/assets/198897cb-8c8c-4927-a405-5e17fd51371f" />

## 1. Introduction 
In quantitative finance, understanding order flow - identifying whether buyers or sellers initiate trades - is critical for analyzing liquidity, price discovery, and short-term market momentum. Because standard historical datasets typically omit explicit trade direction labels, researchers rely on market microstructure rules to infer them. One of the most seminal frameworks for this is the Lee-Ready Algorithm, which dynamically combines the Quote Rule and the Tick Test to classify trades based on their positioning relative to prevailing bid-ask spreads and historical price changes. 

This project applies the Lee-Ready framework to historical daily data for ExxonMobil (XOM) spanning from 2023 to 2025 to generate target classification labels (Buy vs. Sell). Using engineered features derived from price fluctuations and volume indicators, we train a Random Forest Classifier to predict these trade directions. The model's predictive performance is sequentially evaluated across training and testing sets to assess its robustness, alongside an analysis of feature importances to determine the key drivers of order flow classification in a major energy equity. 

## 2. Objectives
- Apply the combined logic of the Quote Rule and Tick-Test to ExxonMobil historical data to construct a robust, rule-based target label for classification.
- Building predictive features from underlying market data to provide the machine learning model with adequate structural context.
- Execute an 80/20 chronological train-test split on the 2023-2025 dataset to preserve time series integrity and prevent future data leakage.
- Assess and compare model performance metrics across both the training and testing sets to detect potential overfitting or baseline drift.
- Extract and interpret the Random Forest feature importance scores to isolate which market metrics exert the strongest influence on trade classification boundaries. 


## 3. Trade Labelling - Lee and Ready Algorithm 


## 4. Feature Engineering 


## 5. Model Training 



## 6. Model Evaluation 



## 7. Future Work 



## 8. Conclusion 



## 9. Libraries 



## References
[1] Menaldo, S. (2026, March 10). Trade Classification Algorithms. Medium. https://medium.com/@simomenaldo/trade-classification-algorithms-6a2fede1e4f5

‌
