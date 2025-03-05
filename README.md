# Financial Market Dynamics Analysis

## Project Overview
This project analyzes the dynamic relationships between equity markets, treasury yields, and volatility indicators using advanced time series analysis techniques. By examining the intricate interplay between these financial variables, the project aims to identify patterns, detect market regimes, and model the time-varying correlations and volatilities that characterize financial markets.

## Data Sources
The analysis uses daily data from January 2007 to December 2024 for the following financial instruments:

- **Equity Indices**: S&P 500 (`^GSPC`)
- **Treasury Yields**: 10-year Treasury (`^TNX`), 30-year Treasury (`^TYX`), and the 30y-10y spread
- **Volatility Index**: VIX (`^VIX`)
- **Additional Indices**: NASDAQ 100 (`^NDX`), NASDAQ Composite (`^IXIC`)

All data is sourced from Yahoo Finance using the `yfinance` Python package.

## Methodology

### Data Preprocessing
1. **Feature Selection**: Based on correlation analysis and Variance Inflation Factor (VIF), redundant variables (NASDAQ 100, NASDAQ Composite, Treasury 30Y, and Treasury 10Y) were removed, keeping only the most informative variables.
2. **Transformations**:
   - Equity indices: Converted to log returns
   - Treasury yields: Calculated changes
   - VIX : Log transformed
   - All series: Standardized to ensure comparable scales

### Time Series Analysis
1. **Stationarity Testing**:
   - Augmented Dickey-Fuller (ADF) test
   - Phillips-Perron (PP) test
   - KPSS test
   - Fourier KPSS test for smooth structural breaks
   - Panel unit root tests (IPS and LLC)

2. **Structural Break Detection**:
   - Chow test
   - Zivot-Andrews test
   - Bai-Perron test

3. **Regime Analysis**: 
   - Historical regimes defined (Pre-GFC, GFC, Post-GFC Recovery, QE Era, Pre-COVID, COVID Crisis, Post-COVID Recovery, Inflation/Rate Hikes, Recent Markets)
   - Summary statistics calculated for each regime

### Volatility Modeling

1. **ARCH Effects Testing**:
   - ARCH-LM tests for heteroskedasticity
   - ACF/PACF analysis of squared residuals

2. **GARCH Modeling**:
   - Comparative analysis of GARCH(1,1), EGARCH(1,1), and GJR-GARCH(1,1) models
   - Model selection based on AIC/BIC criteria
   
3. **Advanced Volatility Analysis**:
   - Hawkes process modeling for self-exciting volatility events
   - Heston model analysis for volatility dynamics

### DCC-GARCH Implementation
1. **Model Specification**:
   - Optimized univariate GARCH specifications for each series:
     - S&P 500: gjrGARCH(2,1) with ARMA(1,0) and skewed Student-t distribution
     - Treasury spread: gjrGARCH(1,1) with ARMA(1,0) and skewed Student-t distribution
     - VIX: gjrGARCH(2,2) with ARMA(1,1) and skewed Student-t distribution
   - DCC specification: DCC(2,2) with multivariate Student-t distribution

2. **Diagnostic Testing**:
   - Ljung-Box tests for serial correlation in standardized residuals
   - ARCH-LM tests for remaining ARCH effects
   - Jarque-Bera tests for normality

3. **Visualization and Analysis**:
   - Conditional volatilities over time
   - Time-varying conditional correlations
   - Regime detection based on volatility and correlation dynamics

### Machine Learning for Volatility Forecasting
1. **Model Comparison**:
   - Traditional GARCH models: Forecasts using optimized specifications
   - LSTM neural networks: Deep learning approach for volatility prediction
   - Hybrid GARCH-LSTM: Combining econometric and machine learning methods

2. **Feature Engineering**:
   - Historical returns and volatilities as base features
   - GARCH residuals and conditional volatilities as additional features
   - Realized volatility calculated using rolling windows for target validation

3. **Training Methodology**:
   - Time-series split for training and validation
   - Hyperparameter optimization for LSTM architecture
   - Multiple prediction horizons (30-day, 90-day, 180-day, 252 day forecasts)

4. **Performance Evaluation**:
   - Mean Squared Error (MSE)
   - Mean Absolute Error (MAE)
   - Directional accuracy
   - Value-at-Risk (VaR) backtesting

5. **Key Findings**:
   - Hybrid GARCH-LSTM model generally outperforms standalone models
   - LSTM shows improved performance during regime transitions
   - Traditional GARCH models provide better interpretability of volatility dynamics
   - Machine learning approach adds value especially during periods of market stress

## Key Findings

### Volatility Dynamics
- All financial series exhibit significant ARCH effects, confirming the presence of volatility clustering
- Asymmetric volatility response (leverage effects) detected across series, with negative shocks increasing volatility more than positive shocks of the same magnitude
- Non-normal return distributions with heavy tails, best captured by skewed Student-t distributions
- Major volatility spikes coincide with financial crises (2008-2009) and the COVID-19 pandemic (2020)

### Correlation Dynamics
- Time-varying correlations between S&P 500, Treasury spread, and VIX
- Strong negative correlation between S&P 500 and VIX, especially during market stress periods
- Variable correlation between Treasury spread and equity/volatility markets, reflecting changing macroeconomic conditions

### Market Regimes
- DCC-GARCH analysis reveals distinct market regimes characterized by different correlation and volatility patterns
- Crisis periods exhibit higher correlations between assets and elevated volatility
- Multivariate Student-t distribution better captures joint extreme movements across assets than normal distribution

## Usage Instructions

### Dependencies
The analysis requires the following Python packages:
- numpy
- pandas
- matplotlib
- seaborn
- yfinance
- statsmodels
- scipy
- arch
- rpy2
- hmmlearn
- scikit-learn
- tensorflow (for LSTM models)
- keras

Install all dependencies using:
```bash
pip install -r requirements.txt
```

Additional R packages required for DCC-GARCH modeling:
```R
install.packages('rmgarch')
```

### Workflow
1. Data collection 
2. Data preprocessing and transformation 
3. Stationarity testing with various unit root tests
4. ARCH effects testing 
5. GARCH model comparison 
6. Volatility dynamics analysis 
7. DCC-GARCH implementation via R's rmgarch package through rpy2
8. Visualization of conditional volatilities and correlations
9. Market regime detection
10. Machine learning model training and evaluation for volatility forecasting
