# Toyota Used Car Market Mispricing Analysis

A data analysis project that identifies pricing deviation patterns in the Toyota used-car market using machine learning regression and statistical indicators.

Instead of claiming a “true value”, the model prediction is treated as a **market consensus benchmark**.  
Listings are evaluated by how much they deviate from that benchmark.

---

## 🌐 Interactive Report (GitHub Pages)

Full visual report (all charts & model outputs):

👉 https://xyh06.github.io/Toyota-Used-Car-Market-Mispricing-Analysis/

---

## Project Objective

Used-car pricing is uncertain. Buyers and sellers do not know whether a listing is relatively expensive or cheap within the market distribution.

This project aims to:

- Estimate expected market price via ML regression
- Measure relative deviation from market consensus
- Detect aggressive vs premium pricing behavior
- Analyze pricing patterns across model types and vehicle age

---

## Methodology

1. Data Cleaning & Feature Engineering  
2. Exploratory Data Analysis  
3. Price Prediction Model (Random Forest)  
   - **Test R²: 0.863 (MAE/RMSE/MAPE reported in notebook; reasonable for noisy domain)**
4. Pricing Deviation Indicator Construction  
5. Market Pattern Analysis  

---

## Market Acceptance Indicator

```
relative_deviation = (actual_price − predicted_price) / predicted_price
```

Interpretation:

| Value | Meaning |
|------|------|
| Negative | Below market level (turnover oriented) |
| Near 0 | Market aligned |
| Positive | Premium positioning |

This measures **relative positioning**, not profitability or arbitrage.

---

## Data Source

Dataset: https://www.kaggle.com/datasets/anassarfraz13/toyota-used-car-market-insights  
Kaggle — *Toyota Used Cars Market Insights (Anas Sarfraz)*

Scope: ~6,738 UK Toyota listings (circa 2020)

Features:
- Model
- Registration year
- Mileage
- Transmission
- Fuel type
- Engine size
- MPG
- Listing price

Constraints:
- Cross-sectional only
- No transaction price
- No time-to-sale
- No dealer identity
- No geography

---

## Key Visualizations

### Price Relationships
![Year vs Price and Mileage vs Price](images/Year%20vs%20Price%20and%20Mileage%20vs%20Price.png)

### Price Distribution by Model
![Price Distribution by Model](images/Price%20Distribution%20by%20Model.png)

### Correlation Matrix
![Correlation Matrix](images/Correlation%20Matrix.png)

---

### Pricing Deviation Analysis
![Distribution of relative_deviation Index](images/Distribution%20of%20relative_deviation%20Index.png)
![relative_deviation Index by Category](images/relative_deviation%20Index%20by%20Category.png)
![relative_deviation by Model](images/relative_deviation%20by%20Model.png)
![relative_deviation index of different vehicle models](images/relative_deviation%20index%20of%20different%20vehicle%20models.png)
![Average relative_deviation Index by Toyota Model](images/Average%20relative_deviation%20Index%20by%20Toyota%20Model.png)

---

### Vehicle Age Effects
![Average relative_deviation Index by Model and Vehicle Age](images/Average%20relative_deviation%20Index%20by%20Model%20and%20Vehicle%20Age.png)
![Average Potential interpretation by Vehicle Age](images/Average%20Potential%20interpretation%20by%20Vehicle%20Age.png)

---

### Suggested Opportunities
![Potential interpretation Distribution for Suggested Vehicles](images/Potential%20interpretation%20Distribution%20for%20Suggested%20Vehicles.png)

This chart highlights vehicles with the strongest relative undervaluation signals.

---

## Repository Structure

```
Toyota-Used-Car-Market-Mispricing-Analysis/
│
├── toyota.csv
├── Toyota-Used-Car-Market-Mispricing-Analysis.ipynb
├── index.html
├── images/
│   ├── (analysis figures)
└── README.md
```

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

---

## Takeaway

A predictive model can be used as a **behavioral benchmark** rather than a valuation tool.

This project demonstrates how interpretable insights can be extracted from incomplete real-world data while avoiding unsupported causal or profitability claims.

---

**Author:** xyh06  
**Last Updated:** February 2026

