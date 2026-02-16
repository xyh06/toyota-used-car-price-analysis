# Used Car Price Mechanism Analysis (Machine Learning)

## Background

Understanding used-car pricing is important for buyers, sellers, and platforms.
Instead of only predicting price, this project explains **how prices are formed**.

Dataset: 6,738 Toyota used cars (UK market, 2013–2020)

---

## Objective

* Identify key factors affecting price
* Build a market benchmark pricing model
* Evaluate relative value retention of different models

---

## Method

* Data cleaning with IQR outlier handling
* Feature engineering (age, mileage, engine specs)
* Random Forest regression (R² ≈ 0.86)
* Price Index = Actual Price / Predicted Market Price

---

## Key Insights

1. Depreciation dominates price formation
   Age and mileage explain most price variation

2. Mechanical features provide stable premium
   Engine size and fuel efficiency affect valuation range

3. Preferences influence relative value rather than absolute price
   Model and fuel type affect positioning within depreciation band

**Conclusion:**
Used car pricing follows a

> Depreciation First → Preference Second
> mechanism

---

## Tech Stack

Python, Pandas, Scikit-learn, Data Visualization

---

## Author

Data Science Undergraduate | Interested in Data Analysis & Applied Statistics

