Used Car Price Formation Mechanism Analysis
Project Overview

This project investigates how used car prices are formed, rather than only predicting prices.

Using 6,738 Toyota used-car records from the UK market (2013–2020),
I construct a machine learning–based market pricing benchmark and analyze the structural mechanism behind valuation.

Research Questions

What factors dominate used-car price formation?

Do preferences affect absolute price or relative positioning?

Is pricing primarily driven by depreciation or configuration?

Methodology

Data Cleaning: IQR-based outlier treatment

Feature Engineering: Vehicle age, mileage intensity, engine structure

Model: Random Forest Regression

Performance: R² ≈ 0.86

Market Price Index = Actual Price / Predicted Price

Key Findings

1️⃣ Depreciation Dominates

Vehicle age and mileage explain most price variation.
Used car markets are primarily structured around depreciation curves.

2️⃣ Mechanical Features Add Stable Premium

Engine size and fuel type affect pricing within the depreciation band,
but do not override depreciation structure.

3️⃣ Preferences Affect Relative Positioning

Model type influences value positioning rather than absolute price level.

Core Conclusion

Used car pricing follows a:

Depreciation First → Configuration Second → Preference Adjustment

mechanism.

Technical Stack

Python · Pandas · Scikit-learn · Matplotlib

Author

Maintained by yh
Undergraduate in Data Science | Preparing for Applied Statistics

https://nbviewer.org/github/xyh06/toyota-used-car-price-analysis/blob/main/toyota_price_analysis.ipynb
