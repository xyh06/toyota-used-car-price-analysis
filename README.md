# Toyota Used Car Market Mispricing Analysis

## Overview

The used car market is highly influenced by brand perception, negotiation, and incomplete information.
As a result, vehicles with similar specifications may be priced very differently.

Instead of only predicting prices, this project focuses on **detecting pricing inefficiencies** in the market.

We build a machine-learning–based fair price model and analyze the deviation between market price and statistically estimated value to identify potential trading opportunities.

---

## Business Problem

The used car market suffers from significant information asymmetry.
Prices are often determined by seller expectations rather than objective value.

For dealerships and traders, the key challenge is not predicting price, but identifying **mispriced vehicles**.

This project answers four practical business questions:

1. Are certain Toyota models systematically overpriced or underpriced?
2. Which vehicles present potential arbitrage opportunities?
3. How should a dealership design a rational acquisition pricing strategy?
4. How can we construct a fair baseline price for valuation?

To solve this, we introduce the **Mispricing Index**:

> **Mispricing Index = Market Price / Model Fair Price**

| Value       | Interpretation                |
| ----------- | ----------------------------- |
| < 0.85      | Undervalued (buy opportunity) |
| 0.85 – 1.15 | Fair price                    |
| > 1.15      | Overpriced                    |

---

## Dataset

* UK Toyota used car listings
* ~6,700 vehicles
* Attributes include:

  * Model
  * Year
  * Mileage
  * Transmission
  * Fuel type
  * Engine size
  * MPG
  * Price

---

## Methodology

### 1. Data Cleaning

* Removed duplicates
* Checked missing values
* Converted categorical variables
* Created vehicle age feature

### 2. Feature Engineering

* Vehicle Age = Current Year − Year
* Encoded categorical features
* Standardized numerical variables where necessary

### 3. Fair Price Model

A **Random Forest Regressor** is trained to estimate the statistically fair market value of each vehicle.

Model output:

> Predicted Price = Fair Price Baseline

### 4. Mispricing Detection

We define a market deviation indicator:

```
Mispricing Index = Actual Price / Predicted Price
```

This transforms a prediction task into a market behavior analysis problem.

---

## Key Findings

### Model-level insights

* Vehicle age and mileage are dominant price drivers
* Transmission and engine size contribute secondary effects
* Brand perception causes systematic deviations from fair value

### Market behavior

* Some models consistently trade above fair value (brand premium)
* Mid-age vehicles tend to be underpriced
* Price dispersion increases significantly for older cars

### Trading implication

The market does not price vehicles purely based on fundamentals —
there exists persistent structural mispricing.

---

## Acquisition Strategy Simulation

We simulate a dealership buying rule:

> Buy when Mispricing Index < 0.85

Potential profit:

```
Potential Profit = Predicted Price − Market Price
```

Result:

* Identifies a subset of listings with statistically positive expected margin
* Demonstrates data-driven purchasing strategy instead of manual appraisal

---

## Project Structure

```
toyota-used-car-price-analysis/
│
├── data/
├── notebook/
│   └── toyota_price_analysis.ipynb
├── images/
├── README.md
```

---

## Tools & Libraries

* Python
* Pandas
* NumPy
* Matplotlib / Seaborn
* Scikit-learn (Random Forest)

---

## How to Run

```bash
git clone https://github.com/yourname/toyota-used-car-price-analysis.git
cd toyota-used-car-price-analysis
pip install -r requirements.txt
jupyter notebook
```

---

## Conclusion

This project demonstrates that:

* Predicting price alone has limited business value
* Detecting pricing deviation provides actionable insight
* Machine learning can support acquisition decision-making

The methodology can be generalized to other second-hand markets such as electronics, housing, or collectibles.

---

## Future Work

* Cross-validation and model comparison (XGBoost, Linear Models)
* SHAP explainability analysis
* Time-series market trend analysis
* Dealer profit back-testing

---

## Author

Independent data analysis project for portfolio demonstration.



