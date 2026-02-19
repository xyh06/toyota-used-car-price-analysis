# Pricing Behavior and Liquidity Risk in the Used Car Market
### Evidence from Toyota Listings (UK Cross-Sectional Dataset)

🔗 **Live Interactive Report**  
https://xyh06.github.io/Toyota-Used-Car-Market-Mispricing-Analysis/

---

## Project Overview

This project analyzes dealer pricing behavior using approximately 6,700 UK Toyota used-car listings.

Instead of estimating a vehicle’s intrinsic value, a machine learning price model is treated as a **market consensus benchmark**.  
Listings are evaluated by how much they deviate from that benchmark.

The goal is behavioral interpretation — not arbitrage detection, profitability estimation, or claims of market inefficiency.

---

## Interactive Visualization

Open the full analysis dashboard:

👉 https://xyh06.github.io/Toyota-Used-Car-Market-Mispricing-Analysis/

The webpage contains all figures, model diagnostics, and descriptive results in a readable report format.

---

## Motivation

Used-car dealers balance two competing objectives:

- Faster turnover — listing below typical market level
- Margin exploration — listing above market level to test buyer willingness

Because the dataset contains **no transaction outcomes**, the project does NOT model:

- Sale probability
- Days-on-market
- Profitability

Instead, it measures relative positioning within the market price distribution.

---

## Data Source

Dataset: Toyota Used Cars Market Insights (Kaggle)  
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

- Cross-sectional snapshot only
- No sale prices
- No time-to-sale
- No dealer identity
- No geography
- No time series variation

---

## Methodology

### 1. Market Consensus Price Model
Random Forest regression estimates expected listing price from vehicle attributes.

Test R²: **0.863**

The prediction is interpreted as a market consensus level, not intrinsic value.

---

### 2. Market Acceptance Index (MAI)

MAI = Listing Price / Predicted Market Price


| MAI Range | Interpretation | Pricing Stance |
|--------|------|------|
| < 0.90 | Below consensus | Turnover oriented |
| 0.90 – 1.10 | Near consensus | Market aligned |
| 1.10 – 1.25 | Premium positioning | Margin exploration |
| > 1.25 | Large deviation | Elevated acceptance difficulty (proxy) |

MAI measures relative positioning, not correctness of price.

---

### 3. Behavioral Pattern Analysis
MAI is analyzed against vehicle characteristics to identify systematic pricing patterns.

### 4. High-Deviation Exposure Indicator
Heuristic threshold:

MAI > 1.15


Represents top-tail deviation exposure (descriptive only, not a sale probability).

---

## Key Findings

- Hybrid vehicles tend to appear at higher relative premiums
- Vehicles aged 3–6 years show strongest positive deviation
- Large engines combined with high mileage frequently appear in high-deviation listings

These patterns describe pricing behavior — not mispricing profitability.

---

## Skills Demonstrated

- Feature engineering under limited observational data
- Non-parametric regression modeling
- Custom behavioral metric design (MAI)
- Careful proxy interpretation
- Translating ML output into interpretable economic insight
- Communicating statistical limitations explicitly

---

## Repository Structure

├── toyota.csv
├── Toyota-Used-Car-Market-Mispricing-Analysis.ipynb
├── index.html
├── images/
└── README.md


---

## Takeaway

A predictive model can be used as a behavioral benchmark rather than a valuation tool.

This project demonstrates how interpretable insights can be extracted from incomplete real-world data while avoiding unsupported causal or profitability claims.

---

Author: xyh06  
Last Updated: February 2026

