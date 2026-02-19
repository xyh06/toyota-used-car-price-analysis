# Pricing Behavior and Liquidity Risk in the Used Car Market
### Evidence from Toyota Listings (UK Cross-Sectional Dataset)

[**▶ Live Interactive Report (Full Analysis)**](https://xyh06.github.io/Toyota-Used-Car-Market-Mispricing-Analysis/)

---

## Abstract

This project analyzes dealer pricing behavior using ~6,700 UK Toyota used-car listings.

Instead of estimating intrinsic value, a machine learning price model is used as a **market consensus benchmark**.  
Listings are interpreted by their deviation from this benchmark.

The objective is behavioral interpretation — not arbitrage detection, profitability estimation, or claims of market inefficiency.

---

## Motivation

Used-car dealers balance two competing objectives:

- Faster turnover — listing below typical market level
- Margin exploration — listing above market level

Because the dataset contains **no transaction outcomes**, the project does NOT model:

- Sale probability
- Days-on-market
- Profitability

Instead, it measures relative market positioning.

---

## Data

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
- No time variation

---

## Methodology

### Market Consensus Price Model
Random Forest regression estimates expected listing price from attributes.

**Test R²: 0.863**

Prediction is interpreted as a consensus level, not intrinsic value.

### Market Acceptance Index (MAI)

`MAI = Listing Price / Predicted Market Price`

| MAI Range | Interpretation | Pricing Stance |
|--------|------|------|
| < 0.90 | Below consensus | Turnover oriented |
| 0.90 – 1.10 | Near consensus | Market aligned |
| 1.10 – 1.25 | Premium positioning | Margin exploration |
| > 1.25 | Large deviation | Elevated acceptance difficulty (proxy) |

### High-Deviation Indicator
`MAI > 1.15`  
Top-tail deviation exposure (descriptive only)

---

## Key Findings

- Hybrid vehicles tend to appear at higher relative premiums
- Vehicles aged 3–6 years show strongest positive deviation
- Large engines with high mileage often appear in high-deviation listings

These patterns describe pricing behavior, not profitability or inefficiency.

---

## Limitations

- No transaction data → cannot validate sale speed
- Cross-sectional only → no regional or dealer effects
- Circa-2020 snapshot
- Consumer pricing noise

---

## Skills Demonstrated

- Feature engineering with incomplete observational data
- Non-parametric regression modeling
- Behavioral metric design (MAI)
- Proxy interpretation discipline
- Translating ML output into interpretable insights
- Explicit statistical limitation handling

---

## Repository Structure

```text
Toyota-Used-Car-Market-Mispricing-Analysis/
├── toyota.csv
├── Toyota-Used-Car-Market-Mispricing-Analysis.ipynb
├── index.html
├── images/
└── README.md
```

---

## Takeaway

A predictive model can function as a behavioral benchmark rather than a valuation tool.

The project demonstrates extracting interpretable patterns from incomplete real-world data while avoiding unsupported causal or profitability claims.

---

Author: xyh06  
Last Updated: February 2026

