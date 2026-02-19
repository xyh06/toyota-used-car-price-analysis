# Pricing Behavior and Liquidity Risk in the Used Car Market
### Evidence from UK Toyota Listings

> Behavioral pricing analysis using machine learning as a market benchmark  
> (~6,700 listings · cross-sectional dataset · descriptive study)

---

## Live Report
👉 https://xyh06.github.io/Toyota-Used-Car-Market-Mispricing-Analysis/

This repository contains the code behind the interactive analytical report above.  
The webpage is the recommended reading entry — the notebook is supporting material.

---

## What This Project Actually Does

This is NOT a price prediction project.

The model is used only to approximate a **market consensus price level**.  
Listings are then evaluated by how much they deviate from that consensus.

The objective is to study dealer pricing behavior under limited observable data.

---

## Why This Matters

Used-car listings contain no:
- transaction price
- time-to-sale
- realized profit

Therefore most “mispricing” studies implicitly assume outcomes they cannot observe.

This project avoids that assumption.

Instead of predicting value → it measures **relative positioning** inside the market.

---

## Method (Conceptual)

1. Estimate consensus price surface (Random Forest)
2. Construct relative pricing index

Market Acceptance Index:

MAI = Listing Price / Predicted Market Price

Interpretation:

| Range | Meaning |
|------|------|
| <0.90 | turnover-oriented |
| 0.90–1.10 | aligned |
| 1.10–1.25 | premium exploration |
| >1.25 | high acceptance difficulty proxy |

The index measures behavior, not correctness.

---

## Key Observations

- Hybrids tend to appear at higher relative premiums
- 3–6 year vehicles show strongest positive deviation
- Large engines + high mileage often occur in high-deviation listings

These describe listing strategy patterns only.

---

## What This Project Does NOT Claim

- No arbitrage opportunities
- No sale probability estimation
- No profitability prediction
- No causal inference

Purely descriptive behavioral analysis.

---

## Skills Demonstrated

- Designing metrics when outcomes are unobserved
- Translating ML outputs into interpretable economic meaning
- Working under strict observational constraints
- Communicating statistical limitations explicitly

---

## Repository Structure

```text
Toyota-Used-Car-Market-Mispricing-Analysis/
├── toyota.csv
├── Toyota-Used-Car-Market-Mispricing-Analysis.ipynb
├── index.html
├── images/
└── README.md


---
Takeaway

Prediction models can function as behavioral benchmarks rather than valuation tools.

This project shows how meaningful insights can be extracted even when the dataset lacks outcome variables — by reframing the question instead of over-interpreting the model.

Author: xyh06
Last Updated: February 2026

