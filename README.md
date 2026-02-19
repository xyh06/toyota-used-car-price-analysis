Pricing Behavior and Liquidity Risk in the Used Car Market
Evidence from Toyota Listings (UK Cross-Sectional Dataset)

🔗 Live Interactive Report
https://xyh06.github.io/Toyota-Used-Car-Market-Mispricing-Analysis/

Project Overview

This project analyzes dealer pricing behavior using approximately 6,700 UK Toyota used-car listings.

Instead of estimating a vehicle’s intrinsic value, a machine learning price model is treated as a market consensus benchmark.
Listings are evaluated by how much they deviate from that benchmark.

The goal is behavioral interpretation — not arbitrage detection, profitability estimation, or claims of market inefficiency.

Interactive Visualization

Open the full analysis dashboard:

👉 https://xyh06.github.io/Toyota-Used-Car-Market-Mispricing-Analysis/

The webpage contains all figures, model diagnostics, and descriptive results in a readable report format.

Motivation

Used-car dealers balance two competing objectives:

Faster turnover — listing below typical market level

Margin exploration — listing above market level to test buyer willingness

Because the dataset contains no transaction outcomes, the project does NOT model:

Sale probability

Days-on-market

Profitability

Instead, it measures relative positioning within the market price distribution.

Data Source

Dataset: Toyota Used Cars Market Insights (Kaggle, Anas Sarfraz)
https://www.kaggle.com/datasets/anassarfraz13/toyota-used-car-market-insights

Scope: ~6,738 UK Toyota listings (circa 2020)

Features:

Model

Registration year

Mileage

Transmission

Fuel type

Engine size

MPG

Listing price

Constraints:

Cross-sectional snapshot only

No sale prices

No time-to-sale

No dealer identity

No geography

No time series variation

Methodology
1. Market Consensus Price Model

Random Forest regression estimates expected listing price from vehicle attributes.

Test R²: 0.863 (MAE/RMSE/MAPE reported in notebook; reasonable for noisy domain)

The prediction is interpreted as a market consensus level, not intrinsic value.

2. Market Acceptance Index (MAI)

MAI = Listing Price / Predicted Market Price

MAI Range	Interpretation	Pricing Stance
< 0.90	Below consensus	Turnover oriented
0.90 – 1.10	Near consensus	Market aligned
1.10 – 1.25	Premium positioning	Margin exploration
> 1.25	Large deviation	Elevated acceptance difficulty (proxy)

MAI measures relative positioning, not correctness of price.

3. Behavioral Pattern Analysis

MAI is analyzed against vehicle characteristics to identify systematic pricing patterns.

4. High-Deviation Exposure Indicator

Heuristic threshold:

MAI > 1.15

Represents top-tail deviation exposure (descriptive only, not a sale probability).

Key Findings

Hybrid vehicles tend to list at higher relative premiums (e.g., average MAI ≈ 1.18 for Prius-type models).

Vehicles aged 3–6 years show the strongest positive deviation (peak mean MAI ≈ 1.12).

High mileage combined with large engines frequently appears in high-deviation listings (~18–22% exceed 1.15 threshold).






These patterns describe pricing behavior — not mispricing profitability.

Business Implications
Dealer Pricing Strategy

High-premium perception vehicles (hybrid trims, low-mileage cars) behave as margin-seeking inventory.
Dealers may prioritize profit per sale rather than rapid turnover.

Commodity vehicles (economy trims, high mileage) function as liquidity inventory and benefit from competitive pricing.

Inventory Allocation

The results imply a two-tier inventory strategy:

Premium-perception vehicles → hold longer, price above benchmark

Commodity vehicles → price competitively to accelerate sales

Market Risk Signal

Persistent positive deviation in specific categories may indicate demand-driven price pressure rather than valuation accuracy.
If demand weakens, these categories face larger downward adjustment risk.

Skills Demonstrated

Feature engineering under limited observational data

Non-parametric regression modeling

Custom behavioral metric design (MAI)

Careful proxy interpretation

Translating ML output into interpretable economic insight

Communicating statistical limitations explicitly

Repository Structure
toyota.csv
Toyota-Used-Car-Market-Mispricing-Analysis.ipynb
index.html
images/
README.md

Takeaway

A predictive model can serve as a behavioral benchmark rather than a valuation tool.

This project does not attempt to identify arbitrage opportunities,
but to understand how market participants systematically price assets under bounded rationality.

Author: xyh06
Last Updated: February 2026

