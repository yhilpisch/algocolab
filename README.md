<img src="https://hilpisch.com/tpq_logo_bic.png" width="25%" align="right">

# Algorithmic Trading with Python & Google Colab
## 3-Part Webinar Series · Companion Code Repository

[![The Python Quants](https://img.shields.io/badge/The%20Python%20Quants-tpq.io-002D5A.svg)](https://tpq.io)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg?logo=pytorch)](https://pytorch.org/)
[![ZeroMQ](https://img.shields.io/badge/ZeroMQ-pyzmq-orange.svg)](https://zeromq.org/)
[![License: Proprietary](https://img.shields.io/badge/License-All%20Rights%20Reserved-lightgrey.svg)](https://tpq.io)

<p align="center">
  <img src="https://hilpisch.com/algo_webinar.png" alt="Algorithmic Trading with Python and Google Colab" width="100%">
</p>

---

## 📌 Overview

This repository contains the complete companion code and interactive Jupyter notebooks for the three-part webinar series on **Algorithmic Trading with Python and Google Colab**, presented by **Dr. Yves J. Hilpisch** ([The Python Quants GmbH](https://tpq.io)).

The curriculum bridges quantitative finance theory and scalable cloud execution across three core pillars:
**Discover** (Hypothesis testing & EMH) →
ightarrow→ **Learn** (GPU deep learning with PyTorch) →
ightarrow→ **Deploy** (ZeroMQ streaming, SQLite persistence & operational risk).

---

## 🚀 Interactive Google Colab Notebooks

Each notebook is completely self-contained and pre-configured to run directly in Google Colab with zero local installation:

| Session | Topic | Key Concepts | Colab Link |
| :--- | :--- | :--- | :--- |
| **Session 1** | **Can Markets Be Predicted?** | EMH benchmark, random walk null model (GBM), return autocorrelation, Ljung-Box test, linear OLS signal model, vectorized backtesting with transaction costs. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yhilpisch/algocolab/blob/main/notebooks/01_can_markets_be_predicted.ipynb) |
| **Session 2** | **Teaching a Neural Network to Trade** | Colab GPU tensor acceleration, quantitative feature engineering, multi-layer `TradingDNN` in PyTorch, cross-entropy vs economic Sharpe, confidence thresholding, model capacity analysis. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yhilpisch/algocolab/blob/main/notebooks/02_deep_learning_gpu_trading.ipynb) |
| **Session 3** | **From Notebook to Trading System** | Exact checkpoint loading, batch/stream parity, historical event replay, SQLite audit trail, failure injection, drawdown flattening, and reconciliation. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yhilpisch/algocolab/blob/main/notebooks/03_cloud_deployment_monitoring.ipynb) |

---

## 📂 Repository Structure

```text
.
├── README.md                           # Repository documentation & Colab access
├── requirements.txt                    # Core scientific & deep learning libraries
├── notebooks/                          # Standalone Jupyter Notebooks
│   ├── 01_can_markets_be_predicted.ipynb
│   ├── 02_deep_learning_gpu_trading.ipynb
│   └── 03_cloud_deployment_monitoring.ipynb
├── src/                                # Modular Python trading system code
│   ├── data.py                         # Historical data loader & lagged feature engine
│   ├── models.py                       # PyTorch TradingDNN & training loop
│   ├── backtest.py                     # Vectorized backtester & performance analytics
│   ├── engine.py                       # Live simulation engine & risk guardrails
│   ├── tick_server.py                  # ZeroMQ PUB socket streaming market ticks
│   ├── tick_database.py                # ZeroMQ SUB client logging ticks to SQLite
│   └── trading_client.py               # ZeroMQ SUB real-time trading engine
└── data/                               # Sample datasets
    └── eod_data.csv                    # Historical EOD prices (SPY, EURUSD, BTC, etc.)
```

---

## 🛠️ Local Installation & Setup

If running locally instead of Google Colab:

### 1. Clone the Repository
```bash
git clone https://github.com/yhilpisch/algocolab.git
cd algocolab
```

### 2. Set Up Virtual Environment & Dependencies
```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Running Live Streaming Simulation
In terminal 1 (start ZeroMQ tick server):
```bash
python src/tick_server.py
```

In terminal 2 (start database recorder):
```bash
python src/tick_database.py
```

In terminal 3 (start live trading client):
```bash
python -m src.trading_client
```

---

## 📊 Dataset

Historical daily price data is available in `data/eod_data.csv` and also served dynamically from:
- `https://hilpisch.com/eod_data.csv`

---

## ⚠️ Disclaimer

The material, code, and notebooks in this repository are provided **for educational, informational, and personal research purposes only**. They do not constitute financial, investment, legal, or professional advice, nor do they represent an offer or solicitation to buy or sell any security, financial instrument, or trading strategy.

Past performance, whether simulated, backtested, or real, is no guarantee of future results. Algorithmic trading and financial market investments carry substantial risk, including the possible loss of principal capital. You are solely responsible for evaluating the merits and risks associated with any decisions made based on this software or material.

---

## ⚖️ Copyright & License

All materials and code are protected by copyright.

&copy; Dr. Yves J. Hilpisch | The Python Quants GmbH  
Website: [https://tpq.io](https://tpq.io) | [https://hilpisch.com](https://hilpisch.com)

This code is licensed for **personal, non-commercial use only**. No part of this repository may be reproduced, republished, or utilized in commercial products, services, or paid training without prior written permission from **The Python Quants GmbH**.
