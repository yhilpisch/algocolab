<img src="https://hilpisch.com/tpq_logo_bic.png" width="25%" align="right">

# Algorithmic Trading with Python & Google Colab
## 3-Part Webinar Series · Companion Code Repository

[![The Python Quants](https://img.shields.io/badge/The%20Python%20Quants-tpq.io-002D5A.svg)](https://tpq.io)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg?logo=pytorch)](https://pytorch.org/)
[![License: Proprietary](https://img.shields.io/badge/License-All%20Rights%20Reserved-lightgrey.svg)](https://tpq.io)

<p align="center">
  <img src="https://hilpisch.com/algo_webinar.png" alt="Algorithmic Trading with Python and Google Colab" width="100%">
</p>

---

## 📌 Overview

This repository contains the complete companion code and interactive Jupyter notebooks for the three-part webinar series on **Algorithmic Trading with Python and Google Colab**, presented by **Dr. Yves J. Hilpisch** ([The Python Quants GmbH](https://tpq.io)).

The curriculum bridges quantitative finance theory and auditable paper-trading
across three core pillars: **Discover** (hypothesis testing and EMH) →
**Learn** (GPU deep learning with PyTorch) → **Replay** (SQLite persistence,
operational risk, and reconciliation).

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
│   ├── 00_colab_introduction.ipynb
│   ├── 01_can_markets_be_predicted.ipynb
│   ├── 02_deep_learning_gpu_trading.ipynb
│   └── 03_cloud_deployment_monitoring.ipynb
├── src/                                # Modular Python trading system code
│   ├── data.py                         # Historical data loader & lagged feature engine
│   ├── models.py                       # PyTorch TradingDNN & training loop
│   ├── backtest.py                     # Vectorized backtester & performance analytics
│   ├── artifacts.py                    # Checksummed Drive run bundles
│   ├── config.py                       # Shared experiment configuration
│   ├── session1.py                     # Session 1 experiment orchestration
│   ├── session2.py                     # Session 2 model/evaluation orchestration
│   └── session3.py                     # Session 3 replay orchestration
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

### 3. Running the Session Replay
Run the participant notebooks in order. In Colab, Session 1 creates the Drive
run bundle, Session 2 adds the ensemble contract, and Session 3 performs the
auditable paper-trading replay.

---

## 📊 Dataset

The bundled `data/eod_data.csv` is the frozen teaching snapshot. Its source is:
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
