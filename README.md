# Trading desk data

Public market data only. A GitHub Action pulls daily bars (Yahoo Finance via yfinance), computes indicators, and writes `data/latest.json` several times each trading day. Hamid's Trading Desk reads it.
