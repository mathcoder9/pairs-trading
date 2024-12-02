# Crypto Pairs Trading

This repository provides an implementation of a simple pairs trading strategy for cryptocurrencies. The Engle-Granger test is used to select suitable pairs and strategies are backtested.

## Repository Structure

- **`analysis`**: Contains the `simple_pairs.ipynb` notebook with exploratory data analysis (EDA), data cleaning, pair selection using the Engle-Granger Test, and backtesting identified pairs.
- **`data`**: Contains the aggregate data used for the analysis. Raw data from Binance is excluded but can be generated from scratch using the scripts provided in `src/data`.
- **[`src/data`](src/data/README.md)**: Includes scripts for downloading and organizing the raw data from Binance. See the [README](src/data/README.md) in this folder for a detailed guide on setting up the data.
- **`src/backtest`**: Contains the code for the backtesting framework.

## Requirements

To set up your environment, install the dependencies listed in `requirements.txt`.
