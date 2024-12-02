# Instructions to get data

#### [Originally from https://github.com/binance/binance-public-data/tree/master/python with some changes.]

## Data to collect

EOD close price and volume data for tickers that we are interested in.

## Steps

- Pick tickers
- Retrieve data for those tickers
- Process the data into CSV

## Picking Tickers

To begin, update `INTERESTING_SYMBOLS` list in `enums.py` with the symbols to consider.

I ran the script below on https://coinmarketcap.com/ to get the top 100 cryptos by mcap.

```
// Create an array to hold the cryptocurrency symbols
let cryptoSymbols = [];

// Select all elements with the specific class for cryptocurrency symbols
const symbolElements = document.querySelectorAll('.coin-item-symbol'); // Use the class name directly

// Loop through each element and extract the text content
symbolElements.forEach(element => {
    cryptoSymbols.push(element.innerText); // Get the text content (symbol)
});

// Log the symbols to the console
console.log(cryptoSymbols);
```

## Getting the daily and hourly data from Binance

From the project root, run:

```
python src/data/download_kline.py -t spot && mv src/data/data . && mv data/spot/monthly/klines/ data/ && rm -r data/spot
```

This will put the aggregated files in ~/data/klines.

## Aggregate the data into CSV and extract relevant data

From the project root:

```
cd src/data
python aggregate_data.py
```

This will put the aggregated files in ~/data/aggregate.
