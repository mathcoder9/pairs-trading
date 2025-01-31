from datetime import *

YEARS = ["2020", "2021", "2022", "2023", "2024"]
INTERVALS = ["1h", "1d"]
DAILY_INTERVALS = ["1h", "1d"]
TRADING_TYPE = ["spot", "um", "cm"]
MONTHS = list(range(1, 13))
PERIOD_START_DATE = "2020-01-01"
BASE_URL = "https://data.binance.vision/"
START_DATE = date(int(YEARS[0]), MONTHS[0], 1)
END_DATE = datetime.date(datetime.now())
INTERESTING_SYMBOLS = {
    "BTCUSDT",
    "ETHUSDT",
    "USDTUSDT",
    "BNBUSDT",
    "SOLUSDT",
    "USDCUSDT",
    "XRPUSDT",
    "DOGEUSDT",
    "TRXUSDT",
    "TONUSDT",
    "ADAUSDT",
    "AVAXUSDT",
    "SHIBUSDT",
    "BCHUSDT",
    "LINKUSDT",
    "DOTUSDT",
    "NEARUSDT",
    "SUIUSDT",
    "LEOUSDT",
    "LTCUSDT",
    "DAIUSDT",
    "APTUSDT",
    "UNIUSDT",
    "PEPEUSDT",
    "TAOUSDT",
    "ICPUSDT",
    "FETUSDT",
    "KASUSDT",
    "XMRUSDT",
    "ETCUSDT",
    "XLMUSDT",
    "POLUSDT",
    "STXUSDT",
    "FDUSDUSDT",
    "RENDERUSDT",
    "WIFUSDT",
    "IMXUSDT",
    "OKBUSDT",
    "AAVEUSDT",
    "FILUSDT",
    "INJUSDT",
    "OPUSDT",
    "CROUSDT",
    "MNTUSDT",
    "ARBUSDT",
    "FTMUSDT",
    "HBARUSDT",
    "VETUSDT",
    "BONKUSDT",
    "ATOMUSDT",
    "RUNEUSDT",
    "SEIUSDT",
    "GRTUSDT",
    "BGBUSDT",
    "FLOKIUSDT",
    "WLDUSDT",
    "TIAUSDT",
    "THETAUSDT",
    "OMUSDT",
    "POPCATUSDT",
    "PYTHUSDT",
    "ARUSDT",
    "JUPUSDT",
    "ENAUSDT",
    "ONDOUSDT",
    "HNTUSDT",
    "KCSUSDT",
    "BRETTUSDT",
    "MKRUSDT",
    "ALGOUSDT",
    "LDOUSDT",
    "BSVUSDT",
    "MATICUSDT",
    "JASMYUSDT",
    "BTTUSDT",
    "AEROUSDT",
    "COREUSDT",
    "BEAMUSDT",
    "FLOWUSDT",
    "NOTUSDT",
    "GTUSDT",
    "NEIROUSDT",
    "GALAUSDT",
    "QNTUSDT",
    "MOGUSDT",
    "MEWUSDT",
    "AXSUSDT",
    "STRKUSDT",
    "WUSDT",
    "ORDIUSDT",
    "PENDLEUSDT",
    "USDDUSDT",
    "EOSUSDT",
    "NEOUSDT",
    "FLRUSDT",
    "EGLDUSDT",
    "CFXUSDT",
    "XECUSDT",
    "XTZUSDT",
    "FTTUSDT",
}
