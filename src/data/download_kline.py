import sys
from datetime import *
import pandas as pd
from enums import *
from utility import (
    download_file,
    get_all_symbols,
    get_parser,
    convert_to_date_object,
    get_path,
)
import concurrent.futures


def download_monthly_klines(
    trading_type,
    symbols,
    intervals,
    years,
    months,
    start_date,
    end_date,
    folder,
):
    current = 0
    date_range = None

    if start_date and end_date:
        date_range = start_date + " " + end_date

    if not start_date:
        start_date = START_DATE
    else:
        start_date = convert_to_date_object(start_date)

    if not end_date:
        end_date = END_DATE
    else:
        end_date = convert_to_date_object(end_date)

    num_symbols = len(symbols)
    print("Found {} symbols".format(num_symbols))

    for symbol in symbols:
        print(
            "[{}/{}] - start download monthly {} klines ".format(
                current + 1, num_symbols, symbol
            )
        )
        for interval in intervals:
            for year in years:
                for month in months:
                    current_date = convert_to_date_object(
                        "{}-{}-01".format(year, month)
                    )
                    if current_date >= start_date and current_date <= end_date:
                        path = get_path(
                            trading_type, "klines", "monthly", symbol, interval
                        )
                        file_name = "{}-{}-{}-{}.zip".format(
                            symbol.upper(), interval, year, "{:02d}".format(month)
                        )
                        yield path, file_name, date_range, folder
                        # download_file(path, file_name, date_range, folder)
        break
        current += 1


def download_monthly_klines_concurrently(
    trading_type,
    symbols,
    intervals,
    years,
    months,
    start_date,
    end_date,
    folder,
):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(download_file, path, file_name, date_range, folder)
            for path, file_name, date_range, folder in download_monthly_klines(
                trading_type,
                symbols,
                intervals,
                years,
                months,
                start_date,
                end_date,
                folder,
            )
        ]
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()  # This will raise exceptions if any occur during download
            except Exception as exc:
                print(f"Error: {exc}")


if __name__ == "__main__":
    parser = get_parser("klines")
    args = parser.parse_args(sys.argv[1:])

    if not args.symbols:
        print("fetching all symbols from exchange")
        symbols = get_all_symbols(args.type)
        num_symbols = len(symbols)
    else:
        symbols = args.symbols
        num_symbols = len(symbols)

    if args.dates:
        dates = args.dates
    else:
        period = convert_to_date_object(
            datetime.today().strftime("%Y-%m-%d")
        ) - convert_to_date_object(PERIOD_START_DATE)
        dates = (
            pd.date_range(end=datetime.today(), periods=period.days + 1)
            .to_pydatetime()
            .tolist()
        )
        dates = [date.strftime("%Y-%m-%d") for date in dates]

    # print(args, symbols)

    interesting_symbols = [
        symbol for symbol in symbols if symbol in INTERESTING_SYMBOLS
    ]

    if args.skip_monthly == 0:
        download_monthly_klines_concurrently(
            args.type,
            interesting_symbols,
            args.intervals,
            args.years,
            args.months,
            args.startDate,
            args.endDate,
            args.folder,
        )
