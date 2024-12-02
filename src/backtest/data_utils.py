import datetime
import pandas as pd


def clean_data(
    data: pd.DataFrame,
    start_date: datetime,
    end_date: datetime,
    ticker1: str,
    ticker2: str,
    date_column="CloseTime",
    pair_column="Pair",
) -> pd.DataFrame:
    """Returns data that is within the start_date and end_date and relevant to the two tickers provided"""
    new_data = data[
        (data[date_column] >= start_date) & (data[date_column] <= end_date)
    ].copy()
    return new_data[
        (new_data[pair_column] == ticker1) | (new_data[pair_column] == ticker2)
    ]


def check_data_valid(
    data: pd.DataFrame,
    start_date: datetime,
    end_date: datetime,
    date_column="CloseTime",
    pair_column="Pair",
    price_column="ClosePrice",
) -> bool:
    """Check data for missing entries"""
    helper = data.copy()

    helper["normalized_date"] = helper[date_column].dt.normalize()
    complete_date_range = pd.date_range(start=start_date, end=end_date, closed="left")

    pivot_table = helper.pivot_table(
        index="normalized_date",
        columns=pair_column,
        values=price_column,
        aggfunc="first",
    )
    pivot_table = pivot_table.reindex(complete_date_range)

    missing_dates = pivot_table.index[pivot_table.isnull().any(axis=1)]
    missing_pairs_by_date = pivot_table[pivot_table.isnull().any(axis=1)]

    if missing_dates.empty and missing_pairs_by_date.empty:
        return True
    else:
        print("Missing entries found:")
        if not missing_dates.empty:
            print("\nMissing Dates:")
            print(missing_dates)
        if not missing_pairs_by_date.empty:
            print("\nDates with missing pairs:")
            print(missing_pairs_by_date)
        return False
