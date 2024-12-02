import os
import zipfile
import pandas as pd
from tqdm import *


def aggregate_zip_csvs(base_dir, time_frame):
    df_list = []
    for pair in tqdm(os.listdir(base_dir), desc="Pair for " + time_frame, ncols=100):
        object_dir = os.path.join(base_dir, pair, time_frame)

        if not os.path.isdir(object_dir):
            continue

        for zip_file in os.listdir(object_dir):
            if zip_file.endswith(".zip"):
                zip_path = os.path.join(object_dir, zip_file)
                with zipfile.ZipFile(zip_path, "r") as z:
                    for file_name in z.namelist():
                        if file_name.endswith(".csv"):
                            with z.open(file_name) as f:
                                df = pd.read_csv(
                                    f, header=None, usecols=[4, 6, 7]
                                ).rename(
                                    columns={
                                        4: "ClosePrice",
                                        6: "CloseTime",
                                        7: "QuoteAssetVolume",
                                    }
                                )

                                df["Pair"] = pair
                                df = df[
                                    [
                                        "Pair",
                                        "ClosePrice",
                                        "CloseTime",
                                        "QuoteAssetVolume",
                                    ]
                                ]

                                df_list.append(df)

    base_path = os.path.join(base_dir, "../aggregate/")
    os.makedirs(base_path, exist_ok=True)
    filename = "output" + time_frame + ".csv"
    file_path = os.path.join(base_path, filename)
    final_df = pd.concat(df_list, ignore_index=True)

    try:
        final_df.to_csv(file_path, index=False)
        print(f"DataFrame saved to {file_path}")
    except Exception as e:
        print(f"Error saving DataFrame to CSV: {e}")


if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(__file__))
    base_directory = os.path.join(project_root, "../../data/klines")
    aggregate_zip_csvs(base_directory, "1D")
    aggregate_zip_csvs(base_directory, "1h")
