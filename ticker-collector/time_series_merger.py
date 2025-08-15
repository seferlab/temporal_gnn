import os
import sys
from dataclasses import dataclass
from functools import reduce

import pandas as pd


@dataclass
class StockDF:
    name: str
    df: pd.DataFrame

    def copy(self, df):
        return StockDF(self.name, df)


def main(input_path, output_path, frequency, num_assets, b_fill_days=None):
    paths = next(os.walk(f"{input_path}-{frequency}"))[2]
    dfs = sorted(
        [StockDF(
            remove_csv_extension(path),
            pd.read_csv(f"{input_path}-{frequency}/{path}"),
        ) for path in paths],
        key=lambda stock_df: len(stock_df.df),
    )[-num_assets:]
    merged_df = (merge_by_bfilling(dfs, b_fill_days) if b_fill_days else merge_by_inner_join(dfs))
    os.makedirs(output_path, exist_ok=True)
    target_path = f"{output_path}/{frequency}_{num_assets}_{len(merged_df)}.csv"
    print(f"Target path: {target_path}")
    merged_df.to_csv(target_path)


def remove_csv_extension(file_name):
    return file_name.replace(".csv", "")


def merge_by_bfilling(dfs, b_fill_days):
    dfs_aligned = [stock_df.copy(df=stock_df.df[-b_fill_days:]) for stock_df in dfs]
    dfs_close = [
        stock_df.df[["Date", "Close"]].rename(columns={
            "Close": stock_df.name
        }).set_index("Date") for stock_df in dfs_aligned
    ]
    merged_df = reduce(lambda df1, df2: df1.merge(df2, on="Date", how="outer"), dfs_close)
    return merged_df.sort_values(by="Date").fillna(method="bfill", axis=0)


def merge_by_inner_join(dfs):
    number_of_days = sorted(list(map(lambda stock_df: len(stock_df.df), dfs)))[0]
    dfs_aligned = [stock_df.copy(df=stock_df.df[-number_of_days:]) for stock_df in dfs]
    dfs_close = [
        stock_df.df[["Date", "Close"]].rename(columns={
            "Close": stock_df.name
        }).set_index("Date") for stock_df in dfs_aligned
    ]
    return reduce(lambda df1, df2: df1.merge(df2, on="Date"), dfs_close)


if __name__ == "__main__":
    input_path, output_path, frequency, num_assets, b_fill_days = sys.argv[1:]
    main(input_path, output_path, frequency, int(num_assets), int(b_fill_days))
