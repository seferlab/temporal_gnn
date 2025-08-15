from datetime import datetime, timedelta

import pandas as pd


def main(input_file):
    df = pd.read_csv(input_file)

    all_weekdays = generate_all_weekdays(
        datetime.strptime(df.iloc[0]["Date"], "%Y-%m-%d"),
        datetime.strptime(df.iloc[-1]["Date"], "%Y-%m-%d"),
    )
    all_weekdays_df = pd.DataFrame({"Date": all_weekdays})

    merged_df = pd.merge(df, all_weekdays_df, on="Date", how="right")
    filled_df = merged_df.fillna(method="ffill")

    df.to_csv(input_file.replace(".csv", "_with_missing_days.csv"), index=False)
    filled_df.to_csv(input_file, index=False)


def generate_all_weekdays(start_date, end_date):
    all_weekdays = []
    current_date = start_date
    while current_date <= end_date:
        if current_date.weekday() < 5:
            all_weekdays.append(current_date.strftime("%Y-%m-%d"))
        current_date += timedelta(days=1)

    return all_weekdays
