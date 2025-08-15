from datetime import datetime

import pandas as pd
import os


def main(input_path):
    output_path = f"{input_path.replace('.csv', '')}_marked.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = pd.read_csv(input_path)
    dates_df = df["Date"]
    datetimes = [datetime.fromisoformat(date).isocalendar() for date in dates_df]
    last_days_of_weeks_df = (pd.DataFrame(datetimes,
                                          columns=["year", "week_of_year",
                                                   "day_of_week"]).groupby(["year",
                                                                            "week_of_year"])["day_of_week"].max())
    last_one_years_weeks_df = pd.DataFrame(
        last_days_of_weeks_df.reset_index().apply(
            lambda row: datetime.strptime(f"{row['year']} {row['week_of_year']} {row['day_of_week']}", "%G %V %u"),
            axis=1,
        ),
        columns=["Date"],
    )
    last_one_years_weeks_df["Date"] = last_one_years_weeks_df["Date"].apply(lambda date: date.strftime("%Y-%m-%d"))
    last_one_years_weeks_df["split_point"] = True
    df_with_split_points = df.merge(last_one_years_weeks_df, how="left", on="Date").fillna(False)

    df_with_split_points.set_index(["Date"]).to_csv(output_path)


if __name__ == "__main__":
    main("out/crypto/daily_20_2190.csv")
