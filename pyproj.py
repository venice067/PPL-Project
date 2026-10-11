import numpy as np
import pandas as pd

WEATHER_COLS = ["MaxT", "MinT", "RH1(%)", "RH2(%)", "RF(mm)", "WS(kmph)", "SSH(hrs)", "EVP(mm)"]
STATION_WEEK = ["Location", "Observation Year", "Standard Week"]
PEST_SERIES = ["PEST NAME", "Location", "Collection Type"]
OBSERVATION = PEST_SERIES + ["Observation Year", "Standard Week"]

LIMITS = {
    "MaxT": (5, 50),
    "MinT": (-5, 35),
    "RH1(%)": (0, 100),
    "RH2(%)": (0, 100),
    "RF(mm)": (0, 600),
    "WS(kmph)": (0, 60),
    "SSH(hrs)": (0, 14),
    "EVP(mm)": (0, 20),
}


def load_data(path):
    df = pd.read_csv(path)
    print("Loaded:", df.shape)
    return df


def check_data(df):
    print("\n--- Checks on raw data ---")
    print("Missing values:", df.isna().sum().sum())
    print("Duplicate rows:", df.duplicated().sum())
    print("Duplicate observations (same pest/station/week):", df.duplicated(OBSERVATION).sum())
    print("Highest MaxT:", df["MaxT"].max(), "| Highest sunshine hours:", df["SSH(hrs)"].max())

    maruteru = df[df["Location"] == "Maruteru"]
    print("\nMaruteru average EVP by year (look at 1994-1997 vs later):")
    print(maruteru.groupby("Observation Year")["EVP(mm)"].mean().round(1).head(8))


def fix_evaporation_units(df):
    early = (df["Location"] == "Maruteru") & (
        (df["Observation Year"] < 1998)
        | ((df["Observation Year"] == 1998) & (df["Standard Week"] <= 10))
    )
    df.loc[early, "EVP(mm)"] = df.loc[early, "EVP(mm)"] / 7
    print("EVP rows divided by 7:", early.sum())

    bad_years = (df["Location"] == "Palampur") & df["Observation Year"].isin([1984, 1989])
    df.loc[bad_years, "EVP(mm)"] = np.nan
    print("EVP rows set to NaN (Palampur 1984/1989):", bad_years.sum())
    return df


def remove_impossible_values(df):
    for col in LIMITS:
        low, high = LIMITS[col]
        bad = (df[col] < low) | (df[col] > high)
        df.loc[bad, col] = np.nan
        if bad.sum() > 0:
            print(f"{col}: {bad.sum()} impossible values set to NaN")

    wrong = df["MinT"] > df["MaxT"]
    df.loc[wrong, ["MinT", "MaxT"]] = np.nan
    print("MinT > MaxT rows:", wrong.sum())

    for col in ["WS(kmph)", "EVP(mm)", "SSH(hrs)"]:
        year_max = df.groupby(["Location", "Observation Year"])[col].transform("max")
        not_measured = year_max == 0
        df.loc[not_measured, col] = np.nan
        print(f"{col}: {not_measured.sum()} zeros from unmeasured years set to NaN")
    return df


def remove_duplicates(df):
    before = len(df)
    columns_to_average = ["Pest Value"] + WEATHER_COLS
    df = df.groupby(OBSERVATION, as_index=False)[columns_to_average].mean()
    print("Duplicate rows merged:", before - len(df))
    return df


def fix_weather(df):
    weather = df.groupby(STATION_WEEK, as_index=False)[WEATHER_COLS].median()
    weather["Weather_Imputed"] = weather[WEATHER_COLS].isna().any(axis=1)

    typical = weather.groupby(["Location", "Standard Week"])[WEATHER_COLS].transform("median")
    weather[WEATHER_COLS] = weather[WEATHER_COLS].fillna(typical).round(1)
    print("Station-weeks with filled weather values:", weather["Weather_Imputed"].sum())

    df = df.drop(columns=WEATHER_COLS)
    df = df.merge(weather, on=STATION_WEEK, how="left")
    return df


def flag_pest_outliers(df):
    positive = df[df["Pest Value"] > 0]
    quartiles = positive.groupby(PEST_SERIES)["Pest Value"].quantile([0.25, 0.75]).unstack()
    quartiles = quartiles.reset_index()
    quartiles.columns = PEST_SERIES + ["Q1", "Q3"]

    df = df.merge(quartiles, on=PEST_SERIES, how="left")
    upper_limit = df["Q3"] + 3 * (df["Q3"] - df["Q1"])
    df["Pest_Outlier"] = df["Pest Value"] > upper_limit
    df["Pest_Value_log"] = np.log1p(df["Pest Value"])
    df = df.drop(columns=["Q1", "Q3"])
    print("Pest outliers flagged (kept):", df["Pest_Outlier"].sum())
    return df


def add_new_columns(df):
    january_1 = pd.to_datetime(df["Observation Year"].astype(str) + "-01-01")
    df["Week_Start_Date"] = january_1 + pd.to_timedelta((df["Standard Week"] - 1) * 7, unit="D")
    df["Month"] = df["Week_Start_Date"].dt.month
    df["Quarter"] = df["Week_Start_Date"].dt.quarter

    in_kharif = df["Standard Week"].between(23, 44)
    df["Season"] = np.where(in_kharif, "Kharif", "Rabi/Summer")

    df["Mean_Temp"] = (df["MaxT"] + df["MinT"]) / 2
    df["Temp_Range"] = df["MaxT"] - df["MinT"]
    df["RH_Mean"] = (df["RH1(%)"] + df["RH2(%)"]) / 2

    df["Pest_Present"] = (df["Pest Value"] > 0).astype(int)
    is_percent = df["Collection Type"].isin(["Percentage", "Percent Damage"])
    df["Pest_Unit"] = np.where(is_percent, "percent", "count")

    df = df.sort_values(PEST_SERIES + ["Observation Year", "Standard Week"])
    df = df.reset_index(drop=True)
    week_number = df["Observation Year"] * 52 + df["Standard Week"]
    previous_week_number = week_number.groupby([df[c] for c in PEST_SERIES]).shift(1)
    is_previous_week = (week_number - previous_week_number) == 1

    for col in ["MaxT", "RF(mm)", "RH1(%)", "Pest_Value_log"]:
        previous_value = df.groupby(PEST_SERIES)[col].shift(1)
        df[col + "_lastweek"] = previous_value.where(is_previous_week)
    return df


def make_summaries(df):
    by_pest = df.groupby("PEST NAME").agg(
        records=("Pest Value", "size"),
        stations=("Location", "nunique"),
        pct_weeks_present=("Pest_Present", "mean"),
        median_value=("Pest Value", "median"),
        max_value=("Pest Value", "max"),
        outliers=("Pest_Outlier", "sum"),
    )
    by_pest["pct_weeks_present"] = (by_pest["pct_weeks_present"] * 100).round(1)

    by_season = df.groupby(["PEST NAME", "Season"])["Pest_Present"].mean() * 100
    by_season = by_season.round(1).unstack()
    return by_pest, by_season


def final_check(df):
    assert df[WEATHER_COLS].isna().sum().sum() == 0, "Weather still has missing values"
    assert (df["MinT"] <= df["MaxT"]).all(), "MinT is higher than MaxT somewhere"
    assert not df.duplicated(OBSERVATION).any(), "Duplicates remain"
    for col in LIMITS:
        low, high = LIMITS[col]
        assert df[col].between(low, high).all(), col + " has impossible values"
    print("\nFinal check passed.")


def main():
    df = load_data("RICE.csv")
    check_data(df)

    print("\n--- Cleaning ---")
    df = fix_evaporation_units(df)
    df = remove_impossible_values(df)
    df = remove_duplicates(df)
    df = fix_weather(df)
    df = flag_pest_outliers(df)

    print("\n--- Adding new columns ---")
    df = add_new_columns(df)
    final_check(df)

    by_pest, by_season = make_summaries(df)
    print("\nSummary by pest:\n", by_pest)
    print("\n% of weeks pest is present, by season:\n", by_season)

    df.to_csv("cleaned_rice_final.csv", index=False)
    by_pest.to_csv("summary_by_pest.csv")
    by_season.to_csv("summary_by_season.csv")
    print("\nSaved cleaned_rice_final.csv, summary_by_pest.csv, summary_by_season.csv")
    print("Final shape:", df.shape)
    print(df.iloc[:5, :8])


main()
