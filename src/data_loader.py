
"""
Data loading utilities.
"""
import pandas as pd
from pathlib import Path

def load_covid_aggregated(csv_path: str) -> pd.DataFrame:
    """
    Load and aggregate the COVID-19 dataset by ObservationDate.
    Returns a dataframe with columns: ObservationDate, Confirmed, Deaths, Recovered, Active.
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    # Normalize the date column name if needed
    date_col = None
    for cand in ["ObservationDate", "Date", "date"]:
        if cand in df.columns:
            date_col = cand
            break
    if date_col is None:
        raise ValueError("No date column found in CSV. Expected one of: ObservationDate, Date, date")
    df[date_col] = pd.to_datetime(df[date_col])
    # Normalize core columns (handle alternate casing)
    def pick(colnames):
        for c in colnames:
            if c in df.columns:
                return c
        return None
    confirmed = pick(["Confirmed", "confirmed"])
    deaths = pick(["Deaths", "deaths"])
    recovered = pick(["Recovered", "recovered"])
    if confirmed is None or deaths is None or recovered is None:
        raise ValueError("Expected Confirmed/Deaths/Recovered columns in the dataset.")
    grouped = (
        df.groupby(df[date_col].dt.date)[[confirmed, deaths, recovered]]
          .sum()
          .reset_index()
          .rename(columns={confirmed:"Confirmed", deaths:"Deaths", recovered:"Recovered", date_col:"ObservationDate"})
    )
    grouped["ObservationDate"] = pd.to_datetime(grouped["ObservationDate"])
    grouped["Active"] = grouped["Confirmed"] - grouped["Deaths"] - grouped["Recovered"]
    grouped = grouped.sort_values("ObservationDate").reset_index(drop=True)
    return grouped
