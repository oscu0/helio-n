from pathlib import Path

import pandas as pd

from Library.Paths import data_path, resolve_repo_path

DEFAULT_ICME_CSV_PATH = data_path("merged_icme_short.csv")


def load_icme_windows(icme_csv_path=DEFAULT_ICME_CSV_PATH):
    """Load ICME intervals from the repo CSV and normalize them to start/end windows."""
    csv_path = resolve_repo_path(icme_csv_path)
    assert csv_path.exists(), f"Missing ICME csv: {csv_path}"

    icme_df = pd.read_csv(csv_path).copy()

    icme_df["T_shock"] = pd.to_datetime(icme_df["T_shock"], errors="coerce")
    icme_df["T_start"] = pd.to_datetime(icme_df["T_start"], errors="coerce")
    icme_df["T_end"] = pd.to_datetime(icme_df["T_end"], errors="coerce")

    # Match the notebook convention: prefer the shock arrival if present.
    icme_df["start"] = icme_df["T_shock"].fillna(icme_df["T_start"])
    icme_df["end"] = icme_df["T_end"]
    icme_df = icme_df.dropna(subset=["start", "end"]).copy()

    return icme_df.sort_values("start").reset_index(drop=True)


def build_icme_mask(times, icme_windows=None, icme_csv_path=DEFAULT_ICME_CSV_PATH):
    """Return a boolean mask marking timestamps that fall inside any ICME interval."""
    timestamps = pd.to_datetime(pd.Index(times))
    mask = pd.Series(False, index=timestamps)

    windows = (
        load_icme_windows(icme_csv_path=icme_csv_path)
        if icme_windows is None
        else icme_windows.copy()
    )

    # Accept either pre-normalized start/end columns or the raw CSV column names.
    if "start" not in windows.columns or "end" not in windows.columns:
        windows["start"] = pd.to_datetime(
            windows.get("T_shock"), errors="coerce"
        ).fillna(pd.to_datetime(windows.get("T_start"), errors="coerce"))
        windows["end"] = pd.to_datetime(windows.get("T_end"), errors="coerce")
        windows = windows.dropna(subset=["start", "end"]).copy()

    for row in windows.itertuples(index=False):
        mask |= (timestamps >= row.start) & (timestamps <= row.end)

    return mask


def drop_icme_periods(
    df,
    datetime_col=None,
    icme_windows=None,
    icme_csv_path=DEFAULT_ICME_CSV_PATH,
):
    """Drop dataframe rows whose timestamps overlap an ICME interval."""
    df_out = df.copy()

    if datetime_col is None:
        # Default to treating the dataframe index itself as the event timeline.
        timestamps = pd.to_datetime(df_out.index)
        mask = build_icme_mask(
            timestamps, icme_windows=icme_windows, icme_csv_path=icme_csv_path
        )
        return df_out.loc[~mask.to_numpy()].copy()

    df_out[datetime_col] = pd.to_datetime(df_out[datetime_col])
    mask = build_icme_mask(
        df_out[datetime_col], icme_windows=icme_windows, icme_csv_path=icme_csv_path
    )
    return df_out.loc[~mask.to_numpy()].copy()
