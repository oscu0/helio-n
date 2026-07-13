import pandas as pd

from Library.Paths import PROJECT_ROOT, data_path, resolve_repo_path

GENERATED_ICME_CSV_PATH = (
    PROJECT_ROOT.parent / "ICME_list" / "out" / "site" / "merged_icme_short.csv"
)
DEFAULT_ICME_CSV_PATH = (
    GENERATED_ICME_CSV_PATH
    if GENERATED_ICME_CSV_PATH.exists()
    else data_path("merged_icme_short.csv")
)
# HELIO4CAST ICMECAT v2.3, archived as Figshare revision 24:
# https://doi.org/10.6084/m9.figshare.6356420.v24
DEFAULT_ICMECAT_CSV_PATH = data_path("HELIO4CAST_ICMECAT_v23.csv")
DEFAULT_POST_ICME_BODY_END_TOLERANCE = pd.Timedelta(hours=12)


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


def load_icmecat_windows(spacecraft, icmecat_csv_path=DEFAULT_ICMECAT_CSV_PATH):
    """Load HELIO4CAST ICMECAT full-event windows for one spacecraft."""
    csv_path = resolve_repo_path(icmecat_csv_path)
    assert csv_path.exists(), f"Missing ICMECAT csv: {csv_path}"

    icmecat_df = pd.read_csv(csv_path).copy()
    required_columns = {
        "icmecat_id",
        "sc_insitu",
        "icme_start_time",
        "mo_start_time",
        "mo_end_time",
        "icme_duration",
    }
    missing_columns = required_columns.difference(icmecat_df.columns)
    assert not missing_columns, (
        f"ICMECAT csv is missing columns: {sorted(missing_columns)}"
    )

    for column in ["icme_start_time", "mo_start_time", "mo_end_time"]:
        icmecat_df[column] = pd.to_datetime(
            icmecat_df[column], errors="coerce", utc=True
        ).dt.tz_convert(None)

    icmecat_df = icmecat_df.loc[icmecat_df["sc_insitu"] == spacecraft].copy()
    icmecat_df["start"] = icmecat_df["icme_start_time"]
    icmecat_df["end"] = icmecat_df["mo_end_time"]
    icmecat_df = icmecat_df.dropna(subset=["start", "end"])

    return icmecat_df.sort_values("start").reset_index(drop=True)


def build_icme_mask(
    times,
    icme_windows=None,
    icme_csv_path=DEFAULT_ICME_CSV_PATH,
    post_body_end_tolerance=DEFAULT_POST_ICME_BODY_END_TOLERANCE,
    inclusive_end=True,
):
    """Return a boolean mask marking timestamps inside ICME intervals plus end tolerance."""
    timestamps = pd.to_datetime(pd.Index(times))
    mask = pd.Series(False, index=timestamps)
    if post_body_end_tolerance is None:
        post_body_end_tolerance = pd.Timedelta(0)
    else:
        post_body_end_tolerance = pd.Timedelta(post_body_end_tolerance)

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

    windows["start"] = pd.to_datetime(windows["start"])
    windows["end"] = pd.to_datetime(windows["end"]) + post_body_end_tolerance

    for row in windows.itertuples(index=False):
        if inclusive_end:
            mask |= (timestamps >= row.start) & (timestamps <= row.end)
        else:
            mask |= (timestamps >= row.start) & (timestamps < row.end)

    return mask


def drop_icme_periods(
    df,
    datetime_col=None,
    icme_windows=None,
    icme_csv_path=DEFAULT_ICME_CSV_PATH,
    post_body_end_tolerance=DEFAULT_POST_ICME_BODY_END_TOLERANCE,
    inclusive_end=True,
):
    """Drop rows whose timestamps overlap an ICME interval plus end tolerance."""
    df_out = df.copy()

    if datetime_col is None:
        # Default to treating the dataframe index itself as the event timeline.
        timestamps = pd.to_datetime(df_out.index)
        mask = build_icme_mask(
            timestamps,
            icme_windows=icme_windows,
            icme_csv_path=icme_csv_path,
            post_body_end_tolerance=post_body_end_tolerance,
            inclusive_end=inclusive_end,
        )
        return df_out.loc[~mask.to_numpy()].copy()

    df_out[datetime_col] = pd.to_datetime(df_out[datetime_col])
    mask = build_icme_mask(
        df_out[datetime_col],
        icme_windows=icme_windows,
        icme_csv_path=icme_csv_path,
        post_body_end_tolerance=post_body_end_tolerance,
        inclusive_end=inclusive_end,
    )
    return df_out.loc[~mask.to_numpy()].copy()
