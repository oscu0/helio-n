from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import Workbook, load_workbook
from openpyxl.comments import Comment
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

from Library.SW.Stats import build_eval_index, build_score_frames


REPORT_COLUMNS = [
    (
        "ch_relative_area",
        ("input", "ch_area", "ch_relative_area"),
        "mean",
        "Средняя за час относительная площадь корональной дыры во входных данных модели. Время относится к исходному времени у Солнца, поэтому эту величину нельзя строка-в-строку считать входом прогноза у КА с той же временной меткой. Контекстная величина, в статистике качества не используется.",
        "0.000000",
    ),
    (
        "v_empirical",
        ("input", "model_input", "v_empirical"),
        "mean",
        "Средняя за час эмпирическая скорость солнечного ветра у Солнца, км/с. Время относится к запуску потока у Солнца, поэтому эту величину нельзя строка-в-строку считать входом прогноза у КА с той же временной меткой. Контекстная величина, в статистике качества не используется.",
        "0.0",
    ),
    (
        "ace_v_real",
        ("analysis", "ace", "v_real"),
        "mean",
        "Средняя за час наблюдаемая скорость солнечного ветра ACE, км/с. Пропуски исходных наблюдений оставлены пустыми.",
        "0.0",
    ),
    (
        "ace_v_predict_raw",
        ("analysis", "ace", "v_predict_raw"),
        "mean",
        "Средний за час исходный прогноз для ACE с базовым медленным солнечным ветром 300 км/с, до динамической постобработки, км/с.",
        "0.0",
    ),
    (
        "ace_v_predict_slow",
        ("analysis", "ace", "v_predict"),
        "mean",
        "Средний за час прогноз модели для ACE с постобработкой медленного солнечного ветра, км/с. Эта серия сравнивается с SWX.",
        "0.0",
    ),
    (
        "ace_slow_sw_patch_mask",
        ("analysis", "ace", "slow_sw_patch_mask"),
        "max",
        "TRUE, если в этом часовом интервале к прогнозу ACE применена постобработка медленного солнечного ветра.",
        "General",
    ),
    (
        "ace_v_1cr_ago",
        ("analysis", "ace", "v_1cr_ago"),
        "mean",
        "Рекуррентный прогноз ACE: наблюдаемая скорость за один оборот Каррингтона до текущего времени, км/с. Линейная интерполяция применяется только между исходными наблюдениями, разделенными не более чем 6 часами; более длинные пропуски оставлены пустыми.",
        "0.0",
    ),
    (
        "ace_v_swx",
        ("analysis", "ace", "v_swx"),
        "mean",
        "Средний за час прогноз SWX для ACE, км/с, восстановленный из замороженной нативной серии прогнозов без интерполяции пропусков.",
        "0.0",
    ),
    (
        "ace_is_icme",
        ("analysis", "ace", "is_icme"),
        "max",
        "TRUE, если часовой интервал пересекается с полуоткрытым интервалом ICME у Земли или 12-часовым интервалом после его конца. Та же маска используется в статистике статьи.",
        "General",
    ),
    (
        "stereo_a_v_real",
        ("analysis", "stereo_a", "v_real"),
        "mean",
        "Средняя за час наблюдаемая скорость солнечного ветра STEREO-A, км/с. Промежутки до 6 часов между непустыми часовыми отсчётами интерполированы; более длинные пропуски оставлены пустыми.",
        "0.0",
    ),
    (
        "stereo_a_v_predict_raw",
        ("analysis", "stereo_a", "v_predict_raw"),
        "mean",
        "Средний за час сырой прогноз модели для STEREO-A, км/с. Модель медленного солнечного ветра к STEREO-A не применяется.",
        "0.0",
    ),
    (
        "stereo_a_v_1cr_ago",
        ("analysis", "stereo_a", "v_1cr_ago"),
        "mean",
        "Рекуррентный прогноз STEREO-A: наблюдаемая скорость за один оборот Каррингтона до текущего времени, км/с. Линейная интерполяция применяется только между исходными наблюдениями, разделенными не более чем 6 часами; более длинные пропуски оставлены пустыми.",
        "0.0",
    ),
    (
        "stereo_a_is_icme",
        ("analysis", "stereo_a", "is_icme"),
        "max",
        "TRUE, если часовой интервал пересекается с полуоткрытым интервалом ICME у STEREO-A или 12-часовым интервалом после его конца. Та же маска используется в статистике статьи.",
        "General",
    ),
]

CSV_COLUMNS = [
    "datetime_utc",
    "ace_ch_area_relative",
    "ace_forecast_speed_km_s",
    "ace_forecast_speed_300_km_s",
    "ace_speed_km_s",
    "ace_density_cm3",
    "ace_temperature_K",
    "ace_b_nT",
    "ace_bx_gse_nT",
    "ace_by_gse_nT",
    "ace_bz_gse_nT",
    "ace_br_rtn_nT",
    "ace_bt_rtn_nT",
    "ace_bn_rtn_nT",
    "stereo_ch_area_relative",
    "stereo_forecast_speed_km_s",
    "stereo_speed_km_s",
    "stereo_density_cm3",
    "stereo_temperature_K",
    "stereo_b_nT",
    "stereo_bx_gse_nT",
    "stereo_by_gse_nT",
    "stereo_bz_gse_nT",
    "stereo_br_rtn_nT",
    "stereo_bt_rtn_nT",
    "stereo_bn_rtn_nT",
    "stereo_hee_latitude_deg",
    "stereo_hee_longitude_deg",
    "stereo_hee_relative_to_ace_latitude_deg",
    "stereo_hee_relative_to_ace_longitude_deg",
]


def build_hourly_report_frame(
    reproduction_frame,
    comparison_frames,
    start_dt,
    end_dt,
    freq="1h",
):
    assert "ace" in comparison_frames, (
        "This report requires native ACE ('ace'); ACE-at-Earth is a different product"
    )
    report_index = build_eval_index(start_dt=start_dt, end_dt=end_dt, freq=freq)
    run_frame = reproduction_frame.copy()
    run_frame.index = pd.to_datetime(run_frame.index)
    run_frame = run_frame.loc[
        (run_frame.index >= pd.Timestamp(start_dt))
        & (run_frame.index < pd.Timestamp(end_dt))
    ]
    assert not run_frame.empty, (
        f"No rows in exact run window [{start_dt}, {end_dt}) after loading "
        "the matching reproduction parquet"
    )

    required_input_columns = [
        source_column
        for _name, source_column, _agg, _description, _number_format in REPORT_COLUMNS
        if source_column[0] == "input"
    ]
    missing_input_columns = [
        source_column
        for source_column in required_input_columns
        if source_column not in run_frame.columns
    ]
    assert not missing_input_columns, (
        f"Missing reproduction input columns: {missing_input_columns}"
    )

    score_frames = build_score_frames(
        comparison_frames=comparison_frames,
        start_dt=start_dt,
        end_dt=end_dt,
        freq=freq,
    )
    report_frame = pd.DataFrame(index=report_index)
    for name, source_column, aggregation, _description, _number_format in REPORT_COLUMNS:
        source_kind = source_column[0]
        if source_kind == "input":
            report_frame[name] = (
                run_frame[source_column]
                .resample(freq)
                .agg(aggregation)
                .reindex(report_index)
            )
        else:
            assert source_kind == "analysis"
            _source_kind, sat_name, column = source_column
            report_frame[name] = score_frames[sat_name][column].reindex(report_index)

    report_frame.insert(0, "time", report_frame.index)
    return report_frame


def _hourly_measurement(frame, column, report_index, freq, aggregation="mean"):
    if column not in frame.columns:
        return pd.Series(np.nan, index=report_index, dtype=float)
    source = pd.to_numeric(frame[column], errors="coerce")
    return source.resample(freq).agg(aggregation).reindex(report_index)


def _relative_hee_angles(stereo_frame, ace_frame, report_index, freq):
    coordinate_columns = ["x_hee_au", "y_hee_au", "z_hee_au"]
    stereo = pd.DataFrame(
        {
            column: _hourly_measurement(stereo_frame, column, report_index, freq)
            for column in coordinate_columns
        },
        index=report_index,
    )
    ace = pd.DataFrame(
        {
            column: _hourly_measurement(ace_frame, column, report_index, freq)
            for column in coordinate_columns
        },
        index=report_index,
    )
    relative = stereo - ace
    radius = np.sqrt((relative ** 2).sum(axis=1))
    longitude = np.mod(np.degrees(np.arctan2(relative["y_hee_au"], relative["x_hee_au"])), 360.0)
    longitude = ((longitude + 180.0) % 360.0) - 180.0
    latitude = np.degrees(
        np.arcsin(
            np.divide(
                relative["z_hee_au"],
                radius,
                out=np.full(len(relative), np.nan),
                where=radius.to_numpy(dtype=float) > 0.0,
            )
        )
    )
    return latitude, longitude


def build_hourly_csv_frame(
    reproduction_frame,
    comparison_frames,
    start_dt,
    end_dt,
    freq="1h",
):
    """Build the hourly CSV export described by the analysis data contract."""
    assert "ace" in comparison_frames, (
        "This report requires native ACE ('ace'); ACE-at-Earth is a different product"
    )
    report_index = build_eval_index(start_dt=start_dt, end_dt=end_dt, freq=freq)
    run_frame = reproduction_frame.copy()
    run_frame.index = pd.to_datetime(run_frame.index)
    run_frame = run_frame.loc[
        (run_frame.index >= pd.Timestamp(start_dt))
        & (run_frame.index < pd.Timestamp(end_dt))
    ]
    assert not run_frame.empty, (
        f"No rows in exact run window [{start_dt}, {end_dt}) after loading "
        "the matching reproduction parquet"
    )
    assert ("input", "ch_area", "ch_relative_area") in run_frame.columns, (
        "Missing reproduction input column ('input', 'ch_area', 'ch_relative_area')"
    )

    for sat, frame in comparison_frames.items():
        ambiguous = {"b_x", "b_y", "b_z"}.intersection(frame.columns)
        assert not ambiguous, (
            f"{sat}: magnetic components lack a frame: {sorted(ambiguous)}; "
            "reload observations with explicit GSE/RTN component names"
        )
    ace = comparison_frames["ace"]
    stereo = comparison_frames["stereo_a"]
    ch_area = _hourly_measurement(
        run_frame,
        ("input", "ch_area", "ch_relative_area"),
        report_index,
        freq,
    )
    stereo_lat, stereo_lon = _relative_hee_angles(
        stereo_frame=stereo,
        ace_frame=ace,
        report_index=report_index,
        freq=freq,
    )
    output = pd.DataFrame(index=report_index)
    output["datetime_utc"] = report_index
    output["ace_ch_area_relative"] = ch_area
    output["ace_forecast_speed_km_s"] = _hourly_measurement(
        ace, "v_predict", report_index, freq
    )
    output["ace_forecast_speed_300_km_s"] = _hourly_measurement(
        ace, "v_predict_raw", report_index, freq
    )
    for output_column, source_column in {
        "ace_speed_km_s": "v_real",
        "ace_density_cm3": "N",
        "ace_temperature_K": "t",
        "ace_b_nT": "b",
        "ace_bx_gse_nT": "b_x_gse",
        "ace_by_gse_nT": "b_y_gse",
        "ace_bz_gse_nT": "b_z_gse",
        "ace_br_rtn_nT": "b_r_rtn",
        "ace_bt_rtn_nT": "b_t_rtn",
        "ace_bn_rtn_nT": "b_n_rtn",
    }.items():
        output[output_column] = _hourly_measurement(
            ace, source_column, report_index, freq
        )
    output["stereo_ch_area_relative"] = ch_area
    output["stereo_forecast_speed_km_s"] = _hourly_measurement(
        stereo, "v_predict", report_index, freq
    )
    for output_column, source_column in {
        "stereo_speed_km_s": "v_real",
        "stereo_density_cm3": "N",
        "stereo_temperature_K": "t",
        "stereo_b_nT": "b",
        "stereo_bx_gse_nT": "b_x_gse",
        "stereo_by_gse_nT": "b_y_gse",
        "stereo_bz_gse_nT": "b_z_gse",
        "stereo_br_rtn_nT": "b_r_rtn",
        "stereo_bt_rtn_nT": "b_t_rtn",
        "stereo_bn_rtn_nT": "b_n_rtn",
    }.items():
        output[output_column] = _hourly_measurement(
            stereo, source_column, report_index, freq
        )
    stereo_x = _hourly_measurement(stereo, "x_hee_au", report_index, freq)
    stereo_y = _hourly_measurement(stereo, "y_hee_au", report_index, freq)
    stereo_z = _hourly_measurement(stereo, "z_hee_au", report_index, freq)
    stereo_radius = np.sqrt(stereo_x ** 2 + stereo_y ** 2 + stereo_z ** 2)
    output["stereo_hee_latitude_deg"] = np.degrees(
        np.arcsin(
            np.divide(
                stereo_z,
                stereo_radius,
                out=np.full(len(stereo_radius), np.nan),
                where=stereo_radius.to_numpy(dtype=float) > 0.0,
            )
        )
    )
    output["stereo_hee_longitude_deg"] = np.mod(
        np.degrees(np.arctan2(stereo_y, stereo_x)), 360.0
    )
    output["stereo_hee_relative_to_ace_latitude_deg"] = stereo_lat
    output["stereo_hee_relative_to_ace_longitude_deg"] = stereo_lon
    return output[CSV_COLUMNS]


def write_csv(report_frame, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = report_frame.copy()
    frame.to_csv(output_path, index=False, date_format="%Y-%m-%dT%H:%M:%SZ")
    loaded = pd.read_csv(output_path)
    assert list(loaded.columns) == list(frame.columns)
    assert len(loaded) == len(frame)
    return output_path


def write_report(report_frame, output_path):
    workbook = Workbook()
    worksheet = workbook.active
    worksheet.title = "Data"

    header_fill = PatternFill("solid", fgColor="1F4E78")
    header_font = Font(color="FFFFFF", bold=True)

    header_specs = [
        (
            "time",
            "Начало часового интервала, UTC. Диапазон отчета полуоткрытый: начало включено, конец не включен.",
            "yyyy-mm-dd hh:mm",
        )
    ] + [
        (name, description, number_format)
        for name, _source_column, _aggregation, description, number_format in REPORT_COLUMNS
    ]

    for column_index, (name, description, _number_format) in enumerate(
        header_specs, start=1
    ):
        cell = worksheet.cell(row=1, column=column_index, value=name)
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = Alignment(
            horizontal="center",
            vertical="center",
            wrap_text=True,
        )
        cell.comment = Comment(description, "Codex")

    for row_index, row in enumerate(
        report_frame.itertuples(index=False, name=None), start=2
    ):
        for column_index, value in enumerate(row, start=1):
            if pd.isna(value):
                value = None
            elif isinstance(value, pd.Timestamp):
                value = value.to_pydatetime()
            elif hasattr(value, "item"):
                value = value.item()
            worksheet.cell(row=row_index, column=column_index, value=value)

    worksheet.freeze_panes = "A2"
    worksheet.auto_filter.ref = (
        f"A1:{get_column_letter(len(header_specs))}{len(report_frame) + 1}"
    )
    worksheet.row_dimensions[1].height = 38

    for column_index, (name, _description, number_format) in enumerate(
        header_specs, start=1
    ):
        column_letter = get_column_letter(column_index)
        if name == "time":
            width = 20
        elif name.endswith("_mask") or name.endswith("_is_icme"):
            width = 23
        else:
            width = min(max(len(name) + 2, 16), 24)
        worksheet.column_dimensions[column_letter].width = width

        for cell in worksheet[column_letter][1:]:
            cell.number_format = number_format

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(output_path)

    loaded = load_workbook(output_path, read_only=False, data_only=True)
    loaded_sheet = loaded["Data"]
    expected_headers = [name for name, _description, _format in header_specs]
    actual_headers = [
        loaded_sheet.cell(row=1, column=column_index).value
        for column_index in range(1, len(header_specs) + 1)
    ]
    assert actual_headers == expected_headers
    assert loaded_sheet.max_row == len(report_frame) + 1
    assert loaded_sheet.max_column == len(header_specs)
    assert loaded_sheet["A2"].value == report_frame.iloc[0]["time"].to_pydatetime()
    assert loaded_sheet["A1"].comment.text.startswith("Начало часового интервала")
    assert "строка-в-строку" in loaded_sheet["B1"].comment.text
    assert "строка-в-строку" in loaded_sheet["C1"].comment.text
    loaded.close()
    return output_path
