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

SATELLITE_DATA_COLUMNS = {
    "time_utc": "datetime_utc",
    "ch_relative_area": "ch_area_relative",
    "v_predict_raw": "v_forecast",
    "v_predict": "v_forecast_slow_sw_model",
    "v_swx": "v_forecast_swx",
    "v_real": "v_observed",
    "N": "proton_density_cm3",
    "t": "proton_temperature_k",
    "b": "magnetic_field_magnitude_nt",
    "b_x_gse": "b_x_gse_nt",
    "b_y_gse": "b_y_gse_nt",
    "b_z_gse": "b_z_gse_nt",
    "x_hee_au": "x_hee_au",
    "y_hee_au": "y_hee_au",
    "z_hee_au": "z_hee_au",
    "hee_beta_deg": "hee_beta_deg",
    "relative_latitude_to_ace_deg": "relative_latitude_to_ace_deg",
    "relative_longitude_to_ace_deg": "relative_longitude_to_ace_deg",
    "sdo_observation_age_days": "sdo_observation_age_days",
    "sdo_age_over_10_days": "sdo_age_over_10_days",
    "hee_beta_over_10_deg": "hee_beta_over_10_deg",
}

VARIABLE_DEFINITIONS_SHEET = "Определения"
VARIABLE_DEFINITIONS_RU = [
    ("datetime_utc", "UTC", "Метка времени часового интервала в UTC."),
    (
        "ch_area_relative",
        "относительная площадь",
        "Относительная площадь корональной дыры во входных данных модели; это контекстный параметр, отнесённый ко времени у Солнца.",
    ),
    (
        "v_forecast",
        "км/с",
        "Исходный баллистический прогноз скорости до динамической поправки медленного ветра; в расчёте используется базовая скорость медленного ветра 300 км/с.",
    ),
    (
        "v_forecast_slow_sw_model",
        "км/с",
        "Прогноз после применения временной модели медленного солнечного ветра. Поправка применяется к ACE в отмеченных интервалах; для остальных спутников совпадает с v_forecast.",
    ),
    (
        "v_forecast_swx",
        "км/с",
        "Нативный прогноз SWX для ACE, сохранённый в архиве запуска. Пропуски не заполняются интерполяцией.",
    ),
    ("v_observed", "км/с", "Наблюдаемая скорость солнечного ветра у спутника."),
    ("proton_density_cm3", "см⁻³", "Плотность протонов солнечного ветра."),
    ("proton_temperature_k", "K", "Температура протонов солнечного ветра."),
    ("magnetic_field_magnitude_nt", "нТл", "Модуль магнитного поля."),
    ("b_x_gse_nt", "нТл", "Компонента X магнитного поля в системе GSE."),
    ("b_y_gse_nt", "нТл", "Компонента Y магнитного поля в системе GSE."),
    ("b_z_gse_nt", "нТл", "Компонента Z магнитного поля в системе GSE."),
    ("x_hee_au", "AU", "Координата X спутника в системе HEE."),
    ("y_hee_au", "AU", "Координата Y спутника в системе HEE."),
    ("z_hee_au", "AU", "Координата Z спутника в системе HEE."),
    (
        "hee_beta_deg",
        "градусы",
        "Широта спутника над эклиптикой в системе HEE: atan2(z, sqrt(x² + y²)).",
    ),
    (
        "relative_latitude_to_ace_deg",
        "градусы",
        "Широтное угловое положение спутника относительно ACE в системе HEE.",
    ),
    (
        "relative_longitude_to_ace_deg",
        "градусы",
        "Долготное угловое положение спутника относительно ACE в системе HEE.",
    ),
    (
        "sdo_observation_age_days",
        "сутки",
        "Сколько суток прошло с тех пор, как SDO в последний раз видел солнечную долготу, обращённую к спутнику; во время видимости возраст равен нулю.",
    ),
    (
        "sdo_age_over_10_days",
        "булево",
        "TRUE, если возраст наблюдения SDO превышает 10 суток. В таблице такие строки выделены жёлтым.",
    ),
    (
        "hee_beta_over_10_deg",
        "булево",
        "TRUE, если модуль широты над эклиптикой превышает 10°. В таблице такие строки выделены розовым.",
    ),
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


def build_satellite_data_frames(
    reproduction_frame,
    comparison_frames,
    start_dt,
    end_dt,
    freq="1h",
):
    """Build one plain hourly observation/forecast frame per archived satellite."""
    assert "ace" in comparison_frames, (
        "Satellite workbook requires native ACE ('ace') for relative coordinates"
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
    ch_area_column = ("input", "ch_area", "ch_relative_area")
    assert ch_area_column in run_frame.columns, (
        f"Missing reproduction input column {ch_area_column}"
    )
    ch_area = _hourly_measurement(run_frame, ch_area_column, report_index, freq)

    ace_frame = comparison_frames["ace"]
    result = {}
    for sat_name, source in comparison_frames.items():
        ambiguous = {"b_x", "b_y", "b_z"}.intersection(source.columns)
        assert not ambiguous, (
            f"{sat_name}: magnetic components lack a frame: {sorted(ambiguous)}"
        )
        frame = pd.DataFrame(index=report_index)
        frame["time_utc"] = report_index
        frame["ch_relative_area"] = ch_area
        for column in (
            "v_predict", "v_predict_raw", "v_real", "N", "t", "b",
            "b_x_gse", "b_y_gse", "b_z_gse", "x_hee_au", "y_hee_au",
            "z_hee_au", "hee_beta_deg", "sdo_observation_age_days",
        ):
            frame[column] = _hourly_measurement(source, column, report_index, freq)
        if sat_name == "ace":
            frame["v_swx"] = _hourly_measurement(source, "v_swx", report_index, freq)
        if sat_name == "ace" and {"x_hee_au", "y_hee_au", "z_hee_au"}.issubset(source.columns):
            has_position = pd.Series(
                np.isfinite(source[["x_hee_au", "y_hee_au", "z_hee_au"]]).all(axis=1),
                index=source.index,
            ).resample(freq).max().reindex(report_index).fillna(False)
            relative_lat = pd.Series(np.where(has_position, 0.0, np.nan), index=report_index)
            relative_lon = relative_lat.copy()
        elif {"x_hee_au", "y_hee_au", "z_hee_au"}.issubset(source.columns) and {
            "x_hee_au", "y_hee_au", "z_hee_au"
        }.issubset(ace_frame.columns):
            relative_lat, relative_lon = _relative_hee_angles(
                source, ace_frame, report_index, freq
            )
        else:
            relative_lat = pd.Series(np.nan, index=report_index)
            relative_lon = pd.Series(np.nan, index=report_index)
        frame["relative_latitude_to_ace_deg"] = relative_lat
        frame["relative_longitude_to_ace_deg"] = relative_lon
        frame["sdo_age_over_10_days"] = frame["sdo_observation_age_days"] > 10.0
        frame["hee_beta_over_10_deg"] = frame["hee_beta_deg"].abs() > 10.0
        columns = [
            column for column in SATELLITE_DATA_COLUMNS
            if column != "v_swx" or sat_name == "ace"
        ]
        result[sat_name] = frame[columns]
    return result


def write_satellite_data_workbook(satellite_frames, output_path):
    """Write plain per-satellite sheets with compact threshold highlighting."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for sat_name, frame in satellite_frames.items():
            sheet_name = str(sat_name).replace("/", "-")[:31]
            frame.rename(columns=SATELLITE_DATA_COLUMNS).to_excel(
                writer, sheet_name=sheet_name, index=False
            )
            worksheet = writer.book[sheet_name]
            worksheet.freeze_panes = "A2"
            worksheet.auto_filter.ref = worksheet.dimensions
            for cell in worksheet[1]:
                cell.font = Font(bold=True)
            headers = {cell.value: cell.column for cell in worksheet[1]}
            time_column = headers[SATELLITE_DATA_COLUMNS["time_utc"]]
            age_flag_column = headers[SATELLITE_DATA_COLUMNS["sdo_age_over_10_days"]]
            beta_flag_column = headers[SATELLITE_DATA_COLUMNS["hee_beta_over_10_deg"]]
            yellow = PatternFill(fill_type="solid", fgColor="FFFF00")
            pink = PatternFill(fill_type="solid", fgColor="F4CCCC")
            for row in range(2, worksheet.max_row + 1):
                worksheet.cell(row, time_column).number_format = "yyyy-mm-dd hh:mm"
                age_flag = bool(worksheet.cell(row, age_flag_column).value)
                beta_flag = bool(worksheet.cell(row, beta_flag_column).value)
                if age_flag or beta_flag:
                    row_fill = pink if beta_flag else yellow
                    for cell in worksheet[row]:
                        cell.fill = row_fill
                if age_flag:
                    worksheet.cell(row, age_flag_column).fill = yellow
                if beta_flag:
                    worksheet.cell(row, beta_flag_column).fill = pink
            worksheet.column_dimensions["A"].width = 21
            for column, variable in enumerate(frame.columns, start=2):
                worksheet.column_dimensions[get_column_letter(column)].width = min(
                    38, max(19, len(SATELLITE_DATA_COLUMNS[variable]) + 2)
                )

        definitions = pd.DataFrame(
            VARIABLE_DEFINITIONS_RU,
            columns=("Переменная", "Единица", "Описание (русский)"),
        )
        definitions.to_excel(writer, sheet_name=VARIABLE_DEFINITIONS_SHEET, index=False)
        worksheet = writer.book[VARIABLE_DEFINITIONS_SHEET]
        worksheet.freeze_panes = "A2"
        worksheet.auto_filter.ref = worksheet.dimensions
        for cell in worksheet[1]:
            cell.font = Font(bold=True)
        worksheet.column_dimensions["A"].width = 36
        worksheet.column_dimensions["B"].width = 22
        worksheet.column_dimensions["C"].width = 90
        worksheet.sheet_properties.pageSetUpPr.fitToPage = True
        worksheet.page_setup.orientation = "landscape"
        worksheet.page_setup.fitToWidth = 1
        worksheet.page_setup.fitToHeight = 0
        for row in worksheet.iter_rows(min_row=2, min_col=3, max_col=3):
            row[0].alignment = Alignment(wrap_text=True, vertical="top")
            worksheet.row_dimensions[row[0].row].height = max(
                30.0, 15.0 * ((len(str(row[0].value)) + 89) // 90)
            )
    return output_path


def write_per_cr_stats_workbook(stats_frame, output_path):
    """Write the long-form CR/satellite statistics with one sheet per satellite."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    assert not stats_frame.empty, "Cannot write an empty per-CR statistics workbook"
    assert {"cr", "sat"}.issubset(stats_frame.columns)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for sat_name in stats_frame["sat"].drop_duplicates():
            sheet_name = str(sat_name).replace("/", "-")[:31]
            frame = stats_frame.loc[stats_frame["sat"] == sat_name]
            frame.to_excel(writer, sheet_name=sheet_name, index=False)
            worksheet = writer.book[sheet_name]
            worksheet.freeze_panes = "A2"
            worksheet.auto_filter.ref = worksheet.dimensions
            for cell in worksheet[1]:
                cell.font = Font(bold=True)
            for column, name in enumerate(frame.columns, start=1):
                values = frame[name].astype(str)
                content_width = int(values.str.len().max()) if not values.empty else 0
                worksheet.column_dimensions[get_column_letter(column)].width = min(
                    38, max(12, len(str(name)) + 2, content_width + 2)
                )
    return output_path


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
