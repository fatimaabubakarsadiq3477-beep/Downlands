from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import json

import numpy as np
try:
    import pandas as pd
except ModuleNotFoundError:  # pragma: no cover
    pd = None
import streamlit as st

from activation_core import (
    solve_activation,
    to_jsonable,
)
from activation_params import get_default_params, get_feedstock_names, apply_feedstock
from activation_plots import (
    build_field_figures,
    build_gas_composition_plot,
    save_figures_png,
    plot_sweep_line,
    plot_feedstock_bar,
    plot_optimization_scores,
    save_all_figures,
)
from activation_report import (
    export_tables_csv,
    export_summary_csv,
    generate_txt_report,
    generate_docx_report,
    save_results_json,
    generate_dissertation_report,
    generate_figure_caption,
    generate_result_interpretation,
    get_dissertation_graph_templates,
)
from activation_validation import (
    validate_params as validate_physical_params,
    validate_results as validate_physical_results,
    classify_regime,
    compute_engineering_indices,
    interpret_regime,
)
from activation_parametric import (
    run_temperature_sweep,
    run_steam_flow_sweep,
    run_particle_size_sweep,
    run_density_sweep,
    compare_feedstocks as run_compare_feedstocks,
    optimize_activation,
    DEFAULT_CONSTRAINTS,
)

st.set_page_config(page_title="Паровая активация угля", layout="wide")
st.title("Моделирование процесса паровой активации угля")

if "params" not in st.session_state:
    st.session_state.params = get_default_params()
if "results" not in st.session_state:
    st.session_state.results = None
if "parametric_rows" not in st.session_state:
    st.session_state.parametric_rows = []
if "temp_sweep_df" not in st.session_state:
    st.session_state.temp_sweep_df = None
if "flow_sweep_df" not in st.session_state:
    st.session_state.flow_sweep_df = None
if "particle_sweep_df" not in st.session_state:
    st.session_state.particle_sweep_df = None
if "density_sweep_df" not in st.session_state:
    st.session_state.density_sweep_df = None
if "feedstock_compare_df" not in st.session_state:
    st.session_state.feedstock_compare_df = None
if "optimization_df" not in st.session_state:
    st.session_state.optimization_df = None
if "optimization_constraints" not in st.session_state:
    st.session_state.optimization_constraints = None

params = st.session_state.params

# Sidebar scenario i/o
st.sidebar.header("Сценарий")
if st.sidebar.button("Сбросить к дефолту"):
    st.session_state.params = get_default_params()
    st.rerun()

uploaded = st.sidebar.file_uploader("Загрузить JSON сценарий", type=["json"])
if uploaded is not None:
    loaded = json.loads(uploaded.read().decode("utf-8"))
    st.session_state.params = loaded
    st.success("Сценарий загружен")
    st.rerun()

scenario_json = json.dumps(params, ensure_ascii=False, indent=2)
st.sidebar.download_button(
    "Сохранить сценарий JSON",
    data=scenario_json.encode("utf-8"),
    file_name="activation_scenario.json",
    mime="application/json",
)


tabs = st.tabs([
    "Сырьё и предшествующие стадии",
    "Режим активации",
    "Интегральные результаты",
    "Поля реактора",
    "Параметрический анализ",
    "Экспорт и отчёт",
])

with tabs[0]:
    st.subheader("Сырьё и предшествующие стадии")
    fs = st.selectbox("Выбор сырья", get_feedstock_names(), index=get_feedstock_names().index(params["feedstock"]))
    if fs != params["feedstock"]:
        params = apply_feedstock(params, fs)

    c1, c2, c3 = st.columns(3)
    pre = params["pre_stages"]
    pre["T_pyrolysis_C"] = c1.number_input("Температура пиролиза, °C", value=float(pre["T_pyrolysis_C"]))
    pre["T_char_in_C"] = c2.number_input("Температура карбонизата на входе, °C", value=float(pre["T_char_in_C"]))
    pre["char_yield_pct"] = c3.number_input("Выход карбонизата, %", value=float(pre["char_yield_pct"]))

    c1, c2, c3 = st.columns(3)
    pre["rho_char_0"] = c1.number_input("Начальная плотность карбонизата, кг/м³", value=float(pre["rho_char_0"]))
    pre["porosity_0"] = c2.number_input("Начальная пористость", value=float(pre["porosity_0"]))
    pre["S_BET_0_m2g"] = c3.number_input("S_BET после пиролиза, м²/г", value=float(pre["S_BET_0_m2g"]))

    c1, c2, c3 = st.columns(3)
    pre["rho_pellet_kgm3"] = c1.number_input("Плотность гранулы после прессования, кг/м³", value=float(pre["rho_pellet_kgm3"]))
    pre["particle_size_mm"] = c2.number_input("Размер частицы/гранулы, мм", value=float(pre["particle_size_mm"]))
    pre["pellet_length_mm"] = c3.number_input("Длина гранулы, мм", value=float(pre["pellet_length_mm"]))
    pre["binder_pct"] = st.number_input("Доля связующего, %", value=float(pre["binder_pct"]))

with tabs[1]:
    st.subheader("Режим активации")
    op = params["operating"]
    r = params["reactor"]
    m = params["mesh"]

    c1, c2, c3 = st.columns(3)
    op["T_steam_in_C"] = c1.number_input("Температура пара на входе, °C", value=float(op["T_steam_in_C"]))
    op["T_wall_C"] = c2.number_input("Температура реактора/стенки, °C", value=float(op["T_wall_C"]))
    op["T_coal_in_C"] = c3.number_input("Температура угля на входе, °C", value=float(op["T_coal_in_C"]))

    c1, c2, c3 = st.columns(3)
    op["G_steam_in"] = c1.number_input("Расход пара, м/с", value=float(op["G_steam_in"]), format="%.4f")
    op["steam_to_coal_ratio"] = c2.number_input("Соотношение пар/уголь", value=float(op["steam_to_coal_ratio"]))
    r["H"] = c3.number_input("Высота зоны активации, м", value=float(r["H"]))

    c1, c2, c3 = st.columns(3)
    r["L"] = c1.number_input("Ширина слоя, м", value=float(r["L"]))
    r["W_s"] = c2.number_input("Скорость опускания слоя, м/с", value=float(r["W_s"]), format="%.6f")
    op["P_reactor"] = c3.number_input("Давление реактора, Па", value=float(op["P_reactor"]))

    tau_min = r["H"] / max(r["W_s"], 1e-9) / 60.0
    st.caption(f"Оценка времени пребывания: {tau_min:.2f} мин")

    c1, c2 = st.columns(2)
    m["N_h"] = int(c1.number_input("Сетка Nh", min_value=12, value=int(m["N_h"])))
    m["N_l"] = int(c2.number_input("Сетка Nl", min_value=8, value=int(m["N_l"])))

    if st.button("Рассчитать", type="primary"):
        params["reactor"]["d_p_0"] = pre["particle_size_mm"] / 1000.0
        params["reactor"]["rho_bulk_0"] = pre["rho_pellet_kgm3"]
        st.session_state.params = params
        with st.spinner("Выполняется расчет..."):
            st.session_state.results = solve_activation(params)
        st.success("Расчет завершен")

with tabs[2]:
    st.subheader("Интегральные результаты")
    res = st.session_state.results
    if res is None:
        st.info("Сначала выполните расчет на вкладке 'Режим активации'.")
    else:
        p = res["passport"]
        m = res["mass_balance"]
        e = res["energy_balance"]
        eta_mean = float(np.mean(res["fields"]["eta_2d"]))
        param_warnings = validate_physical_params(params)
        result_warnings = validate_physical_results(res, params)
        status = classify_regime(res, params)
        eng_idx = compute_engineering_indices(res, params)
        interpretation = interpret_regime(status, res)

        cols = st.columns(4)
        cols[0].metric("Средняя степень обгара X, %", f"{p['Burnoff_Mean_Pct']:.2f}")
        cols[1].metric("Выход активированного угля, %", f"{p['Yield_Pct']:.2f}")
        cols[2].metric("S_BET, м²/г", f"{p['BET_Surface_m2g']:.1f}")
        cols[3].metric("Макс. диаметр пор, нм", f"{p['Max_Pore_Diameter_nm']:.2f}")

        cols = st.columns(4)
        cols[0].metric("Средний фактор η", f"{eta_mean:.3f}")
        cols[1].metric("Газовая невязка, %", f"{m['gas_error_pct']:.2f}")
        cols[2].metric("Тепловой сток на эндотермику, кВт", f"{abs(e['Q_rxn_W'])/1000:.2f}")
        cols[3].metric("Статус режима", status)

        if status == "Рациональный режим":
            st.success(f"Статус режима: {status}")
        elif status in ("Слабая активация", "Требуется проверка параметров"):
            st.warning(f"Статус режима: {status}")
        else:
            st.error(f"Статус режима: {status}")

        st.markdown("**Предупреждения по входным параметрам**")
        if param_warnings:
            for w in param_warnings:
                st.warning(w)
        else:
            st.info("Предупреждений по входным параметрам нет.")

        st.markdown("**Предупреждения по результатам**")
        if result_warnings:
            for w in result_warnings:
                st.warning(w)
        else:
            st.info("Предупреждений по результатам нет.")

        st.markdown("**Инженерные индексы**")
        st.json({
            "I_act": eng_idx.get("I_act"),
            "K_ret": eng_idx.get("K_ret"),
            "K_quality": eng_idx.get("K_quality"),
            "strength_factor": eng_idx.get("strength_factor"),
            "K_T": eng_idx.get("K_T"),
        })
        st.markdown("**Краткая интерпретация режима**")
        st.info(interpretation)

with tabs[3]:
    st.subheader("Поля реактора")
    res = st.session_state.results
    if res is None:
        st.info("Нет данных для отображения.")
    else:
        required_fields = {
            "Ts": "Ts_2d",
            "Tg": "Tg_2d",
            "X": "X_2d",
            "S_BET": "S_BET_2d",
            "P_H2O": "P_h2o",
            "eta": "eta_2d",
            "r_act": "r_act_2d",
            "rho_bed": "rho_bed_2d",
            "D_eff": "D_eff_2d",
        }
        figs = build_field_figures(res)
        for plot_key, field_key in required_fields.items():
            if field_key in res["fields"] and plot_key in figs:
                st.pyplot(figs[plot_key], clear_figure=False)
            else:
                st.info(f"{plot_key}: поле отсутствует в results")

        if all(k in res["fields"] for k in ["y_H2O_2d", "y_CO_2d", "y_H2_2d", "y_CO2_2d"]):
            st.pyplot(build_gas_composition_plot(res), clear_figure=False)
        else:
            st.info("Состав газа: поле отсутствует в results")

with tabs[4]:
    st.subheader("Параметрический анализ")
    res = st.session_state.results
    if res is None:
        st.info("Сначала выполните базовый расчет.")
    elif pd is None:
        st.error("pandas не установлен: параметрический анализ недоступен. Установите зависимости из requirements.txt.")
    else:
        # 1) Серия по температуре
        st.markdown("### Серия по температуре пара")
        c1, c2, c3 = st.columns(3)
        tmin = c1.number_input("Tmin, °C", value=760.0, key="tmin")
        tmax = c2.number_input("Tmax, °C", value=940.0, key="tmax")
        tpoints = int(c3.number_input("Точек", min_value=2, max_value=25, value=6, key="tpoints"))
        if st.button("Рассчитать серию по температуре"):
            with st.spinner("Серия по температуре..."):
                pb = st.progress(5)
                vals = np.linspace(tmin, tmax, tpoints).tolist()
                st.session_state.temp_sweep_df = run_temperature_sweep(params, vals)
                pb.progress(100)
            st.success("Серия по температуре завершена")
        if st.session_state.temp_sweep_df is not None:
            df = st.session_state.temp_sweep_df
            st.dataframe(df)
            st.pyplot(plot_sweep_line(df, "T_steam", "X_mean", "X_mean от T_steam", "T_steam, °C", "X_mean, %"))
            st.pyplot(plot_sweep_line(df, "T_steam", "Y_AU", "Y_AU от T_steam", "T_steam, °C", "Y_AU, %"))
            st.pyplot(plot_sweep_line(df, "T_steam", "S_BET", "S_BET от T_steam", "T_steam, °C", "S_BET, м²/г"))
            st.pyplot(plot_sweep_line(df, "T_steam", "I_act", "I_act от T_steam", "T_steam, °C", "I_act"))

        # 2) Серия по расходу пара
        st.markdown("### Серия по расходу пара")
        c1, c2, c3 = st.columns(3)
        fmin = c1.number_input("flow_min, м/с", value=0.12, key="fmin")
        fmax = c2.number_input("flow_max, м/с", value=0.40, key="fmax")
        fpoints = int(c3.number_input("Точек ", min_value=2, max_value=25, value=6, key="fpoints"))
        if st.button("Рассчитать серию по расходу"):
            with st.spinner("Серия по расходу..."):
                pb = st.progress(5)
                vals = np.linspace(fmin, fmax, fpoints).tolist()
                st.session_state.flow_sweep_df = run_steam_flow_sweep(params, vals)
                pb.progress(100)
            st.success("Серия по расходу завершена")
        if st.session_state.flow_sweep_df is not None:
            df = st.session_state.flow_sweep_df
            st.dataframe(df)
            st.pyplot(plot_sweep_line(df, "steam_flow", "X_mean", "X_mean от steam_flow", "steam_flow, м/с", "X_mean, %"))
            st.pyplot(plot_sweep_line(df, "steam_flow", "Y_AU", "Y_AU от steam_flow", "steam_flow, м/с", "Y_AU, %"))
            st.pyplot(plot_sweep_line(df, "steam_flow", "S_BET", "S_BET от steam_flow", "steam_flow, м/с", "S_BET, м²/г"))
            st.pyplot(plot_sweep_line(df, "steam_flow", "gas_residual", "gas_residual от steam_flow", "steam_flow, м/с", "gas_residual, %"))

        # 3) Серия по размеру частиц
        st.markdown("### Серия по размеру частиц")
        c1, c2, c3 = st.columns(3)
        dmin = c1.number_input("d_min, мм", value=1.0, key="dmin")
        dmax = c2.number_input("d_max, мм", value=8.0, key="dmax")
        dpoints = int(c3.number_input("Точек  ", min_value=2, max_value=25, value=6, key="dpoints"))
        if st.button("Рассчитать серию по размеру"):
            with st.spinner("Серия по размеру частиц..."):
                pb = st.progress(5)
                vals = np.linspace(dmin, dmax, dpoints).tolist()
                st.session_state.particle_sweep_df = run_particle_size_sweep(params, vals)
                pb.progress(100)
            st.success("Серия по размеру завершена")
        if st.session_state.particle_sweep_df is not None:
            df = st.session_state.particle_sweep_df
            st.dataframe(df)
            st.pyplot(plot_sweep_line(df, "d_particle", "eta_mean", "eta_mean от d_particle", "d_particle, мм", "eta_mean"))
            st.pyplot(plot_sweep_line(df, "d_particle", "X_mean", "X_mean от d_particle", "d_particle, мм", "X_mean, %"))
            st.pyplot(plot_sweep_line(df, "d_particle", "S_BET", "S_BET от d_particle", "d_particle, мм", "S_BET, м²/г"))

        # 4) Серия по плотности гранул
        st.markdown("### Серия по плотности гранул")
        c1, c2, c3 = st.columns(3)
        rmin = c1.number_input("rho_min, кг/м³", value=600.0, key="rmin")
        rmax = c2.number_input("rho_max, кг/м³", value=950.0, key="rmax")
        rpoints = int(c3.number_input("Точек   ", min_value=2, max_value=25, value=6, key="rpoints"))
        if st.button("Рассчитать серию по плотности"):
            with st.spinner("Серия по плотности..."):
                pb = st.progress(5)
                vals = np.linspace(rmin, rmax, rpoints).tolist()
                st.session_state.density_sweep_df = run_density_sweep(params, vals)
                pb.progress(100)
            st.success("Серия по плотности завершена")
        if st.session_state.density_sweep_df is not None:
            df = st.session_state.density_sweep_df
            st.dataframe(df)
            st.pyplot(plot_sweep_line(df, "rho_granule", "X_mean", "X_mean от rho_granule", "rho_granule, кг/м³", "X_mean, %"))
            st.pyplot(plot_sweep_line(df, "rho_granule", "S_BET", "S_BET от rho_granule", "rho_granule, кг/м³", "S_BET, м²/г"))
            st.pyplot(plot_sweep_line(df, "rho_granule", "strength_factor", "strength_factor от rho_granule", "rho_granule, кг/м³", "strength_factor"))
            st.pyplot(plot_sweep_line(df, "rho_granule", "K_quality", "K_quality от rho_granule", "rho_granule, кг/м³", "K_quality"))

        # 5) Сравнение 7 видов сырья
        st.markdown("### Сравнить 7 видов сырья")
        if st.button("Сравнить все виды сырья"):
            with st.spinner("Сравнение сырья..."):
                pb = st.progress(5)
                st.session_state.feedstock_compare_df = run_compare_feedstocks(params, get_feedstock_names())
                pb.progress(100)
            st.success("Сравнение сырья завершено")
        if st.session_state.feedstock_compare_df is not None:
            df = st.session_state.feedstock_compare_df
            st.dataframe(df)
            st.pyplot(plot_feedstock_bar(df, "X_mean", "X_mean по видам сырья", "X_mean, %"))
            st.pyplot(plot_feedstock_bar(df, "Y_AU", "Y_AU по видам сырья", "Y_AU, %"))
            st.pyplot(plot_feedstock_bar(df, "S_BET", "S_BET по видам сырья", "S_BET, м²/г"))
            st.pyplot(plot_feedstock_bar(df, "I_act", "I_act по видам сырья", "I_act"))
            st.pyplot(plot_feedstock_bar(df, "K_ret", "K_ret по видам сырья", "K_ret"))

        # 6) Оптимизация
        st.markdown("### Оптимизация режима")
        c = st.columns(6)
        sb_min = c[0].number_input("S_BET_min", value=float(DEFAULT_CONSTRAINTS["S_BET_min"]))
        y_min = c[1].number_input("Y_AU_min", value=float(DEFAULT_CONSTRAINTS["Y_AU_min"]))
        x_min = c[2].number_input("X_min", value=float(DEFAULT_CONSTRAINTS["X_min"]))
        x_max = c[3].number_input("X_max", value=float(DEFAULT_CONSTRAINTS["X_max"]))
        gas_max = c[4].number_input("gas_residual_max", value=float(DEFAULT_CONSTRAINTS["gas_residual_max"]))
        ts_max = c[5].number_input("T_steam_max", value=float(DEFAULT_CONSTRAINTS["T_steam_max"]))

        c = st.columns(3)
        optimize_t = c[0].checkbox("оптимизировать температуру пара", value=True)
        optimize_g = c[1].checkbox("оптимизировать расход пара", value=True)
        optimize_rho = c[2].checkbox("оптимизировать плотность гранулы", value=False)
        optimize_d = st.checkbox("оптимизировать размер частицы", value=False)

        c = st.columns(5)
        tmin_opt = c[0].number_input("T_steam_min", value=780.0)
        tmax_opt = c[1].number_input("T_steam_max(search)", value=980.0)
        tpts_opt = int(c[2].number_input("T_points", min_value=1, max_value=12, value=4))
        fpts_opt = int(c[3].number_input("flow_points", min_value=1, max_value=12, value=4))
        max_cases = int(c[4].number_input("max_cases", min_value=5, max_value=200, value=50))

        c = st.columns(4)
        flow_min_opt = c[0].number_input("steam_flow_min", value=0.12)
        flow_max_opt = c[1].number_input("steam_flow_max", value=0.40)
        rho_min_opt = c[2].number_input("rho_granule_min", value=600.0)
        rho_max_opt = c[3].number_input("rho_granule_max", value=950.0)
        c = st.columns(4)
        rho_pts_opt = int(c[0].number_input("rho_points", min_value=1, max_value=12, value=2))
        dmin_opt = c[1].number_input("d_particle_min", value=1.0)
        dmax_opt = c[2].number_input("d_particle_max", value=8.0)
        dpts_opt = int(c[3].number_input("d_points", min_value=1, max_value=12, value=2))

        if st.button("Найти рациональный режим"):
            constraints = {
                "S_BET_min": sb_min,
                "Y_AU_min": y_min,
                "X_min": x_min,
                "X_max": x_max,
                "gas_residual_max": gas_max,
                "T_steam_max": ts_max,
                "prefer_strength": True,
            }
            st.session_state.optimization_constraints = constraints
            search_space = {
                "T_steam": np.linspace(tmin_opt, tmax_opt, tpts_opt).tolist() if optimize_t else [params["operating"]["T_steam_in_C"]],
                "steam_flow": np.linspace(flow_min_opt, flow_max_opt, fpts_opt).tolist() if optimize_g else [params["operating"]["G_steam_in"]],
                "bed_velocity": [params["reactor"]["W_s"]],
                "rho_granule": np.linspace(rho_min_opt, rho_max_opt, rho_pts_opt).tolist() if optimize_rho else [params["pre_stages"]["rho_pellet_kgm3"]],
                "d_particle": np.linspace(dmin_opt, dmax_opt, dpts_opt).tolist() if optimize_d else [params["pre_stages"]["particle_size_mm"]],
            }
            total_cases = len(search_space["T_steam"]) * len(search_space["steam_flow"]) * len(search_space["bed_velocity"]) * len(search_space["rho_granule"]) * len(search_space["d_particle"])
            if total_cases > max_cases:
                st.warning(f"Комбинаций {total_cases} > max_cases={max_cases}. Будут обработаны только первые {max_cases}.")
            with st.spinner("Оптимизация режима..."):
                pb = st.progress(5)
                st.session_state.optimization_df = optimize_activation(params, search_space, constraints, max_cases=max_cases)
                pb.progress(100)
            st.success("Оптимизация завершена")

        if st.session_state.optimization_df is not None:
            df = st.session_state.optimization_df
            st.dataframe(df.head(10))
            if not df.empty:
                best = df.iloc[0]
                st.markdown("#### Лучший режим")
                st.json(best.to_dict())
                reasons = []
                if best.get("S_BET", 0) >= sb_min:
                    reasons.append("достигнута S_BET")
                if x_min <= best.get("X_mean", -1) <= x_max:
                    reasons.append("X в рациональном диапазоне")
                if best.get("Y_AU", 0) >= y_min:
                    reasons.append("выход выше минимума")
                if best.get("gas_residual", 999) <= gas_max:
                    reasons.append("газовая невязка допустима")
                if best.get("warnings_count", 99) <= 3:
                    reasons.append("предупреждений мало")
                st.info("Причины выбора: " + (", ".join(reasons) if reasons else "нужно дополнительное обоснование"))
            st.pyplot(plot_optimization_scores(df.head(30)))

with tabs[5]:
    st.subheader("Экспорт и отчёт")
    st.markdown("#### Сценарий расчёта (JSON)")
    scenario_json_tab = json.dumps(params, ensure_ascii=False, indent=2)
    st.download_button(
        "Сохранить сценарий расчёта (JSON)",
        data=scenario_json_tab.encode("utf-8"),
        file_name="activation_scenario.json",
        mime="application/json",
        key="download_scenario_tab6",
    )
    scenario_upload_tab = st.file_uploader(
        "Загрузить сценарий из JSON",
        type=["json"],
        key="upload_scenario_tab6",
    )
    if scenario_upload_tab is not None:
        loaded_params = json.loads(scenario_upload_tab.read().decode("utf-8"))
        st.session_state.params = loaded_params
        st.success("Сценарий загружен. Нажмите 'Рассчитать' на вкладке режима.")

    res = st.session_state.results
    if res is None:
        st.info("Сначала выполните расчет.")
    else:
        out_dir = Path(params["meta"].get("output_dir", "section_2_5_output"))
        out_dir.mkdir(parents=True, exist_ok=True)

        if st.button("Сохранить графики в PNG"):
            figs = build_field_figures(res)
            figs["gas_composition"] = build_gas_composition_plot(res)
            paths = save_figures_png(figs, str(out_dir))
            st.success(f"Сохранено PNG: {len(paths)} файлов")

        if st.button("Сохранить таблицы в CSV"):
            csvs = export_tables_csv(res, str(out_dir))
            st.success(f"Сохранено CSV: {len(csvs)} файлов")

        if st.button("Сформировать TXT-отчёт"):
            report_payload = deepcopy(res)
            report_payload["parametric"] = {
                "temperature_sweep": [] if st.session_state.temp_sweep_df is None else st.session_state.temp_sweep_df.to_dict(orient="records"),
                "steam_flow_sweep": [] if st.session_state.flow_sweep_df is None else st.session_state.flow_sweep_df.to_dict(orient="records"),
                "particle_size_sweep": [] if st.session_state.particle_sweep_df is None else st.session_state.particle_sweep_df.to_dict(orient="records"),
                "density_sweep": [] if st.session_state.density_sweep_df is None else st.session_state.density_sweep_df.to_dict(orient="records"),
                "feedstock_compare": [] if st.session_state.feedstock_compare_df is None else st.session_state.feedstock_compare_df.to_dict(orient="records"),
            }
            report_payload["optimization"] = [] if st.session_state.optimization_df is None else st.session_state.optimization_df.to_dict(orient="records")
            report_payload["optimization_constraints"] = st.session_state.optimization_constraints or {}
            p = generate_txt_report(report_payload, str(out_dir / "activation_report.txt"))
            st.success(f"TXT отчёт: {p}")

        if st.button("Сформировать DOCX-отчёт"):
            report_payload = deepcopy(res)
            report_payload["parametric"] = {
                "temperature_sweep": [] if st.session_state.temp_sweep_df is None else st.session_state.temp_sweep_df.to_dict(orient="records"),
                "steam_flow_sweep": [] if st.session_state.flow_sweep_df is None else st.session_state.flow_sweep_df.to_dict(orient="records"),
                "particle_size_sweep": [] if st.session_state.particle_sweep_df is None else st.session_state.particle_sweep_df.to_dict(orient="records"),
                "density_sweep": [] if st.session_state.density_sweep_df is None else st.session_state.density_sweep_df.to_dict(orient="records"),
                "feedstock_compare": [] if st.session_state.feedstock_compare_df is None else st.session_state.feedstock_compare_df.to_dict(orient="records"),
            }
            report_payload["optimization"] = [] if st.session_state.optimization_df is None else st.session_state.optimization_df.to_dict(orient="records")
            report_payload["optimization_constraints"] = st.session_state.optimization_constraints or {}
            p = generate_docx_report(report_payload, str(out_dir / "activation_report.docx"))
            if p:
                st.success(f"DOCX отчёт: {p}")
            else:
                st.warning("python-docx не установлен; DOCX не сформирован.")

        if st.button("Сохранить результаты JSON"):
            p = save_results_json(to_jsonable(res), str(out_dir / "activation_results.json"))
            st.success(f"JSON результатов: {p}")

        if st.button("Сохранить summary CSV"):
            if st.session_state.parametric_rows:
                p = export_summary_csv(st.session_state.parametric_rows, str(out_dir / "parametric_summary.csv"))
                st.success(f"Сводная параметрика: {p}")
            else:
                summary = [{
                    "X_pct": res["passport"]["Burnoff_Mean_Pct"],
                    "Y_AU_pct": res["passport"]["Yield_Pct"],
                    "S_BET_m2g": res["passport"]["BET_Surface_m2g"],
                    "gas_residual_pct": res["mass_balance"]["gas_error_pct"],
                    "status": res["status"],
                }]
                p = export_summary_csv(summary, str(out_dir / "summary_single_run.csv"))
                st.success(f"Summary CSV: {p}")

        st.markdown("#### Экспорт серийных расчётов (CSV)")
        series_exports = [
            ("Температурная серия", st.session_state.temp_sweep_df, "temperature_sweep.csv"),
            ("Серия по расходу пара", st.session_state.flow_sweep_df, "steam_flow_sweep.csv"),
            ("Серия по размеру частиц", st.session_state.particle_sweep_df, "particle_size_sweep.csv"),
            ("Серия по плотности", st.session_state.density_sweep_df, "density_sweep.csv"),
            ("Сравнение сырья", st.session_state.feedstock_compare_df, "feedstock_compare.csv"),
            ("Оптимизация", st.session_state.optimization_df, "optimization_results.csv"),
        ]
        for label, df, fname in series_exports:
            if st.button(f"Сохранить CSV: {label}", key=f"export_{fname}"):
                if df is None or df.empty:
                    st.warning(f"Нет данных для экспорта: {label}")
                else:
                    path = out_dir / fname
                    df.to_csv(path, index=False, encoding="utf-8-sig")
                    st.success(f"Сохранено: {path}")

        st.markdown("### Диссертационный экспорт")
        parametric_pack = {
            "temperature_sweep": st.session_state.temp_sweep_df,
            "steam_flow_sweep": st.session_state.flow_sweep_df,
            "particle_size_sweep": st.session_state.particle_sweep_df,
            "density_sweep": st.session_state.density_sweep_df,
            "feedstock_compare": st.session_state.feedstock_compare_df,
            "optimization": st.session_state.optimization_df,
        }

        if st.button("Сохранить все графики 300 dpi"):
            created = save_all_figures(res, parametric_pack, str(out_dir), dpi=300)
            if created:
                st.success("Созданы файлы:")
                for p in created:
                    st.write(p)
            else:
                st.warning("Графики не созданы: недостаточно данных.")

        if st.button("Сформировать DOCX для раздела 4.5"):
            rep = generate_dissertation_report(
                res,
                params,
                parametric_results=parametric_pack,
                optimization_results=st.session_state.optimization_df,
                output_path=str(out_dir / "dissertation_4_5_report.docx"),
            )
            st.success("Диссертационный отчёт сформирован")
            st.json(rep)

        if st.button("Сформировать TXT для раздела 4.5"):
            rep = generate_dissertation_report(
                res,
                params,
                parametric_results=parametric_pack,
                optimization_results=st.session_state.optimization_df,
                output_path=str(out_dir / "dissertation_4_5_report.txt"),
            )
            st.success("Диссертационный TXT сформирован")
            st.json(rep)

        if st.button("Сохранить captions и interpretation в TXT/MD"):
            templates = get_dissertation_graph_templates()
            lines = ["# Captions и интерпретации (раздел 4.5)", ""]
            for t in templates:
                caption = generate_figure_caption("feedstock_comparison" if t["id"] == 1 else "optimization_result", {})
                lines.append(f"## {t['id']}. {t['title']}")
                lines.append(f"- Caption: {caption}")
                lines.append(f"- Before: {t['before_text']}")
                lines.append(f"- After: {t['after_text']}")
                lines.append("")
            lines.append("## Общая интерпретация")
            opt_df_interp = st.session_state.optimization_df if st.session_state.optimization_df is not None else pd.DataFrame()
            lines.append(generate_result_interpretation("optimization", opt_df_interp, params))
            md_path = out_dir / "4_5_captions_interpretation.md"
            txt_path = out_dir / "4_5_captions_interpretation.txt"
            md_path.write_text("\n".join(lines), encoding="utf-8")
            txt_path.write_text("\n".join(lines), encoding="utf-8")
            st.success(f"Сохранено: {md_path} и {txt_path}")
