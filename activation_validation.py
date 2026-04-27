"""Слой физической валидации и инженерной интерпретации результатов."""

from __future__ import annotations

from typing import Any

import numpy as np


def _safe_get(d: dict, *path, default=None):
    cur: Any = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


def validate_params(params: dict) -> list[str]:
    w: list[str] = []
    op = params.get("operating", {})
    pre = params.get("pre_stages", {})

    t_steam = float(op.get("T_steam_in_C", np.nan))
    t_reactor = float(op.get("T_wall_C", np.nan))
    t_char = float(pre.get("T_char_in_C", np.nan))
    g_steam = float(op.get("G_steam_in", np.nan))
    d_mm = float(pre.get("particle_size_mm", np.nan))
    rho_gr = float(pre.get("rho_pellet_kgm3", np.nan))

    # Температура пара
    if t_steam < 700:
        w.append("Температура пара ниже области эффективной паровой активации; вероятна слабая газификация углерода.")
    elif t_steam < 800:
        w.append("Мягкий режим активации; возможно недостаточное развитие микропор.")
    elif t_steam <= 900:
        w.append("Рабочая область паровой активации для большинства растительных карбонизатов.")
    elif t_steam <= 1000:
        w.append("Интенсивная активация; требуется контроль обгара и прочности.")
    else:
        w.append("Риск переактивации, укрупнения пор и потери прочности.")

    # Температура реактора
    if np.isfinite(t_reactor) and np.isfinite(t_steam):
        if t_steam - t_reactor > 150:
            w.append("Температура реактора значительно ниже температуры пара (>150 °C): возможна тепловая несогласованность.")
        if t_reactor > 1050:
            w.append("Температура реактора выше 1050 °C: чрезмерный тепловой режим.")

    # Температура карбонизата
    if t_char < 400:
        w.append("Карбонизат слишком холодный; значительная часть зоны активации будет расходоваться на догрев.")
    elif 500 <= t_char <= 750:
        w.append("Допустимый входной тепловой режим.")
    elif t_char > 850:
        w.append("Высокая температура карбонизата; возможен быстрый выход в режим интенсивной газификации.")

    # Расход пара (preliminary thresholds)
    if g_steam < 0.12:
        w.append("Вероятно исчерпание H2O в начальной зоне слоя (preliminary threshold).")
    elif g_steam > 0.40:
        w.append("Риск избыточного теплового и газодинамического воздействия; возможна переактивация (preliminary threshold).")

    # Размер частиц
    if d_mm < 0.5:
        w.append("Высокая доступность пара, но возможен унос мелочи.")
    elif 1 <= d_mm <= 5:
        w.append("Рациональная область для активации частиц.")
    elif d_mm > 10:
        w.append("Вероятны внутридиффузионные ограничения.")

    # Плотность гранул
    if rho_gr < 600:
        w.append("Низкая плотность; риск разрушения гранулы.")
    elif 750 <= rho_gr <= 950:
        w.append("Рациональная область для гранулированных растительных карбонизатов.")
    elif rho_gr > 1100:
        w.append("Высокая плотность; вероятны внутридиффузионные ограничения.")

    return w


def compute_engineering_indices(results: dict, params: dict) -> dict:
    warnings = []
    p = results.get("passport", {})
    x = p.get("Burnoff_Mean_Pct")
    y = p.get("Yield_Pct")
    sbet = p.get("BET_Surface_m2g")

    if x is None or y is None or sbet is None:
        warnings.append("Недостаточно данных для расчёта инженерных индексов.")
        return {
            "I_act": None,
            "K_ret": None,
            "K_quality": None,
            "strength_factor": None,
            "K_T": None,
            "_warnings": warnings,
        }

    x = float(max(x, 1e-9))
    y = float(y)
    sbet = float(sbet)

    # Условная модель прочности
    if x <= 40:
        strength = 1.0
    elif x <= 45:
        strength = 0.92
    elif x <= 55:
        strength = 0.80
    else:
        strength = 0.65

    return {
        "I_act": sbet / x,
        "K_ret": y * sbet / 100.0,
        "K_quality": sbet * strength / x,
        "strength_factor": strength,
        "K_T": None,
        "_warnings": warnings,
    }


def validate_results(results: dict, params: dict) -> list[str]:
    w: list[str] = []
    p = results.get("passport", {})
    f = results.get("fields", {})
    mb = results.get("mass_balance", {})

    x = p.get("Burnoff_Mean_Pct")
    y = p.get("Yield_Pct")
    sbet = p.get("BET_Surface_m2g")
    gas = mb.get("gas_error_pct")

    # X
    if x is not None:
        x = float(x)
        if x < 20:
            w.append("X < 20 %: слабая активация, вероятно недостаточное развитие пор.")
        elif x < 30:
            w.append("X = 20–30 %: умеренная активация.")
        elif x <= 40:
            w.append("X = 30–40 %: рациональная область для гранулированных растительных карбонизатов.")
        elif x <= 50:
            w.append("X = 40–50 %: интенсивная активация, нужен контроль прочности.")
        elif x <= 70:
            w.append("X > 50 %: риск потери прочности.")
        else:
            w.append("X > 70 %: аварийная переактивация / физически подозрительный режим.")

    # Y
    if y is not None and sbet is not None:
        y = float(y)
        sbet = float(sbet)
        if y > 80 and sbet > 800:
            w.append("Y > 80 % при высокой S_BET: проверить физическую согласованность, возможно обгар занижен.")
        if y < 40:
            w.append("Y < 40 %: сильная потеря массы, возможна переактивация.")
        if x is not None and abs((100.0 - float(x)) - y) > 12:
            w.append("Связь Y ≈ 100 - X заметно нарушена: проверьте корректность параметров и поправок.")

    # S_BET
    if sbet is not None:
        sbet = float(sbet)
        if sbet < 300:
            w.append("S_BET < 300 м²/г: слабое развитие пористой структуры.")
        elif sbet < 600:
            w.append("S_BET = 300–600 м²/г: умеренная активация.")
        elif sbet <= 1000:
            w.append("S_BET = 600–1000 м²/г: рабочая область для растительных активированных углей.")
        elif sbet > 1500:
            w.append("S_BET > 1500 м²/г: проверить корректность модели и параметров.")
        if x is not None and x > 50 and sbet > 1000:
            w.append("Поверхность монотонно растёт при высоком обгаре; требуется проверить учёт разрушения микропористой структуры.")

    # Газовая невязка
    if gas is not None:
        gas = float(gas)
        if gas < 7:
            w.append("Газовая невязка < 7 %: хорошо.")
        elif gas <= 10:
            w.append("Газовая невязка 7–10 %: допустимо для предварительного инженерного расчёта.")
        elif gas <= 20:
            w.append("Газовая невязка > 10 %: требуется проверка баланса, расхода пара или сетки.")
        else:
            w.append("Газовая невязка > 20 %: физически подозрительный расчёт.")

    # Температурные поля
    Ts = f.get("Ts_2d")
    Tg = f.get("Tg_2d")
    if Ts is not None:
        ts_c = np.asarray(Ts) - 273.15
        if np.mean(ts_c < 650) > 0.5:
            w.append("Ts < 650 °C в большей части зоны: слабая активация.")
        if np.max(ts_c) > 1000:
            w.append("Ts > 1000 °C: риск переактивации.")
    if Tg is not None:
        tg_c = np.asarray(Tg) - 273.15
        if np.max(tg_c) > 1100:
            w.append("Tg > 1100 °C: проверить тепловой режим.")
    if Ts is not None and Tg is not None:
        delta = np.abs((np.asarray(Tg) - np.asarray(Ts)))
        if np.mean(delta > 300) > 0.2:
            w.append("abs(Tg - Ts) > 300 °C на значимой области: сильная межфазная неравновесность.")

    # P_H2O
    P_h2o = f.get("P_h2o")
    if P_h2o is not None and "l_grid" in f:
        arr = np.asarray(P_h2o)
        first = arr[:, 0]
        early_idx = max(1, int(arr.shape[1] * 0.08))
        early = arr[:, early_idx]
        last = arr[:, -1]
        if np.mean(early / np.maximum(first, 1e-9) < 0.15) > 0.3:
            w.append("H2O почти полностью исчезает в первых 5–10 % длины слоя: проверьте расход пара/скорость реакции.")
        if np.mean(last / np.maximum(first, 1e-9) > 0.90) > 0.6:
            w.append("H2O почти не расходуется по всему слою: реакция активации слабая.")

    # eta
    eta = f.get("eta_2d")
    if eta is not None:
        eta_a = np.asarray(eta)
        eta_mean = float(np.mean(eta_a))
        if eta_mean < 0.2:
            w.append("eta < 0.2: сильные внутридиффузионные ограничения.")
        elif eta_mean <= 0.7:
            w.append("eta 0.2–0.7: смешанный кинетико-диффузионный режим.")
        else:
            w.append("eta > 0.7: реакция близка к кинетическому режиму.")
        d_mm = _safe_get(params, "pre_stages", "particle_size_mm", default=None)
        if d_mm is not None and float(d_mm) > 5 and np.mean(np.isclose(eta_a, 1.0, atol=1e-6)) > 0.9:
            w.append("eta постоянно равен 1 при крупных частицах: возможна потеря диффузионной модели.")

    return w


def classify_regime(results: dict, params: dict) -> str:
    p = results.get("passport", {})
    x = p.get("Burnoff_Mean_Pct")
    gas = _safe_get(results, "mass_balance", "gas_error_pct", default=None)

    if x is None:
        return "Требуется проверка параметров"

    x = float(x)
    if gas is not None and float(gas) > 20:
        return "Физически подозрительный расчёт"
    if x > 70:
        return "Физически подозрительный расчёт"
    if x > 50:
        return "Риск потери прочности"
    if x > 40:
        return "Риск переактивации"
    if x < 20:
        return "Слабая активация"
    if 30 <= x <= 40:
        return "Рациональный режим"
    return "Требуется проверка параметров"


def interpret_regime(status: str, results: dict) -> str:
    p = results.get("passport", {})
    mb = results.get("mass_balance", {})
    x = p.get("Burnoff_Mean_Pct")
    s = p.get("BET_Surface_m2g")
    gas = mb.get("gas_error_pct")

    if status == "Рациональный режим":
        return (
            "Режим соответствует рациональной области паровой активации: "
            f"X={x:.1f}% находится около диапазона 30–40%, "
            f"S_BET={s:.0f} м²/г в рабочей области, "
            f"газовая невязка={gas:.2f}% в допустимых пределах."
        )
    if status in ("Риск переактивации", "Риск потери прочности", "Физически подозрительный расчёт"):
        return (
            "Режим характеризуется риском переактивации: повышенный обгар может снижать "
            "прочность гранул и приводить к укрупнению пор. Требуется пересмотр параметров."
        )
    if status == "Слабая активация":
        return "Режим слабой активации: вероятно недостаточное развитие пористой структуры."
    return "Требуется дополнительная проверка параметров и качества входных данных."

