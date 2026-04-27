"""Параметры сценариев паровой активации (v3.2 wrapper).

Все значения по видам сырья являются предварительными (preliminary / to be calibrated)
и предназначены для инженерных сценарных расчётов.
"""

from __future__ import annotations

from copy import deepcopy

# Предварительные параметры 7 видов сырья (требуют калибровки)
_FEEDSTOCKS = {
    "Скорлупа грецкого ореха": {
        "rho_bulk_0": 540.0,
        "eps0": 0.34,
        "S0": 2.1e5,
        "reactivity_factor": 1.00,
        "stability_factor": 0.82,
        "micro_meso_bias": "micro",
        "ash_factor": 1.00,
        "steam_temp_range_C": [780.0, 900.0],
        "burnoff_range_pct": [30.0, 42.0],
        "notes": "preliminary / to be calibrated",
    },
    "Скорлупа косточки абрикоса": {
        "rho_bulk_0": 560.0,
        "eps0": 0.32,
        "S0": 1.95e5,
        "reactivity_factor": 0.96,
        "stability_factor": 0.86,
        "micro_meso_bias": "micro",
        "ash_factor": 1.03,
        "steam_temp_range_C": [790.0, 910.0],
        "burnoff_range_pct": [30.0, 40.0],
        "notes": "preliminary / to be calibrated",
    },
    "Лузга подсолнечника": {
        "rho_bulk_0": 470.0,
        "eps0": 0.40,
        "S0": 1.75e5,
        "reactivity_factor": 1.08,
        "stability_factor": 0.70,
        "micro_meso_bias": "meso",
        "ash_factor": 1.12,
        "steam_temp_range_C": [760.0, 880.0],
        "burnoff_range_pct": [28.0, 38.0],
        "notes": "preliminary / to be calibrated",
    },
    "Костра льна": {
        "rho_bulk_0": 430.0,
        "eps0": 0.43,
        "S0": 1.68e5,
        "reactivity_factor": 1.12,
        "stability_factor": 0.66,
        "micro_meso_bias": "meso",
        "ash_factor": 1.15,
        "steam_temp_range_C": [750.0, 870.0],
        "burnoff_range_pct": [26.0, 36.0],
        "notes": "preliminary / to be calibrated",
    },
    "Лузга гречихи": {
        "rho_bulk_0": 500.0,
        "eps0": 0.38,
        "S0": 1.85e5,
        "reactivity_factor": 1.02,
        "stability_factor": 0.74,
        "micro_meso_bias": "micro-meso",
        "ash_factor": 1.09,
        "steam_temp_range_C": [770.0, 890.0],
        "burnoff_range_pct": [28.0, 39.0],
        "notes": "preliminary / to be calibrated",
    },
    "Жом свекольный": {
        "rho_bulk_0": 420.0,
        "eps0": 0.45,
        "S0": 1.60e5,
        "reactivity_factor": 1.15,
        "stability_factor": 0.62,
        "micro_meso_bias": "meso",
        "ash_factor": 1.18,
        "steam_temp_range_C": [740.0, 860.0],
        "burnoff_range_pct": [24.0, 34.0],
        "notes": "preliminary / to be calibrated",
    },
    "Ствол подсолнечника": {
        "rho_bulk_0": 450.0,
        "eps0": 0.41,
        "S0": 1.72e5,
        "reactivity_factor": 1.10,
        "stability_factor": 0.67,
        "micro_meso_bias": "meso",
        "ash_factor": 1.14,
        "steam_temp_range_C": [755.0, 875.0],
        "burnoff_range_pct": [26.0, 36.0],
        "notes": "preliminary / to be calibrated",
    },
}

RANGES = {
    "T_steam_in_C": (650.0, 1050.0),
    "T_wall_C": (650.0, 1100.0),
    "T_char_in_C": (450.0, 1000.0),
    "G_steam_in": (0.01, 1.20),
    "H": (0.2, 8.0),
    "L": (0.05, 2.5),
    "W_s": (1e-6, 3e-3),
    "P_reactor": (6e4, 3e5),
    "N_h": (12, 260),
    "N_l": (8, 160),
    "d_p_0": (0.0005, 0.03),
    "binder_pct": (0.0, 25.0),
    "burnoff_target_pct": (10.0, 75.0),
}


def get_feedstock_names() -> list[str]:
    return list(_FEEDSTOCKS.keys())


def get_feedstock_params(name: str) -> dict:
    if name not in _FEEDSTOCKS:
        raise KeyError(f"Неизвестный вид сырья: {name}")
    return deepcopy(_FEEDSTOCKS[name])


def get_default_params() -> dict:
    feedstock = "Скорлупа грецкого ореха"
    fs = get_feedstock_params(feedstock)

    return {
        "feedstock": feedstock,
        "feedstock_preliminary": True,
        "pre_stages": {
            "T_pyrolysis_C": 520.0,
            "T_char_in_C": 780.0,
            "char_yield_pct": 33.0,
            "rho_char_0": fs["rho_bulk_0"],
            "porosity_0": fs["eps0"],
            "S_BET_0_m2g": 180.0,
            "rho_pellet_kgm3": 690.0,
            "particle_size_mm": 4.0,
            "pellet_length_mm": 8.0,
            "binder_pct": 6.0,
        },
        "operating": {
            "T_steam_in_C": 880.0,
            "T_wall_C": 920.0,
            "T_coal_in_C": 800.0,
            "G_steam_in": 0.25,
            "steam_to_coal_ratio": 1.4,
            "P_reactor": 101325.0,
        },
        "reactor": {
            "H": 2.2,
            "L": 0.35,
            "W_s": 2.0e-4,
            "rho_bulk_0": fs["rho_bulk_0"],
            "d_p_0": 0.004,
        },
        "mesh": {
            "N_h": 80,
            "N_l": 32,
        },
        "rpm": {
            "S0": fs["S0"],
            "eps0": fs["eps0"],
            "psi": 6.2,
            "S_area_floor": 1e-9,
            "d_pore_min": 0.4e-9,
            "d_pore_max": 5.0e-6,
        },
        "kinetics": {
            "k1": 3.6e-8,
            "E1": 130e3,
            "k2": 1.4e-5,
            "E2": 42e3,
            "k3": 1.0e-5,
            "E3": 35e3,
            "k_wgs": 2.8e-7,
            "E_wgs": 82e3,
        },
        "micro": {
            "particle_geometry": "sphere",
            "eta_min": 0.03,
            "eta_max": 1.0,
            "D_eff_min": 1e-10,
            "D_eff_max": 5e-3,
            "macro_accessibility": 0.03,
            "kinetic_scale": fs["reactivity_factor"],
        },
        "targets": {
            "S_BET_min_m2g": 750.0,
            "Y_AU_min_pct": 60.0,
            "X_min_pct": 30.0,
            "X_max_pct": 40.0,
            "gas_residual_max_pct": 10.0,
            "T_s_max_C": 980.0,
        },
        "meta": {
            "feedstock_profile": fs,
            "version": "v3.2-wrapper",
            "output_dir": "section_2_5_output",
        },
    }


def apply_feedstock(base_params: dict, feedstock: str) -> dict:
    p = deepcopy(base_params)
    fs = get_feedstock_params(feedstock)
    p["feedstock"] = feedstock
    p["reactor"]["rho_bulk_0"] = fs["rho_bulk_0"]
    p["rpm"]["S0"] = fs["S0"]
    p["rpm"]["eps0"] = fs["eps0"]
    p["micro"]["kinetic_scale"] = fs["reactivity_factor"]
    p["meta"]["feedstock_profile"] = fs
    p["pre_stages"]["rho_char_0"] = fs["rho_bulk_0"]
    p["pre_stages"]["porosity_0"] = fs["eps0"]
    return p


def validate_params(params: dict) -> list[str]:
    warnings = []

    def _check(name: str, value: float):
        lo, hi = RANGES[name]
        if value < lo or value > hi:
            warnings.append(f"{name}={value} вне рекомендуемого диапазона [{lo}, {hi}].")

    op = params["operating"]
    r = params["reactor"]
    m = params["mesh"]
    pre = params["pre_stages"]

    _check("T_steam_in_C", op["T_steam_in_C"])
    _check("T_wall_C", op["T_wall_C"])
    _check("T_char_in_C", pre["T_char_in_C"])
    _check("G_steam_in", op["G_steam_in"])
    _check("H", r["H"])
    _check("L", r["L"])
    _check("W_s", r["W_s"])
    _check("P_reactor", op["P_reactor"])
    _check("N_h", m["N_h"])
    _check("N_l", m["N_l"])
    _check("d_p_0", r["d_p_0"])
    _check("binder_pct", pre["binder_pct"])

    if op["T_steam_in_C"] < pre["T_char_in_C"]:
        warnings.append("T_steam_in_C ниже температуры карбонизата на входе: проверьте постановку режима.")

    if m["N_h"] < 20 or m["N_l"] < 12:
        warnings.append("Сетка грубая: возможна повышенная газовая невязка и потеря детализации полей.")

    return warnings
