"""Параметрический анализ и оптимизация режима (надстройка над solve_activation)."""

from __future__ import annotations

from copy import deepcopy
from itertools import product

import numpy as np
try:
    import pandas as pd
except ModuleNotFoundError:  # pragma: no cover
    pd = None

from activation_core import solve_activation
from activation_params import apply_feedstock, get_feedstock_names


DEFAULT_CONSTRAINTS = {
    "S_BET_min": 800.0,
    "Y_AU_min": 60.0,
    "X_min": 30.0,
    "X_max": 40.0,
    "gas_residual_max": 10.0,
    "T_steam_max": 1000.0,
    "prefer_strength": True,
}


def _require_pandas():
    if pd is None:
        raise RuntimeError("pandas не установлен: параметрический анализ недоступен. Установите зависимости из requirements.txt.")


def _summary_from_result(result: dict) -> dict:
    p = result.get("passport", {})
    m = result.get("mass_balance", {})
    idx = result.get("engineering_indicators", {})
    eta = result.get("fields", {}).get("eta_2d")
    eta_mean = float(np.mean(eta)) if eta is not None else None
    warnings = result.get("warnings", [])

    return {
        "X_mean": p.get("Burnoff_Mean_Pct"),
        "Y_AU": p.get("Yield_Pct"),
        "S_BET": p.get("BET_Surface_m2g"),
        "gas_residual": m.get("gas_error_pct"),
        "I_act": idx.get("I_act"),
        "K_ret": idx.get("K_ret"),
        "K_quality": idx.get("K_quality"),
        "strength_factor": idx.get("strength_factor_model", idx.get("strength_factor")),
        "eta_mean": eta_mean,
        "status": result.get("status"),
        "warnings_count": len(warnings),
        "top_warning": warnings[0] if warnings else None,
    }


def _run_case(params: dict) -> tuple[dict, str | None]:
    try:
        result = solve_activation(params)
        return _summary_from_result(result), None
    except Exception as exc:  # noqa: BLE001
        return {
            "X_mean": None,
            "Y_AU": None,
            "S_BET": None,
            "gas_residual": None,
            "I_act": None,
            "K_ret": None,
            "K_quality": None,
            "strength_factor": None,
            "eta_mean": None,
            "status": "error",
            "warnings_count": 1,
            "top_warning": str(exc),
        }, str(exc)


def run_temperature_sweep(base_params: dict, temperatures: list[float]) -> pd.DataFrame:
    _require_pandas()
    rows = []
    for t in temperatures:
        p = deepcopy(base_params)
        p["operating"]["T_steam_in_C"] = float(t)
        s, _ = _run_case(p)
        s["T_steam"] = float(t)
        rows.append(s)
    cols = ["T_steam", "X_mean", "Y_AU", "S_BET", "gas_residual", "I_act", "K_ret", "K_quality", "status", "warnings_count", "top_warning"]
    return pd.DataFrame(rows)[cols]


def run_steam_flow_sweep(base_params: dict, steam_flows: list[float]) -> pd.DataFrame:
    _require_pandas()
    rows = []
    for g in steam_flows:
        p = deepcopy(base_params)
        p["operating"]["G_steam_in"] = float(g)
        s, _ = _run_case(p)
        s["steam_flow"] = float(g)
        rows.append(s)
    cols = ["steam_flow", "X_mean", "Y_AU", "S_BET", "gas_residual", "I_act", "K_ret", "K_quality", "status", "warnings_count", "top_warning"]
    return pd.DataFrame(rows)[cols]


def run_particle_size_sweep(base_params: dict, particle_sizes: list[float]) -> pd.DataFrame:
    _require_pandas()
    rows = []
    for d in particle_sizes:
        p = deepcopy(base_params)
        p["pre_stages"]["particle_size_mm"] = float(d)
        p["reactor"]["d_p_0"] = float(d) / 1000.0
        s, _ = _run_case(p)
        s["d_particle"] = float(d)
        rows.append(s)
    cols = ["d_particle", "X_mean", "Y_AU", "S_BET", "eta_mean", "gas_residual", "I_act", "K_ret", "status", "warnings_count", "top_warning"]
    return pd.DataFrame(rows)[cols]


def run_density_sweep(base_params: dict, densities: list[float]) -> pd.DataFrame:
    _require_pandas()
    rows = []
    for rho in densities:
        p = deepcopy(base_params)
        p["pre_stages"]["rho_pellet_kgm3"] = float(rho)
        p["reactor"]["rho_bulk_0"] = float(rho)
        s, _ = _run_case(p)
        s["rho_granule"] = float(rho)
        rows.append(s)
    cols = ["rho_granule", "X_mean", "Y_AU", "S_BET", "strength_factor", "K_quality", "status", "warnings_count", "top_warning"]
    return pd.DataFrame(rows)[cols]


def compare_feedstocks(base_params: dict, feedstocks: list[str] | None = None) -> pd.DataFrame:
    _require_pandas()
    rows = []
    feedstocks = feedstocks or get_feedstock_names()
    for fs in feedstocks:
        p = apply_feedstock(base_params, fs)
        s, _ = _run_case(p)
        s["feedstock"] = fs
        rows.append(s)
    cols = ["feedstock", "X_mean", "Y_AU", "S_BET", "I_act", "K_ret", "K_quality", "gas_residual", "status", "top_warning", "warnings_count"]
    return pd.DataFrame(rows)[cols]


def score_regime(summary: dict, constraints: dict) -> float:
    c = {**DEFAULT_CONSTRAINTS, **(constraints or {})}

    x = summary.get("X_mean")
    y = summary.get("Y_AU")
    s = summary.get("S_BET")
    gas = summary.get("gas_residual")
    t = summary.get("T_steam")
    kret = summary.get("K_ret")
    kq = summary.get("K_quality")
    wc = summary.get("warnings_count", 0) or 0

    if any(v is None for v in [x, y, s, gas]):
        return -1e6

    score = 0.0

    # hard constraints penalties
    if s < c["S_BET_min"]:
        score -= 300 + (c["S_BET_min"] - s) * 0.5
    if y < c["Y_AU_min"]:
        score -= 300 + (c["Y_AU_min"] - y) * 3.0
    if gas > c["gas_residual_max"]:
        score -= 500 + (gas - c["gas_residual_max"]) * 20.0
    if t is not None and t > c["T_steam_max"]:
        score -= 400 + (t - c["T_steam_max"]) * 5.0

    # soft targets
    x_mid = (c["X_min"] + c["X_max"]) / 2.0
    if c["X_min"] <= x <= c["X_max"]:
        score += 150.0
    else:
        score -= abs(x - x_mid) * 8.0

    score += max(0.0, s - c["S_BET_min"]) * 0.2
    score += max(0.0, y - c["Y_AU_min"]) * 2.0
    score += (10.0 - min(gas, 10.0)) * 6.0

    if kret is not None:
        score += float(kret) * 0.02
    if kq is not None:
        score += float(kq) * 0.05

    if c.get("prefer_strength", True):
        sf = summary.get("strength_factor")
        if sf is not None:
            score += float(sf) * 40.0

    score -= wc * 8.0
    return float(score)


def optimize_activation(
    base_params: dict,
    search_space: dict,
    constraints: dict,
    max_cases: int = 50,
) -> pd.DataFrame:
    _require_pandas()
    t_vals = list(search_space.get("T_steam", [base_params["operating"]["T_steam_in_C"]]))
    g_vals = list(search_space.get("steam_flow", [base_params["operating"]["G_steam_in"]]))
    ws_vals = list(search_space.get("bed_velocity", [base_params["reactor"]["W_s"]]))
    rho_vals = list(search_space.get("rho_granule", [base_params["pre_stages"]["rho_pellet_kgm3"]]))
    d_vals = list(search_space.get("d_particle", [base_params["pre_stages"]["particle_size_mm"]]))

    combos = list(product(t_vals, g_vals, ws_vals, rho_vals, d_vals))
    truncated = False
    if len(combos) > max_cases:
        combos = combos[:max_cases]
        truncated = True

    rows = []
    c = {**DEFAULT_CONSTRAINTS, **(constraints or {})}
    for t, g, ws, rho, d in combos:
        p = deepcopy(base_params)
        p["operating"]["T_steam_in_C"] = float(t)
        p["operating"]["G_steam_in"] = float(g)
        p["reactor"]["W_s"] = float(ws)
        p["pre_stages"]["rho_pellet_kgm3"] = float(rho)
        p["reactor"]["rho_bulk_0"] = float(rho)
        p["pre_stages"]["particle_size_mm"] = float(d)
        p["reactor"]["d_p_0"] = float(d) / 1000.0

        s, err = _run_case(p)
        s.update({
            "T_steam": float(t),
            "steam_flow": float(g),
            "bed_velocity": float(ws),
            "residence_time": p["reactor"]["H"] / max(p["reactor"]["W_s"], 1e-9) / 60.0,
            "rho_granule": float(rho),
            "d_particle": float(d),
        })

        feasible = (
            err is None
            and s["S_BET"] is not None and s["S_BET"] >= c["S_BET_min"]
            and s["Y_AU"] is not None and s["Y_AU"] >= c["Y_AU_min"]
            and s["X_mean"] is not None and c["X_min"] <= s["X_mean"] <= c["X_max"]
            and s["gas_residual"] is not None and s["gas_residual"] <= c["gas_residual_max"]
            and s["T_steam"] <= c["T_steam_max"]
        )
        s["feasible"] = bool(feasible)
        s["score"] = score_regime(s, c)
        rows.append(s)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("score", ascending=False).reset_index(drop=True)
    df.attrs["truncated"] = truncated
    df.attrs["max_cases"] = max_cases
    return df
