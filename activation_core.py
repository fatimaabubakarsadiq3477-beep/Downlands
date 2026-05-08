"""Расчетное ядро для паровой активации (wrapper над legacy v3.2).

Физические зависимости legacy-скрипта не изменяются: модуль использует
исходные классы и алгоритмы через динамическую загрузку.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import importlib.util
import json

import numpy as np

from activation_params import get_default_params, apply_feedstock, validate_params, get_feedstock_names
from activation_validation import (
    validate_params as validate_physical_params,
    validate_results as validate_physical_results,
    classify_regime,
    compute_engineering_indices,
    interpret_regime,
)


LEGACY_FILENAME = "2.5f_revised_v3_2 (1) (1).py"


def _deep_update(base: dict, patch: dict) -> dict:
    out = deepcopy(base)
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out


def _to_legacy_config(params: dict) -> dict:
    p = deepcopy(params)
    if p.get("feedstock"):
        p = apply_feedstock(p, p["feedstock"])

    cfg = {
        "output_dir": p["meta"].get("output_dir", "section_2_5_output"),
        "operating": {
            "G_steam_in": p["operating"]["G_steam_in"],
            "T_steam_in_C": p["operating"]["T_steam_in_C"],
            "T_wall_C": p["operating"]["T_wall_C"],
            "P_reactor": p["operating"]["P_reactor"],
        },
        "mesh": {
            "N_h": int(p["mesh"]["N_h"]),
            "N_l": int(p["mesh"]["N_l"]),
        },
        "reactor": {
            "H": p["reactor"]["H"],
            "L": p["reactor"]["L"],
            "W_s": p["reactor"]["W_s"],
            "rho_bulk_0": p["reactor"]["rho_bulk_0"],
            "d_p_0": p["reactor"]["d_p_0"],
        },
        "rpm": deepcopy(p["rpm"]),
        "kinetics": deepcopy(p["kinetics"]),
        "micro": deepcopy(p["micro"]),
    }
    return cfg


_LEGACY_MODULE = None


def _load_legacy_module():
    global _LEGACY_MODULE
    if _LEGACY_MODULE is not None:
        return _LEGACY_MODULE

    path = Path(__file__).resolve().parent / LEGACY_FILENAME
    spec = importlib.util.spec_from_file_location("legacy_activation_v32", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    _LEGACY_MODULE = mod
    return mod


def validate_results(results: dict, params: dict) -> list[str]:
    return validate_physical_results(results, params)


def _calc_engineering_indicators(results: dict) -> dict:
    idx = compute_engineering_indices(results, {})
    return {
        "I_act": idx.get("I_act"),
        "K_ret": idx.get("K_ret"),
        "strength_factor_model": idx.get("strength_factor"),
        "K_quality": idx.get("K_quality"),
        "K_T": idx.get("K_T"),
        "label": "engineering indicators / научно-инженерные критерии, требующие экспериментальной калибровки",
    }


def _run_single(mod, legacy_config: dict):
    thermo = mod.NASAThermoCore(mod.GlobalConst)
    transport = mod.KineticTransportTheory(thermo)
    stefan = mod.MulticomponentDiffusionSolver(transport)
    rpm = mod.RandomPoreEvolutionModel(legacy_config)
    micro = mod.MicroScaleNumericalCore(thermo, transport, stefan, rpm, legacy_config)
    bed = mod.BedThermalDynamics(thermo, transport)
    reactor = mod.CrossFlowActivationReactor(micro, bed, thermo, transport, legacy_config)

    op = legacy_config["operating"]
    mesh = legacy_config["mesh"]
    sol = reactor.solve_reactor(
        v_gas_in=op["G_steam_in"],
        T_steam_in=op["T_steam_in_C"] + 273.15,
        T_wall=op["T_wall_C"] + 273.15,
        N_h_steps=mesh["N_h"],
        P_reactor=op["P_reactor"],
    )

    recon = mod.FieldReconstructionEngine(
        reactor,
        sol,
        op["G_steam_in"],
        op["T_steam_in_C"] + 273.15,
        op["P_reactor"],
    ).reconstruct_all_fields()

    metrics = mod.PerformanceMetricsCalculator(recon)
    passport = metrics.generate_reactor_passport()

    val = mod.DataExportAndValidation(recon, legacy_config["output_dir"])
    solid_err, gas_err = val.check_global_mass_balance()
    q_gas, q_sol, q_rxn = val.check_global_energy_balance()

    fields = {
        "h_grid": recon.h_grid,
        "l_grid": recon.l_grid,
        "Ts_2d": recon.Ts_2d,
        "Tg_2d": recon.Tg_2d,
        "X_2d": recon.X_2d,
        "S_v_2d": recon.S_v_2d,
        "S_BET_2d": np.vectorize(mod.estimate_bet_surface_from_sv)(recon.S_v_2d, recon.rm.rho_bulk_0, recon.X_2d),
        "P_h2o": recon.P_h2o,
        "P_co": recon.P_co,
        "P_h2": recon.P_h2,
        "P_co2": recon.P_co2,
        "y_H2O_2d": recon.P_h2o / op["P_reactor"],
        "y_CO_2d": recon.P_co / op["P_reactor"],
        "y_H2_2d": recon.P_h2 / op["P_reactor"],
        "y_CO2_2d": recon.P_co2 / op["P_reactor"],
        "eta_2d": recon.eta_2d,
        "r_act_2d": recon.R_carb_2d,
        "Q_conv_2d": recon.Q_conv_2d,
        "rho_bed_2d": recon.rho_bed_2d,
        "D_eff_2d": recon.Deff_2d,
        "dpore_2d": recon.dpore_2d,
    }

    out = {
        "passport": passport,
        "mass_balance": {
            "solid_error_pct": float(solid_err),
            "gas_error_pct": float(gas_err),
        },
        "energy_balance": {
            "Q_gas_W": float(q_gas),
            "Q_solid_W": float(q_sol),
            "Q_rxn_W": float(q_rxn),
        },
        "fields": fields,
        "_objects": {
            "module": mod,
            "reactor": reactor,
            "solution": sol,
            "recon": recon,
            "metrics": metrics,
            "validation": val,
        },
    }
    return out


def solve_activation(params: dict | None = None) -> dict:
    base = get_default_params()
    user = params or {}
    merged = _deep_update(base, user)
    param_warnings = validate_params(merged) + validate_physical_params(merged)

    legacy_cfg = _to_legacy_config(merged)
    mod = _load_legacy_module()
    result = _run_single(mod, legacy_cfg)

    result["params"] = merged
    result["engineering_indicators"] = _calc_engineering_indicators(result)
    result_warnings = validate_results(result, merged)
    result["warnings"] = param_warnings + result_warnings

    status = classify_regime(result, merged)
    result["status"] = status
    result["interpretation"] = interpret_regime(status, result)

    return result


def run_parametric_analysis(params: dict, mode: str, values: list[float]) -> list[dict]:
    rows = []
    prev = None
    for v in values:
        p = deepcopy(params)
        if mode == "T_steam_in_C":
            p["operating"]["T_steam_in_C"] = float(v)
        elif mode == "G_steam_in":
            p["operating"]["G_steam_in"] = float(v)
        elif mode == "particle_size_mm":
            p["pre_stages"]["particle_size_mm"] = float(v)
            p["reactor"]["d_p_0"] = float(v) / 1000.0
        elif mode == "rho_pellet_kgm3":
            p["pre_stages"]["rho_pellet_kgm3"] = float(v)
            p["reactor"]["rho_bulk_0"] = float(v)
        else:
            raise ValueError(f"Неизвестный режим параметрики: {mode}")

        res = solve_activation(p)
        row = {
            "value": float(v),
            "X_pct": res["passport"]["Burnoff_Mean_Pct"],
            "Y_AU_pct": res["passport"]["Yield_Pct"],
            "S_BET_m2g": res["passport"]["BET_Surface_m2g"],
            "eta_mean": float(np.mean(res["fields"]["eta_2d"])),
            "gas_residual_pct": res["mass_balance"]["gas_error_pct"],
        }
        if prev is not None and mode == "T_steam_in_C":
            dT = row["value"] - prev["value"]
            dS = row["S_BET_m2g"] - prev["S_BET_m2g"]
            row["K_T"] = dS / dT if abs(dT) > 1e-9 else np.nan
        else:
            row["K_T"] = np.nan
        rows.append(row)
        prev = row
    return rows


def optimize_rational_mode(params: dict, grid: dict | None = None) -> dict:
    g = grid or {
        "T_steam_in_C": [800, 840, 880, 920],
        "G_steam_in": [0.18, 0.25, 0.32],
        "W_s": [1.6e-4, 2.0e-4, 2.4e-4],
        "rho_bulk_0": [params["reactor"]["rho_bulk_0"], params["reactor"]["rho_bulk_0"] * 0.9],
    }

    tgt = params.get("targets", {})
    best = None
    all_rows = []

    for t in g["T_steam_in_C"]:
        for flow in g["G_steam_in"]:
            for ws in g["W_s"]:
                for rho in g["rho_bulk_0"]:
                    p = deepcopy(params)
                    p["operating"]["T_steam_in_C"] = float(t)
                    p["operating"]["G_steam_in"] = float(flow)
                    p["reactor"]["W_s"] = float(ws)
                    p["reactor"]["rho_bulk_0"] = float(rho)

                    r = solve_activation(p)
                    x = r["passport"]["Burnoff_Mean_Pct"]
                    y = r["passport"]["Yield_Pct"]
                    s = r["passport"]["BET_Surface_m2g"]
                    gas = r["mass_balance"]["gas_error_pct"]
                    ts_c = r["passport"]["Ts_Exit_Mean_K"] - 273.15

                    penalties = [
                        max(0.0, tgt.get("S_BET_min_m2g", 0.0) - s),
                        max(0.0, tgt.get("Y_AU_min_pct", 0.0) - y),
                        max(0.0, tgt.get("X_min_pct", 0.0) - x),
                        max(0.0, x - tgt.get("X_max_pct", 100.0)),
                        max(0.0, gas - tgt.get("gas_residual_max_pct", 100.0)) * 8.0,
                        max(0.0, ts_c - tgt.get("T_s_max_C", 1e6)) * 0.2,
                    ]
                    score = float(sum(penalties))
                    row = {
                        "T_steam_C": t,
                        "steam_flow": flow,
                        "residence_time_min": p["reactor"]["H"] / p["reactor"]["W_s"] / 60.0,
                        "X_pct": x,
                        "Y_AU_pct": y,
                        "S_BET_m2g": s,
                        "gas_residual_pct": gas,
                        "warnings": "; ".join(r["warnings"][:3]),
                        "score": score,
                    }
                    all_rows.append(row)
                    if best is None or row["score"] < best["score"]:
                        best = row

    return {"best": best, "table": all_rows}


def compare_all_feedstocks(params: dict) -> list[dict]:
    rows = []
    for name in get_feedstock_names():
        p = deepcopy(params)
        p["feedstock"] = name
        r = solve_activation(p)
        rows.append(
            {
                "feedstock": name,
                "X_pct": r["passport"]["Burnoff_Mean_Pct"],
                "Y_AU_pct": r["passport"]["Yield_Pct"],
                "S_BET_m2g": r["passport"]["BET_Surface_m2g"],
                "I_act": r["engineering_indicators"]["I_act"],
                "recommended_regime": r["status"],
            }
        )
    return rows


def to_jsonable(results: dict) -> dict:
    def conv(v):
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, (np.floating, np.integer)):
            return float(v)
        if isinstance(v, dict):
            return {k: conv(x) for k, x in v.items() if k != "_objects"}
        if isinstance(v, list):
            return [conv(x) for x in v]
        return v

    return conv(results)


def save_scenario(params: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(params, f, ensure_ascii=False, indent=2)


def load_scenario(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
