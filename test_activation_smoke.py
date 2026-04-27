"""Smoke-test для локальной проверки физической эквивалентности v3.2.

Запуск:
    python test_activation_smoke.py
"""

from __future__ import annotations

import json
from pathlib import Path



TARGETS = {
    "X_mean_target": 33.45,
    "Y_AU_target": 66.55,
    "S_BET_target": 838.0,
    "gas_residual_target": 6.5,
}

TOLERANCES = {
    "X_mean_rel": 0.07,   # ±7%
    "Y_AU_rel": 0.07,     # ±7%
    "S_BET_rel": 0.10,    # ±10%
    "gas_residual_max": 10.0,
}


def _within_rel(value: float, target: float, rel_tol: float) -> bool:
    return abs(value - target) <= abs(target) * rel_tol


def run_smoke() -> int:
    from activation_core import solve_activation
    from activation_params import get_default_params
    scenario_path = Path("examples/default_sunflower_husk.json")
    if scenario_path.exists():
        params = json.loads(scenario_path.read_text(encoding="utf-8"))
    else:
        params = get_default_params()
        params["feedstock"] = "Лузга подсолнечника"
        params["operating"]["T_steam_in_C"] = 920.0
        # Температуру стенки оставляем как в текущем default/v3.2-wrapper
        params["operating"]["G_steam_in"] = 0.300
        params["reactor"]["H"] = 2.0
        params["reactor"]["L"] = 0.5
        params["reactor"]["W_s"] = 2.00e-5

    result = solve_activation(params)

    passport = result["passport"]
    fields = result["fields"]

    x_mean = float(passport["Burnoff_Mean_Pct"])
    y_au = float(passport["Yield_Pct"])
    s_bet = float(passport["BET_Surface_m2g"])
    gas_residual = float(result["mass_balance"]["gas_error_pct"])

    ts_min = float(fields["Ts_2d"].min() - 273.15)
    ts_max = float(fields["Ts_2d"].max() - 273.15)
    tg_min = float(fields["Tg_2d"].min() - 273.15)
    tg_max = float(fields["Tg_2d"].max() - 273.15)

    print("=== Расчётные значения ===")
    print(f"X_mean = {x_mean:.3f} %")
    print(f"Y_AU = {y_au:.3f} %")
    print(f"S_BET = {s_bet:.3f} m2/g")
    print(f"gas_residual = {gas_residual:.3f} %")
    print(f"Ts_min = {ts_min:.3f} C, Ts_max = {ts_max:.3f} C")
    print(f"Tg_min = {tg_min:.3f} C, Tg_max = {tg_max:.3f} C")

    print("\n=== Контрольные значения v3.2 ===")
    print(f"X_mean_target = {TARGETS['X_mean_target']:.2f}")
    print(f"Y_AU_target = {TARGETS['Y_AU_target']:.2f}")
    print(f"S_BET_target = {TARGETS['S_BET_target']:.2f}")
    print(f"gas_residual_target = {TARGETS['gas_residual_target']:.2f}")

    checks = {
        "X_mean": _within_rel(x_mean, TARGETS["X_mean_target"], TOLERANCES["X_mean_rel"]),
        "Y_AU": _within_rel(y_au, TARGETS["Y_AU_target"], TOLERANCES["Y_AU_rel"]),
        "S_BET": _within_rel(s_bet, TARGETS["S_BET_target"], TOLERANCES["S_BET_rel"]),
        "gas_residual": gas_residual <= TOLERANCES["gas_residual_max"],
    }

    print("\n=== Проверка допусков ===")
    print(f"X_mean ±7%: {'OK' if checks['X_mean'] else 'FAIL'}")
    print(f"Y_AU ±7%: {'OK' if checks['Y_AU'] else 'FAIL'}")
    print(f"S_BET ±10%: {'OK' if checks['S_BET'] else 'FAIL'}")
    print(f"gas_residual <= 10%: {'OK' if checks['gas_residual'] else 'FAIL'}")

    all_ok = all(checks.values())
    if not all_ok:
        print(
            "\nФизическая эквивалентность v3.2 не подтверждена. "
            "Проверьте параметры, единицы измерения и перенос коэффициентов."
        )
        return 1

    print("\nФизическая эквивалентность v3.2 подтверждена в заданных допусках.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(run_smoke())
    except ModuleNotFoundError as exc:
        print(
            "Не удалось запустить smoke-test: отсутствуют зависимости. "
            "Создайте локальное venv и установите requirements.txt."
        )
        print(f"Исходная ошибка: {exc}")
        raise SystemExit(2)
