"""Минимальная проверка импортов ключевых модулей."""

from __future__ import annotations

MODULES = [
    "activation_params",
    "activation_core",
    "activation_plots",
    "activation_report",
    "activation_validation",
    "activation_parametric",
]


def main() -> int:
    try:
        for name in MODULES:
            __import__(name)
        print("All core modules imported successfully.")
        return 0
    except ModuleNotFoundError as exc:
        print(f"Import check skipped: missing dependency ({exc}). Install requirements.txt locally.")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
