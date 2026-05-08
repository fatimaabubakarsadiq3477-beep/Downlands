"""Построение графиков по результатам solve_activation."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def _field_plot(h, l, z, title, cbar, cmap="viridis"):
    L, H = np.meshgrid(l, h)
    fig, ax = plt.subplots(figsize=(8, 5.6))
    cs = ax.contourf(L, H, z, levels=40, cmap=cmap)
    ax.invert_yaxis()
    ax.set_xlabel("l, м")
    ax.set_ylabel("h, м")
    ax.set_title(title)
    cb = fig.colorbar(cs, ax=ax)
    cb.set_label(cbar)
    fig.tight_layout()
    return fig


def build_field_figures(results: dict) -> dict:
    f = results["fields"]
    h = f["h_grid"]
    l = f["l_grid"]
    figs = {}
    if "Ts_2d" in f:
        figs["Ts"] = _field_plot(h, l, f["Ts_2d"] - 273.15, "Ts(h,l)", "°C", "inferno")
    if "Tg_2d" in f:
        figs["Tg"] = _field_plot(h, l, f["Tg_2d"] - 273.15, "Tg(h,l)", "°C", "magma")
    if "X_2d" in f:
        figs["X"] = _field_plot(h, l, f["X_2d"] * 100.0, "X(h,l)", "%", "viridis")
    if "S_BET_2d" in f:
        figs["S_BET"] = _field_plot(h, l, f["S_BET_2d"], "S_BET(h,l)", "м²/г", "plasma")
    if "P_h2o" in f:
        figs["P_H2O"] = _field_plot(h, l, f["P_h2o"] / 1000.0, "P_H2O(h,l)", "кПа", "YlGnBu")
    if "eta_2d" in f:
        figs["eta"] = _field_plot(h, l, f["eta_2d"], "η(h,l)", "-", "cubehelix")
    if "r_act_2d" in f:
        figs["r_act"] = _field_plot(h, l, f["r_act_2d"], "r_act(h,l)", "моль/(м³·с)", "Reds")
    if "Q_conv_2d" in f:
        figs["Q_conv"] = _field_plot(h, l, f["Q_conv_2d"] / 1000.0, "Q_conv(h,l)", "кВт/м³", "RdBu_r")
    if "rho_bed_2d" in f:
        figs["rho_bed"] = _field_plot(h, l, f["rho_bed_2d"], "ρ_bed(h,l)", "кг/м³", "cividis")
    if "D_eff_2d" in f:
        figs["D_eff"] = _field_plot(h, l, f["D_eff_2d"] * 1e6, "D_eff(h,l)", "мм²/с", "PuBuGn")
    return figs


def build_gas_composition_plot(results: dict):
    f = results["fields"]
    h = f["h_grid"]
    l = f["l_grid"]
    idx = int(np.argmin(np.abs(h - np.median(h))))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(l, f["y_H2O_2d"][idx, :], label="H2O")
    ax.plot(l, f["y_CO_2d"][idx, :], label="CO")
    ax.plot(l, f["y_H2_2d"][idx, :], label="H2")
    ax.plot(l, f["y_CO2_2d"][idx, :], label="CO2")
    ax.set_xlabel("l, м")
    ax.set_ylabel("Мольная доля")
    ax.set_title("Состав газа по ширине слоя")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def build_parametric_plot(rows: list[dict], x_label: str, x_key: str = "value"):
    x = [r[x_key] for r in rows]
    y1 = [r["X_pct"] for r in rows]
    y2 = [r["S_BET_m2g"] for r in rows]

    fig, ax1 = plt.subplots(figsize=(8, 5.5))
    ax1.plot(x, y1, "o-", color="tab:red", label="X, %")
    ax1.set_xlabel(x_label)
    ax1.set_ylabel("X, %", color="tab:red")
    ax1.tick_params(axis="y", labelcolor="tab:red")

    ax2 = ax1.twinx()
    ax2.plot(x, y2, "s--", color="tab:blue", label="S_BET")
    ax2.set_ylabel("S_BET, м²/г", color="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:blue")

    fig.tight_layout()
    return fig


def save_figures_png(figures: dict, out_dir: str, dpi: int = 300) -> list[str]:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    paths = []
    for name, fig in figures.items():
        p = Path(out_dir) / f"{name}.png"
        fig.savefig(p, dpi=dpi, bbox_inches="tight")
        paths.append(str(p))
    return paths


def plot_sweep_line(df, x_col, y_col, title, x_label, y_label):
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    if x_col in df.columns and y_col in df.columns:
        ax.plot(df[x_col], df[y_col], marker="o")
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_feedstock_bar(df, value_col, title, y_label):
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    if "feedstock" in df.columns and value_col in df.columns:
        ax.bar(df["feedstock"], df[value_col])
        ax.tick_params(axis="x", rotation=35)
    ax.set_title(title)
    ax.set_ylabel(y_label)
    ax.grid(alpha=0.2, axis="y")
    fig.tight_layout()
    return fig


def plot_optimization_scores(df):
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    if "score" in df.columns:
        ax.plot(range(1, len(df) + 1), df["score"], marker="o")
    ax.set_title("Скоринг режимов оптимизации")
    ax.set_xlabel("Ранг режима")
    ax.set_ylabel("Score")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def save_all_figures(results, parametric_results, output_dir, dpi=300) -> list[str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    created = []

    field_figs = build_field_figures(results)
    mapping = {
        "Ts": "4_5_field_Ts.png",
        "Tg": "4_5_field_Tg.png",
        "X": "4_5_field_X.png",
        "S_BET": "4_5_field_S_BET.png",
        "P_H2O": "4_5_field_P_H2O.png",
        "eta": "4_5_field_eta.png",
        "r_act": "4_5_field_r_act.png",
        "rho_bed": "4_5_field_rho_bed.png",
        "D_eff": "4_5_field_D_eff.png",
    }
    for key, fname in mapping.items():
        if key in field_figs:
            p = out / fname
            field_figs[key].savefig(p, dpi=dpi, bbox_inches="tight")
            created.append(str(p))

    if parametric_results:
        tdf = parametric_results.get("temperature_sweep")
        if tdf is not None and not tdf.empty:
            p = out / "4_5_temperature_sweep.png"
            plot_sweep_line(tdf, "T_steam", "S_BET", "Температурная серия", "T_steam, °C", "S_BET, м²/г").savefig(p, dpi=dpi, bbox_inches="tight")
            created.append(str(p))

        fdf = parametric_results.get("steam_flow_sweep")
        if fdf is not None and not fdf.empty:
            p = out / "4_5_steam_flow_sweep.png"
            plot_sweep_line(fdf, "steam_flow", "S_BET", "Серия по расходу пара", "steam_flow, м/с", "S_BET, м²/г").savefig(p, dpi=dpi, bbox_inches="tight")
            created.append(str(p))

        fsdf = parametric_results.get("feedstock_compare")
        if fsdf is not None and not fsdf.empty:
            p = out / "4_5_feedstock_comparison.png"
            plot_feedstock_bar(fsdf, "S_BET", "Сравнение сырья", "S_BET, м²/г").savefig(p, dpi=dpi, bbox_inches="tight")
            created.append(str(p))

        odf = parametric_results.get("optimization")
        if odf is not None and not odf.empty:
            p = out / "4_5_optimization_scores.png"
            plot_optimization_scores(odf).savefig(p, dpi=dpi, bbox_inches="tight")
            created.append(str(p))

    return created
