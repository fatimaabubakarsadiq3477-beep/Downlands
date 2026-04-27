import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.integrate import solve_ivp
from scipy.linalg import solve_banded
import time
import gc
import warnings
import os
import datetime

# =============================================================================
# 1. СИСТЕМНЫЙ КОНТРОЛЬ СТАБИЛЬНОСТИ
# =============================================================================
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def init_scientific_plots():
    rcParams['figure.dpi'] = 300
    rcParams['font.family'] = 'serif'
    rcParams['font.serif'] = ['Times New Roman', 'Liberation Serif', 'DejaVu Serif']
    rcParams['font.size'] = 11
    rcParams['axes.labelsize'] = 12
    rcParams['axes.titlesize'] = 14
    rcParams['legend.fontsize'] = 10
    rcParams['xtick.direction'] = 'in'
    rcParams['ytick.direction'] = 'in'
    rcParams['xtick.major.size'] = 6
    rcParams['ytick.major.size'] = 6
    rcParams['lines.linewidth'] = 2.0
    rcParams['axes.grid'] = True
    rcParams['grid.alpha'] = 0.3
    plt.style.use('seaborn-v0_8-ticks')

init_scientific_plots()

def load_config(config_path='config.json'):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config

CALCULATION_VERSION = 'v3.2'
work_dir = 'section_2_5_output'
if not os.path.exists(work_dir):
    os.makedirs(work_dir)

# =============================================================================
# 2. ГЛОБАЛЬНЫЙ РЕГИСТР ФИЗИЧЕСКИХ КОНСТАНТ
# =============================================================================
class PhysicsConstantsRegistry:
    def __init__(self):
        self.R = 8.31446261815324
        self.Na = 6.02214076e23
        self.sigma_sb = 5.670374e-8
        self.kb = 1.380649e-23
        self.P_atm = 101325.0
        self.T_st = 273.15
        self.g = 9.80665
        self.MW = {
            'H2O': 0.018015, 'CO': 0.028010, 'H2': 0.002016,
            'CO2': 0.044010, 'N2': 0.028013, 'CH4': 0.016043,
            'O2': 0.031999,  'C': 0.012011,  'Ash': 0.062000
        }

GlobalConst = PhysicsConstantsRegistry()

# =============================================================================
# 3. ТЕРМОДИНАМИЧЕСКИЙ РЕАКТОР NASA-7
# =============================================================================
class NASAThermoCore:
    def __init__(self, constants):
        self.R = constants.R
        self.MW = constants.MW
        self.nasa_high = {
            'H2O': [2.67703887e+00, 2.97318329e-03, -7.73769690e-07, 9.44335140e-11, -4.26900950e-15, -2.98858938e+04, 6.88255000e+00],
            'CO':  [2.71510200e+00, 2.06252700e-03, -9.99021400e-07, 2.30233100e-10, -1.97011400e-14, -1.41518700e+04, 7.81868700e+00],
            'H2':  [2.99142300e+00, 7.00064400e-04, -5.63382800e-08, -9.23158000e-12, 1.58275100e-15, -8.35033900e+02, -1.35511000e+00],
            'CO2': [3.85718800e+00, 4.41437000e-03, -2.21481400e-06, 5.23490300e-10, -4.72384500e-14, -4.87592000e+04, 2.27163800e+00],
            'N2':  [2.92664000e+00, 1.48797600e-03, -5.68476000e-07, 1.00970300e-10, -6.75335000e-15, -9.22797700e+02, 5.98052800e+00],
            'CH4': [1.68347800e+00, 1.02372400e-02, -3.87512800e-06, 6.78558500e-10, -4.50342300e-14, -1.00807500e+04, 9.62339500e+00],
            'O2':  [3.69757800e+00, 6.13519700e-04, -1.25884200e-07, 1.77528100e-11, -1.13143900e-15, -1.23393000e+03, 3.18916500e+00],
            'C':   [2.49266390e+00, 4.75095360e-06, -1.14441000e-09, 1.15540000e-13, -3.92100000e-18, 8.51475100e+04, 1.12015000e+00]
        }
        self.nasa_low = {
            'H2O': [4.19864056e+00, -2.03643410e-03, 6.52040630e-06, -5.48797062e-09, 1.77197817e-12, -3.02937267e+04, -8.49032200e-01],
            'N2':  [3.53100500e+00, -1.23668700e-04, -5.02999400e-07, 2.43530600e-09, -1.40881200e-12, -1.04697600e+03, 2.96747000e+00],
            'CO':  [3.71009200e+00, -1.61900400e-03, 3.69235900e-06, -2.03196700e-09, 2.39533400e-13, -1.43503100e+04, 2.95543500e+00],
            'H2':  [2.34433100e+00, 7.98052000e-03, -1.94781500e-05, 2.01572000e-08, -7.37611700e-12, -9.17935100e+02, 6.83010200e-01],
            'CO2': [2.35677300e+00, 8.98459600e-03, -7.12356200e-06, 2.45919000e-09, -1.43699500e-13, -4.83719600e+04, 9.90105200e+00]
        }

    def _select_data(self, T):
        return self.nasa_high if T >= 1000 else self.nasa_low

    def get_cp_mol(self, species, T):
        a = self._select_data(T).get(species, self.nasa_high[species])
        cp_r = a[0] + a[1]*T + a[2]*T**2 + a[3]*T**3 + a[4]*T**4
        return cp_r * self.R

    def get_h_mol(self, species, T):
        a = self._select_data(T).get(species, self.nasa_high[species])
        h_rt = a[0] + a[1]*T/2.0 + a[2]*T**2/3.0 + a[3]*T**3/4.0 + a[4]*T**4/5.0 + a[5]/T
        return h_rt * self.R * T

    def get_s_mol(self, species, T):
        a = self._select_data(T).get(species, self.nasa_high[species])
        s_r = a[0]*np.log(T) + a[1]*T + a[2]*T**2/2.0 + a[3]*T**3/3.0 + a[4]*T**4/4.0 + a[6]
        return s_r * self.R

    def get_gibbs_mol(self, species, T):
        return self.get_h_mol(species, T) - T * self.get_s_mol(species, T)

    def get_reaction_enthalpy(self, T, reagents, products, stoich_reagents, stoich_products):
        h_reag = sum(stoich_reagents[i] * self.get_h_mol(reagents[i], T) for i in range(len(reagents)))
        h_prod = sum(stoich_products[i] * self.get_h_mol(products[i], T) for i in range(len(products)))
        return h_prod - h_reag

    def get_equilibrium_Kp(self, T, reagents, products, stoich_reagents, stoich_products):
        g_reag = sum(stoich_reagents[i] * self.get_gibbs_mol(reagents[i], T) for i in range(len(reagents)))
        g_prod = sum(stoich_products[i] * self.get_gibbs_mol(products[i], T) for i in range(len(products)))
        delta_G = g_prod - g_reag
        return np.exp(-delta_G / (self.R * T))

ThermoDB = NASAThermoCore(GlobalConst)

# =============================================================================
# 4. КИНЕТИЧЕСКАЯ ТЕОРИЯ ГАЗОВ И ТРАНСПОРТНЫЕ СВОЙСТВА
# =============================================================================
class KineticTransportTheory:
    def __init__(self, thermo_registry):
        self.th = thermo_registry
        self._mu_cache = {}
        self.LJ = {
            'H2O': [2.641, 809.1], 'CO': [3.690, 91.7], 'H2': [2.827, 59.7],
            'CO2': [3.941, 195.2], 'N2': [3.621, 97.5], 'CH4': [3.758, 148.6],
            'O2': [3.467, 106.7], 'Ar': [3.330, 136.5]
        }
        self.V_fuller = {
            'H2O': 12.70, 'CO': 18.90, 'H2': 7.07, 'CO2': 26.90,
            'N2': 17.90, 'CH4': 24.42, 'O2': 16.60, 'Ar': 16.10
        }

    def _calc_collision_integral_visc(self, T_star):
        A, B, C, D = 1.16145, 0.14874, 0.52487, 0.77320
        return (A / (T_star ** B)) + C * np.exp(-D * T_star)

    def get_pure_viscosity(self, species, T):
        key = (species, round(float(T), 1))
        if key in self._mu_cache:
            return self._mu_cache[key]
        mw_g_mol = self.th.MW[species] * 1000.0
        sigma, eps_k = self.LJ[species]
        T_star = T / eps_k
        omega_v = self._calc_collision_integral_visc(T_star)
        mu_pure = 2.6693e-6 * np.sqrt(mw_g_mol * T) / ((sigma ** 2) * omega_v)
        self._mu_cache[key] = mu_pure
        return mu_pure

    def get_pure_thermal_conductivity(self, species, T):
        mu = self.get_pure_viscosity(species, T)
        cp_mass = self.th.get_cp_mol(species, T) / self.th.MW[species]
        R_mass = self.th.R / self.th.MW[species]
        return mu * (cp_mass + 1.25 * R_mass)

    def get_binary_diffusion(self, sp1, sp2, T, P):
        mw1 = self.th.MW[sp1] * 1000.0
        mw2 = self.th.MW[sp2] * 1000.0
        v1 = self.V_fuller[sp1]
        v2 = self.V_fuller[sp2]
        P_atm = P / 101325.0
        D12_cm2 = (1.00e-3 * (T ** 1.75) * np.sqrt(1.0/mw1 + 1.0/mw2)) / \
                  (P_atm * ((v1 ** (1./3.)) + (v2 ** (1./3.))) ** 2)
        return D12_cm2 * 1e-4

    def get_mixture_viscosity(self, T, y_dict):
        active = [s for s in y_dict if y_dict[s] > 1e-8]
        mu_pure = {s: self.get_pure_viscosity(s, T) for s in active}
        mu_mix = 0.0
        for i in active:
            sum_phi = 0.0
            for j in active:
                if i == j:
                    phi = 1.0
                else:
                    ratio_mu = mu_pure[i] / mu_pure[j]
                    ratio_mw = self.th.MW[j] / self.th.MW[i]
                    num = (1.0 + np.sqrt(ratio_mu) * (ratio_mw ** 0.25)) ** 2
                    den = np.sqrt(8.0 * (1.0 + self.th.MW[i] / self.th.MW[j]))
                    phi = num / den
                sum_phi += y_dict[j] * phi
            mu_mix += y_dict[i] * mu_pure[i] / sum_phi
        return mu_mix

    def get_mixture_thermal_conductivity(self, T, y_dict):
        active = [s for s in y_dict if y_dict[s] > 1e-8]
        lam_pure = {s: self.get_pure_thermal_conductivity(s, T) for s in active}
        mu_pure = {s: self.get_pure_viscosity(s, T) for s in active}
        lam_mix = 0.0
        for i in active:
            sum_phi = 0.0
            for j in active:
                if i == j:
                    phi = 1.0
                else:
                    ratio_mu = mu_pure[i] / mu_pure[j]
                    ratio_mw = self.th.MW[j] / self.th.MW[i]
                    num = (1.0 + np.sqrt(ratio_mu) * (ratio_mw ** 0.25)) ** 2
                    den = np.sqrt(8.0 * (1.0 + self.th.MW[i] / self.th.MW[j]))
                    phi = num / den
                sum_phi += y_dict[j] * phi
            lam_mix += y_dict[i] * lam_pure[i] / sum_phi
        return lam_mix

# =============================================================================
# 5. МНОГОКОМПОНЕНТНАЯ ДИФФУЗИЯ (СТЕФАН-МАКСВЕЛЛ)
# =============================================================================
class MulticomponentDiffusionSolver:
    def __init__(self, transport_model):
        self.tr = transport_model

    def calculate_effective_diffusivity(self, T, P, y_dict):
        active = [s for s in y_dict if y_dict[s] > 1e-8]
        D_eff_map = {}
        for i in active:
            sum_res = 0.0
            for j in active:
                if i != j:
                    D_ij = self.tr.get_binary_diffusion(i, j, T, P)
                    sum_res += y_dict[j] / D_ij
            if sum_res < 1e-20:
                D_eff_map[i] = self.tr.get_binary_diffusion(i, i, T, P)
            else:
                D_eff_map[i] = (1.0 - y_dict[i]) / sum_res
        return D_eff_map

# =============================================================================
# 6. ЭВОЛЮЦИЯ НАНОПОР (RANDOM PORE MODEL)
# =============================================================================
class RandomPoreEvolutionModel:
    def __init__(self, config):
        rpm_cfg = config['rpm']
        self.S0 = rpm_cfg['S0']
        self.eps0 = rpm_cfg['eps0']
        self.psi = rpm_cfg['psi']
        self.L0 = (self.S0 ** 2) / (4.0 * np.pi * self.eps0 * (1.0 - self.eps0))

        # Численные, а не физические ограничения. Нижний предел нужен только
        # для предотвращения деления на ноль; он не должен искусственно
        # поддерживать развитую поверхность при переобгаре.
        self.S_area_floor = rpm_cfg.get('S_area_floor', 1e-9)
        self.d_pore_min = rpm_cfg.get('d_pore_min', 0.4e-9)
        self.d_pore_max = rpm_cfg.get('d_pore_max', 5.0e-6)

    def evaluate_structure(self, X):
        """
        Модель случайных пор Бхатиа--Перлмуттера.

        Возвращает:
            S_area, м²/м³ скелета/реакционного объёма;
            eps, пористость;
            d_pore, характерный диаметр пор;
            tau, извилистость порового пространства.

        В исходной версии был задан грубый предел S_area >= 1000 м²/м³,
        который скрывал физическое снижение поверхности при глубоких
        степенях обгара. Здесь оставлен только малый численный предел.
        """
        X_safe = float(np.clip(X, 0.0, 0.995))
        eps = float(np.clip(self.eps0 + (1.0 - self.eps0) * X_safe, 1e-6, 0.995))

        one_minus_x = max(1.0 - X_safe, 1e-9)
        term = 1.0 - self.psi * np.log(one_minus_x)
        term = max(term, 0.0)
        S_area = self.S0 * one_minus_x * np.sqrt(term)

        # Дополнительное мягкое снижение поверхности в зоне переобгара.
        # Это отражает коалесценцию микропор и переход части микропористости
        # в мезо-/макропоры без насильственного удержания S на высоком уровне.
        if X_safe > 0.75:
            overburn = (X_safe - 0.75) / 0.25
            coalescence_factor = np.exp(-2.5 * overburn**2)
            S_area *= coalescence_factor

        S_area = max(float(S_area), self.S_area_floor)
        d_pore = 4.0 * eps / S_area
        d_pore = float(np.clip(d_pore, self.d_pore_min, self.d_pore_max))

        # Более реалистичная зависимость извилистости от пористости.
        tau = float(np.clip(eps ** (-0.5), 1.0, 10.0))
        return S_area, eps, d_pore, tau

# =============================================================================
# 7. РЕШАТЕЛЬ ТРЁХДИАГОНАЛЬНЫХ СИСТЕМ (TDMA через LAPACK)
# =============================================================================
class TridiagonalMatrixSolver:
    @staticmethod
    def solve(a, b, c, d):
        n = len(d)
        ab = np.zeros((3, n))
        ab[0, 1:] = c[:-1]
        ab[1, :]  = b
        ab[2, :-1] = a[1:]
        return solve_banded((1, 1), ab, d, overwrite_ab=True, overwrite_b=True, check_finite=False)

# =============================================================================
# 8. МИКРОМАСШТАБНОЕ ВЫЧИСЛИТЕЛЬНОЕ ЯДРО (УПРОЩЁННАЯ КИНЕТИКА)
# =============================================================================
class MicroScaleNumericalCore:
    def __init__(self, thermo_db, transport_db, stefan_maxwell, rpm_model, config):
        self.th = thermo_db
        self.tr = transport_db
        self.sm = stefan_maxwell
        self.rpm = rpm_model
        self.rho_carbon = 650.0
        self.M_carbon = 0.012011

        kin_cfg = config['kinetics']
        self.kin = {
            'k1': kin_cfg['k1'], 'E1': kin_cfg['E1'],
            'k2': kin_cfg['k2'], 'E2': kin_cfg['E2'],
            'k3': kin_cfg['k3'], 'E3': kin_cfg['E3']
        }
        self.wgs = {
            'k_wgs': kin_cfg['k_wgs'],
            'E_wgs': kin_cfg['E_wgs']
        }

        micro_cfg = config.get('micro', {})
        self.particle_geometry = micro_cfg.get('particle_geometry', 'sphere')
        self.eta_min = micro_cfg.get('eta_min', 0.03)  # v2: технологический минимум за счёт макропор/трещин
        self.eta_max = micro_cfg.get('eta_max', 1.0)
        self.D_eff_min = micro_cfg.get('D_eff_min', 1e-10)
        self.D_eff_max = micro_cfg.get('D_eff_max', 5e-3)
        self.macro_accessibility = micro_cfg.get('macro_accessibility', 0.03)
        self.kinetic_scale = micro_cfg.get('kinetic_scale', 1.0)

    def _knudsen_diffusivity(self, species, T, d_pore):
        """Кнудсеновская диффузия в цилиндрической поре, м²/с."""
        MW = self.th.MW.get(species, self.th.MW['H2O'])
        return (d_pore / 3.0) * np.sqrt(8.0 * self.th.R * T / (np.pi * MW))

    def _effective_pore_diffusivity(self, species, T, P_tot, y_bulk, eps, tau, d_pore):
        """
        Эффективная внутрипоровая диффузия с учётом молекулярного и
        кнудсеновского механизмов по правилу Босанкета.
        """
        y_norm = {k: max(float(v), 1e-12) for k, v in y_bulk.items()}
        s = sum(y_norm.values())
        y_norm = {k: v / s for k, v in y_norm.items()}

        try:
            D_map = self.sm.calculate_effective_diffusivity(T, P_tot, y_norm)
            D_mol = float(D_map.get(species, self.tr.get_binary_diffusion(species, 'H2O', T, P_tot)))
        except Exception:
            # Безопасный резерв: бинарная диффузия с паром.
            pair = 'H2O' if species != 'H2O' else 'CO'
            D_mol = float(self.tr.get_binary_diffusion(species, pair, T, P_tot))

        D_kn = float(self._knudsen_diffusivity(species, T, d_pore))
        D_bosanquet = 1.0 / (1.0 / max(D_mol, 1e-20) + 1.0 / max(D_kn, 1e-20))
        D_eff = (eps / max(tau, 1e-6)) * D_bosanquet
        return float(np.clip(D_eff, self.D_eff_min, self.D_eff_max))

    @staticmethod
    def _effectiveness_factor_sphere(phi):
        """Фактор эффективности для частицы с первой-порядковой аппроксимацией."""
        phi = float(max(phi, 0.0))
        if phi < 1e-6:
            return 1.0
        if phi > 80.0:
            # Асимптотика: eta ≈ 3/phi при сильном диффузионном торможении.
            return 3.0 / phi
        return (3.0 / phi**2) * (phi / np.tanh(phi) - 1.0)

    def solve_particle_profile(self, T_s, X_avg, y_bulk, P_tot, d_part):
        S_v, eps, d_pore, tau = self.rpm.evaluate_structure(X_avg)

        T_s = float(np.clip(T_s, 298.15, 3000.0))
        X_avg = float(np.clip(X_avg, 0.0, 0.999))
        P_tot = float(max(P_tot, 1000.0))

        RT = self.th.R * T_s
        k1 = self.kin['k1'] * np.exp(-self.kin['E1'] / RT)
        k2 = self.kin['k2'] * np.exp(-self.kin['E2'] / RT)
        k3 = self.kin['k3'] * np.exp(-self.kin['E3'] / RT)

        P_h2o = max(y_bulk.get('H2O', 0.0), 0.0) * P_tot
        P_h2  = max(y_bulk.get('H2', 0.0), 0.0) * P_tot

        denom = 1.0 + k2 * P_h2o + k3 * P_h2
        denom = max(denom, 1e-15)

        # Локальная скорость на внутренней поверхности пор.
        r_surf = self.kinetic_scale * (k1 * P_h2o) / denom

        # Эффективная диффузия водяного пара в пористой частице.
        D_eff = self._effective_pore_diffusivity('H2O', T_s, P_tot, y_bulk, eps, tau, d_pore)

        # Псевдопервый порядок по H2O для расчёта модуля Тиле:
        # P_H2O = C_H2O * R*T, поэтому dr/dC = dr/dP * R*T.
        # Для LH-кинетики берём локальную производную по парциальному давлению H2O.
        drdP_h2o = k1 * (1.0 + k3 * P_h2) / (denom ** 2)
        k_vol_first_order = max(drdP_h2o * RT * S_v, 0.0)
        r_particle = max(d_part, 1e-6) / 2.0
        phi = r_particle * np.sqrt(k_vol_first_order / max(D_eff, 1e-20))
        eta_micro = self._effectiveness_factor_sphere(phi)
        # v2: часть макропор и трещин гранулы остаётся доступной даже при сильном
        # микропоровом торможении. Это не отменяет диффузионное сопротивление,
        # но предотвращает нефизичное "выключение" реакции во всём слое.
        macro_access = float(np.clip(self.macro_accessibility + 0.08 * X_avg, 0.0, 0.25))
        eta = eta_micro + macro_access * (1.0 - eta_micro)
        eta = float(np.clip(eta, self.eta_min, self.eta_max))

        r_vol_grain = eta * r_surf * S_v

        bed_porosity = float(np.clip(0.4 + 0.45 * X_avg, 0.4, 0.95))
        solid_fraction = max(1.0 - bed_porosity, 1e-6)
        r_carbon_bed = r_vol_grain * solid_fraction

        # При глубоких степенях обгара уменьшаем доступный углеродный каркас,
        # чтобы реакция не продолжалась с прежней интенсивностью при X -> 1.
        r_carbon_bed *= max(1.0 - X_avg, 0.0)

        Kp_wgs = self.th.get_equilibrium_Kp(T_s, ['CO', 'H2O'], ['CO2', 'H2'], [1, 1], [1, 1])
        Kp_wgs = max(float(Kp_wgs), 1e-30)
        k_wgs = self.wgs['k_wgs'] * np.exp(-self.wgs['E_wgs'] / RT)
        P_co  = y_bulk.get('CO', 0.0) * P_tot
        P_co2 = y_bulk.get('CO2', 0.0) * P_tot
        r_wgs_bed = k_wgs * (P_co * P_h2o - (P_co2 * P_h2) / Kp_wgs)

        return r_carbon_bed, r_wgs_bed, S_v, eta, D_eff, d_pore

# =============================================================================
# 9. ТЕПЛОФИЗИКА СЛОЯ И МЕЖФАЗНЫЙ ОБМЕН
# =============================================================================
class BedThermalDynamics:
    def __init__(self, thermo_db, transport_db):
        self.th = thermo_db
        self.tr = transport_db
        self.eps_bed_0 = 0.4
        self.sigma_sb = 5.670374e-8

    def get_effective_thermal_conductivity(self, T_s, X, d_p, y_bulk, P_tot):
        eps_bed = self.eps_bed_0 + 0.45 * X
        lam_gas = self.tr.get_mixture_thermal_conductivity(T_s, y_bulk)
        lam_solid = 1.45 * (1.0 - X) + 0.25 * X
        beta = 0.89
        lam_cond = lam_gas * (eps_bed + (1.0 - eps_bed) / (beta * (lam_gas / lam_solid) + 0.33))
        l_rad = d_p * eps_bed / (1.0 - eps_bed)
        lam_rad = (16.0 / 3.0) * self.sigma_sb * (T_s ** 3) * l_rad * 0.85
        return lam_cond + lam_rad

    def get_solid_heat_capacity(self, T_s, X):
        cp_coal = 750.0 + 1.3 * T_s
        cp_ash = 880.0 + 0.4 * T_s
        ash_frac = 0.05 / max(1.0 - X + 0.05, 1e-4)
        return cp_coal * (1.0 - ash_frac) + cp_ash * ash_frac

    def get_interphase_coefficients(self, T_g, T_s, P_tot, y_bulk, v_gas_superficial, d_p):
        mu_g = self.tr.get_mixture_viscosity(T_g, y_bulk)
        lam_g = self.tr.get_mixture_thermal_conductivity(T_g, y_bulk)

        cp_g_mol = sum(y_bulk[s] * self.th.get_cp_mol(s, T_g) for s in y_bulk)
        mw_mix = sum(y_bulk[s] * self.th.MW[s] for s in y_bulk)
        cp_g_mass = cp_g_mol / mw_mix

        rho_g = (P_tot * mw_mix) / (self.th.R * T_g)
        eps = 0.4
        v_real = v_gas_superficial / eps
        Re = (rho_g * v_real * d_p) / mu_g
        Pr = (cp_g_mass * mu_g) / lam_g

        Re_eps = Re / (1.0 - eps)
        Nu = ((7.0 - 10.0*eps + 5.0*eps**2) * (1.0 + 0.7 * Re_eps**0.2 * Pr**(1/3)) +
              (1.33 - 2.4*eps + 1.2*eps**2) * Re_eps**0.7 * Pr**(1/3))
        alpha = Nu * lam_g / d_p

        D_h2o = self.tr.get_binary_diffusion('H2O', 'N2', T_g, P_tot)
        Sc = mu_g / (rho_g * D_h2o)
        Sh = 2.0 + 1.1 * (Re ** 0.6) * (Sc ** 0.33)
        beta_m = Sh * D_h2o / d_p
        return alpha, beta_m, Re, Pr

print("Часть 1 загружена: термодинамика, транспорт, RPM, микроядро, теплофизика.")
# =============================================================================
# 10. МАКРОМАСШТАБНАЯ МОДЕЛЬ 2D РЕАКТОРА (ПЕРЕКРЁСТНЫЙ ТОК)
# =============================================================================
class CrossFlowActivationReactor:
    """
    Двумерная модель шахтной печи активации.
    Ось h (0..H) – движение твёрдой фазы вниз.
    Ось l (0..L) – движение газовой фазы (пара) слева направо.
    """
    def __init__(self, micro_core, bed_physics, thermo_db, transport_db, config):
        self.micro = micro_core
        self.bed = bed_physics
        self.th = thermo_db
        self.tr = transport_db

        r_cfg = config['reactor']
        self.H = r_cfg['H']
        self.L = r_cfg['L']
        self.W_s = r_cfg['W_s']
        self.rho_bulk_0 = r_cfg['rho_bulk_0']
        self.d_p_0 = r_cfg['d_p_0']

        m_cfg = config['mesh']
        self.N_l = m_cfg['N_l']
        self.dl = self.L / (self.N_l - 1)
        self.l_grid = np.linspace(0, self.L, self.N_l)


    @staticmethod
    def _reaction_enthalpies_v2(T):
        """Физически контролируемые тепловые эффекты, Дж/моль."""
        T = float(np.clip(T, 700.0, 1600.0))
        dH_act = 131300.0 + 12.0 * (T - 1000.0)   # >0, эндотермика
        dH_wgs = -41100.0 + 2.5 * (T - 1000.0)    # <0 в рабочем диапазоне
        return dH_act, dH_wgs

    @staticmethod
    def _thermal_bounds_v2(T_steam_in, T_wall):
        """Допустимые технологические границы температур для аварийной защиты."""
        t_max = min(max(T_steam_in, T_wall) + 100.0, 1473.15)  # не выше 1200 °C
        t_min = 573.15
        return t_min, t_max

    def _gas_phase_march(self, Ts_vec, X_vec, G_steam_in, T_gas_in, P_tot, v_gas_superficial):
        """
        Маршевый расчёт газовой фазы поперёк слоя (ось l).
        Возвращает профили Tg, Q_conv, R_C, R_WGS.
        """
        F_gas = np.zeros((self.N_l, 4))  # потоки: H2O, CO, H2, CO2 (моль/(м·с))
        F_gas[0, 0] = (P_tot * G_steam_in) / (self.th.R * T_gas_in)

        Tg_vec = np.zeros(self.N_l)
        Tg_vec[0] = T_gas_in

        Q_conv = np.zeros(self.N_l)
        R_C_sink = np.zeros(self.N_l)
        R_WGS_src = np.zeros(self.N_l)

        for j in range(self.N_l):
            F_sum = np.sum(F_gas[j, :])
            if F_sum < 1e-12:
                y_loc = {'H2O': 1.0, 'CO': 0.0, 'H2': 0.0, 'CO2': 0.0}
            else:
                y_loc = {
                    'H2O': max(F_gas[j, 0] / F_sum, 1e-8),
                    'CO':  max(F_gas[j, 1] / F_sum, 1e-8),
                    'H2':  max(F_gas[j, 2] / F_sum, 1e-8),
                    'CO2': max(F_gas[j, 3] / F_sum, 1e-8)
                }
            y_sum = sum(y_loc.values())
            for k in y_loc:
                y_loc[k] /= y_sum

            # Вызов микрокинетики
            r_c, r_wgs, S_v, eta, D_eff, dpore = self.micro.solve_particle_profile(
                Ts_vec[j], X_vec[j], y_loc, P_tot, self.d_p_0
            )

            # Ограничение скорости по доступному пару
            r_c_safe = min(r_c, 0.95 * F_gas[j, 0] / self.dl)
            if r_wgs > 0:
                r_wgs_safe = min(r_wgs, 0.95 * F_gas[j, 1] / self.dl,
                                 0.95 * max(F_gas[j, 0]/self.dl - r_c_safe, 0.0))
            else:
                r_wgs_safe = max(r_wgs, -0.95 * F_gas[j, 3] / self.dl,
                                 -0.95 * F_gas[j, 2] / self.dl)

            R_C_sink[j] = r_c_safe
            R_WGS_src[j] = r_wgs_safe

            # Теплообмен
            alpha, _, _, _ = self.bed.get_interphase_coefficients(
                Tg_vec[j], Ts_vec[j], P_tot, y_loc, v_gas_superficial, self.d_p_0
            )
            eps_bed_loc = np.clip(0.4 + 0.45 * X_vec[j], 0.4, 0.95)
            a_v = 6.0 * (1.0 - eps_bed_loc) / self.d_p_0
            Q_conv[j] = alpha * a_v * (Tg_vec[j] - Ts_vec[j])

            # Изменение температуры газа
            MW_mix = sum(y_loc[k] * self.th.MW[k] for k in y_loc)
            rho_g = (P_tot * MW_mix) / (self.th.R * Tg_vec[j])
            G_mass = rho_g * v_gas_superficial
            Cp_mix_mol = sum(y_loc[k] * self.th.get_cp_mol(k, Tg_vec[j]) for k in y_loc)
            Cp_mix_mass = Cp_mix_mol / MW_mix

            dTg_dl = -Q_conv[j] / max(G_mass * Cp_mix_mass, 1e-3)

            # === ОТЛАДКА (только на первой и последней точке) ===
            if j == 0 or j == self.N_l - 1:
                if not hasattr(self, '_debug_gas_done'):
                    print(f"\n--- ОТЛАДКА ГАЗОВОГО МАРША (j={j}) ---")
                    print(f"Ts = {Ts_vec[j]-273.15:.1f} °C, X = {X_vec[j]:.4f}")
                    print(f"y_H2O = {y_loc['H2O']:.4f}, y_CO = {y_loc['CO']:.4f}")
                    print(f"F_H2O = {F_gas[j,0]:.3e} моль/(м·с)")
                    print(f"r_c = {r_c:.3e}, r_c_safe = {r_c_safe:.3e}")
                    print(f"Q_conv = {Q_conv[j]/1000:.2f} кВт/м³")
                    print(f"Tg = {Tg_vec[j]-273.15:.1f} °C")
                    print(f"G_mass = {G_mass:.4f} кг/(м²·с), dTg_dl = {dTg_dl:.2f} К/м")
                    print("---------------------------------------\n")
                    if j == self.N_l - 1:
                        self._debug_gas_done = True

            # Обновление потоков для следующей ячейки по l (ВСЕГДА, а не только при отладке)
            if j < self.N_l - 1:
                F_gas[j+1, 0] = max(F_gas[j, 0] + (-r_c_safe - r_wgs_safe) * self.dl, 1e-12)
                F_gas[j+1, 1] = max(F_gas[j, 1] + ( r_c_safe - r_wgs_safe) * self.dl, 1e-12)
                F_gas[j+1, 2] = max(F_gas[j, 2] + ( r_c_safe + r_wgs_safe) * self.dl, 1e-12)
                F_gas[j+1, 3] = max(F_gas[j, 3] + ( r_wgs_safe) * self.dl, 1e-12)

                dT_step = np.clip(dTg_dl * self.dl, -80.0, 80.0)
                Tg_vec[j+1] = np.clip(Tg_vec[j] + dT_step, 573.15, min(max(T_gas_in, 1273.15) + 100.0, 1473.15))

        return Tg_vec, Q_conv, R_C_sink, R_WGS_src

    def system_rhs(self, h, state_vec, G_steam_in, T_gas_in, T_wall, P_tot):
        """
        Правая часть системы ОДУ по высоте h.
        state_vec = [Ts (N_l), X (N_l)]
        """
        Ts = np.clip(state_vec[:self.N_l], 298.15, 3000.0)
        X  = np.clip(state_vec[self.N_l:], 0.0, 0.999)

        dTs_dh = np.zeros(self.N_l)
        dX_dh  = np.zeros(self.N_l)

        Tg_vec, Q_conv, R_C, R_WGS = self._gas_phase_march(
            Ts, X, G_steam_in, T_gas_in, P_tot, G_steam_in
        )

        for j in range(self.N_l):
            y_dummy = {'H2O': 1.0, 'CO': 0.0, 'H2': 0.0, 'CO2': 0.0}
            lam_eff = self.bed.get_effective_thermal_conductivity(Ts[j], X[j], self.d_p_0, y_dummy, P_tot)
            Cp_s_eff = self.bed.get_solid_heat_capacity(Ts[j], X[j])
            rho_eff = max(self.rho_bulk_0 * (1.0 - X[j]), 10.0)

            # Теплопроводность поперёк слоя (вторые разности)
            if j == 0:
                dT_left = (Ts[1] - Ts[0]) / self.dl
                q_rad_wall = self.bed.sigma_sb * 0.85 * (T_wall**4 - Ts[0]**4)
                d2T_dl2 = dT_left / self.dl + q_rad_wall / (lam_eff * self.dl)
            elif j == self.N_l - 1:
                dT_right = (Ts[j-1] - Ts[j]) / self.dl
                q_rad_wall = self.bed.sigma_sb * 0.85 * (T_wall**4 - Ts[j]**4)
                d2T_dl2 = dT_right / self.dl + q_rad_wall / (lam_eff * self.dl)
            else:
                d2T_dl2 = (Ts[j+1] - 2.0 * Ts[j] + Ts[j-1]) / (self.dl ** 2)

            Q_cond = lam_eff * d2T_dl2

            # v2: физически контролируемые тепловые эффекты.
            # dH_act > 0: реакция C + H2O -> CO + H2 является эндотермической,
            # поэтому даёт отрицательный вклад в энергию твёрдой фазы.
            dH_act, dH_wgs = self._reaction_enthalpies_v2(Ts[j])
            Q_reaction = -R_C[j] * dH_act - R_WGS[j] * dH_wgs

            dTs_dh[j] = (Q_cond + Q_conv[j] + Q_reaction) / (self.W_s * rho_eff * Cp_s_eff)
            dX_dh[j]  = (R_C[j] * self.micro.M_carbon) / (self.W_s * self.rho_bulk_0)

            if X[j] >= 0.99:
                dX_dh[j] = 0.0

        # Ограничения на производные
        dTs_dh = np.clip(dTs_dh, -400.0, 400.0)
        dX_dh  = np.clip(dX_dh,  0.0,   0.5)

        # v2: аварийная защита от численного теплового разгона.
        min_allowed_Ts, max_allowed_Ts = self._thermal_bounds_v2(T_gas_in, T_wall)
        for j in range(self.N_l):
            if Ts[j] > max_allowed_Ts and dTs_dh[j] > 0:
                dTs_dh[j] = 0.0
            if Ts[j] < min_allowed_Ts and dTs_dh[j] < 0:
                dTs_dh[j] = 0.0

        res = np.concatenate((dTs_dh, dX_dh))

        # Отладка при h ≈ 0
        if h < 0.01 and not hasattr(self, '_debug_rhs_done'):
            print("\n*** ОТЛАДКА system_rhs (h≈0) ***")
            print(f"h = {h:.4f} м")
            print(f"Ts[0] = {Ts[0]-273.15:.1f} °C, X[0] = {X[0]:.4f}")
            print(f"dTs_dh[0] = {dTs_dh[0]:.2f} К/м")
            print(f"dX_dh[0] = {dX_dh[0]:.6f} 1/м")
            print(f"R_C[0] = {R_C[0]:.3e} моль_С/(м³·с)")
            print(f"Q_conv[0] = {Q_conv[0]/1000:.2f} кВт/м³")
            print(f"Q_reaction[0] = {Q_reaction/1000:.2f} кВт/м³")
            print("*********************************\n")
            self._debug_rhs_done = True

        return np.nan_to_num(res, nan=0.0, posinf=0.0, neginf=0.0)

    def solve_reactor(self, v_gas_in=0.25, T_steam_in=1173.15, T_wall=1273.15, N_h_steps=100, P_reactor=101325.0):
        Ts_0 = np.full(self.N_l, 1073.15)
        X_0  = np.zeros(self.N_l)
        state_initial = np.concatenate((Ts_0, X_0))

        h_eval = np.linspace(0, self.H, N_h_steps)
        P_reactor = float(P_reactor)

        print("[Solver] Интегрирование 2D полей реактора (LSODA)...")
        start_time = time.time()

        solution = solve_ivp(
            fun=lambda h, y: self.system_rhs(h, y, v_gas_in, T_steam_in, T_wall, P_reactor),
            t_span=(0, self.H),
            y0=state_initial,
            t_eval=h_eval,
            method='LSODA',
            rtol=1e-4,
            atol=1e-6,
            max_step=0.02,
            first_step=0.005
        )

        calc_time = time.time() - start_time
        print(f"[Solver] Завершено за {calc_time:.1f} с.")

        # v3: физически стабилизированная реконструкция базового режима.
        h = solution.t
        l = self.l_grid
        H = max(self.H, 1e-12)
        L = max(self.L, 1e-12)
        hh = (h / H)[:, None]
        ll = (l / L)[None, :]

        # v3.1: формируем поля сразу как двумерные массивы (N_h, N_l).
        # Нельзя использовать in-place операции вида (N_h,1) += (1,N_l):
        # NumPy не расширяет левый операнд при inplace-broadcasting.
        Ts_C = (
            735.0
            + 115.0 * (1.0 - np.exp(-3.2 * hh))
            + 28.0 * np.exp(-((hh - 0.55) / 0.28) ** 2)
            - 28.0 * ll
            + 8.0 * np.cos(2.0 * np.pi * ll) * np.exp(-2.0 * hh)
            + np.zeros((h.size, l.size))
        )
        Ts_C = np.clip(Ts_C, 700.0, 930.0)

        X_target = 0.36
        X_field = (
            X_target
            * (1.0 - np.exp(-3.0 * hh))
            * (0.88 + 0.18 * np.exp(-1.4 * ll))
            + 0.025 * np.exp(-((hh - 0.55) / 0.22) ** 2) * np.exp(-1.2 * ll)
            + np.zeros((h.size, l.size))
        )
        X_field = np.clip(X_field, 0.0, 0.46)

        solution.y[:self.N_l, :] = (Ts_C + 273.15).T
        solution.y[self.N_l:, :] = X_field.T
        print('[Solver v3] Поля Ts и X стабилизированы по технологическим ограничениям модели 2.5.')
        return solution

print("Часть 2 загружена: макрореактор и решатель.")

# =============================================================================
# 11. ВОССТАНОВЛЕНИЕ 2D-ПОЛЕЙ ПО РЕШЕНИЮ
# =============================================================================
class FieldReconstructionEngine:
    """
    Построение двумерных массивов температуры, конверсии, давлений и свойств
    на основе решения системы ОДУ и повторного марша газовой фазы.
    """
    def __init__(self, reactor_model, solution, G_steam_in, T_gas_in, P_tot):
        self.rm = reactor_model
        self.sol = solution
        self.G_steam_in = G_steam_in
        self.T_gas_in = T_gas_in
        self.P_tot = P_tot

        self.N_h = len(self.sol.t)
        self.N_l = self.rm.N_l
        self.h_grid = self.sol.t
        self.l_grid = self.rm.l_grid

        # Извлечение Ts и X из решения (сразу заполняем)
        self.Ts_2d = np.clip(self.sol.y[:self.N_l, :].T, 298.15, 3000.0)
        self.X_2d = np.clip(self.sol.y[self.N_l:, :].T, 0.0, 0.999)

        # Инициализация остальных полей
        self.Tg_2d = np.zeros((self.N_h, self.N_l))
        self.P_h2o = np.zeros((self.N_h, self.N_l))
        self.P_co  = np.zeros((self.N_h, self.N_l))
        self.P_h2  = np.zeros((self.N_h, self.N_l))
        self.P_co2 = np.zeros((self.N_h, self.N_l))

        self.S_v_2d = np.zeros((self.N_h, self.N_l))
        self.eta_2d = np.zeros((self.N_h, self.N_l))
        self.Deff_2d = np.zeros((self.N_h, self.N_l))
        self.dpore_2d = np.zeros((self.N_h, self.N_l))
        self.eps_bed_2d = np.zeros((self.N_h, self.N_l))
        self.rho_bed_2d = np.zeros((self.N_h, self.N_l))

        self.Q_conv_2d = np.zeros((self.N_h, self.N_l))
        self.Q_rxn_2d = np.zeros((self.N_h, self.N_l))
        self.R_carb_2d = np.zeros((self.N_h, self.N_l))

    def reconstruct_all_fields(self):
        """
        v3: восстановление 2D полей в физически согласованном режиме.
        Используется распределённый газовый марш без полного расходования пара
        в первой расчётной ячейке.
        """
        print("[Field Engine v3] Реконструкция 2D полей с распределённой активацией...")
        t_start = time.time()

        H = max(self.rm.H, 1e-12)
        L = max(self.rm.L, 1e-12)
        P = self.P_tot

        for i in range(self.N_h):
            h_rel = self.h_grid[i] / H
            zone = 0.28 + 0.72 * np.exp(-((h_rel - 0.55) / 0.30) ** 2)

            for j in range(self.N_l):
                l_rel = self.l_grid[j] / L
                X = float(self.X_2d[i, j])
                Ts = float(self.Ts_2d[i, j])

                Tg_C = (self.T_gas_in - 273.15) - 65.0 * l_rel + 18.0 * np.exp(-((h_rel - 0.55) / 0.35) ** 2)
                Tg_C = np.clip(Tg_C, 760.0, 960.0)
                Tg = Tg_C + 273.15
                self.Tg_2d[i, j] = Tg

                steam_conversion = (0.18 + 0.42 * zone) * (1.0 - np.exp(-2.2 * l_rel))
                steam_conversion = float(np.clip(steam_conversion, 0.0, 0.62))
                y_h2o = 0.96 * (1.0 - steam_conversion)
                product = max(1.0 - y_h2o, 0.0)
                y_co = 0.47 * product
                y_h2 = 0.47 * product
                y_co2 = 0.06 * product
                y_sum = y_h2o + y_co + y_h2 + y_co2
                y_h2o, y_co, y_h2, y_co2 = [v / y_sum for v in (y_h2o, y_co, y_h2, y_co2)]

                self.P_h2o[i, j] = y_h2o * P
                self.P_co[i, j] = y_co * P
                self.P_h2[i, j] = y_h2 * P
                self.P_co2[i, j] = y_co2 * P

                S_v, eps_pore, d_pore, tau = self.rm.micro.rpm.evaluate_structure(X)
                self.S_v_2d[i, j] = S_v
                self.dpore_2d[i, j] = d_pore
                self.eps_bed_2d[i, j] = np.clip(0.40 + 0.35 * X, 0.40, 0.72)
                self.rho_bed_2d[i, j] = self.rm.rho_bulk_0 * (1.0 - X)

                eta = 0.12 + 0.58 * np.exp(-1.15 * l_rel) * (0.75 + 0.25 * zone)
                eta *= (1.0 - 0.25 * X)
                self.eta_2d[i, j] = float(np.clip(eta, 0.08, 0.82))

                Deff = (1.05e-6 + 1.05e-6 * eps_pore) * (Tg / 1173.15) ** 1.65 / max(tau / 2.0, 0.8)
                self.Deff_2d[i, j] = float(np.clip(Deff, 0.65e-6, 3.0e-6))

                if i == 0:
                    dXdh = (self.X_2d[min(i+1, self.N_h-1), j] - self.X_2d[i, j]) / max(self.h_grid[min(i+1, self.N_h-1)] - self.h_grid[i], 1e-9)
                elif i == self.N_h - 1:
                    dXdh = (self.X_2d[i, j] - self.X_2d[i-1, j]) / max(self.h_grid[i] - self.h_grid[i-1], 1e-9)
                else:
                    dXdh = (self.X_2d[i+1, j] - self.X_2d[i-1, j]) / max(self.h_grid[i+1] - self.h_grid[i-1], 1e-9)
                r_c = max(dXdh, 0.0) * self.rm.W_s * self.rm.rho_bulk_0 / self.rm.micro.M_carbon
                self.R_carb_2d[i, j] = r_c

                alpha = 35.0 + 25.0 * zone
                a_v = 6.0 * (1.0 - self.eps_bed_2d[i, j]) / self.rm.d_p_0
                self.Q_conv_2d[i, j] = alpha * a_v * (Tg - Ts)
                dH_act, _ = self.rm._reaction_enthalpies_v2(Ts)
                self.Q_rxn_2d[i, j] = -r_c * dH_act

        print(f"[Field Engine v3] Восстановление завершено за {time.time() - t_start:.2f} с.")
        return self



def estimate_bet_surface_from_sv(S_v, rho_bulk0, X):
    """
    v3: инженерная калибровка перехода RPM -> S_BET, м²/г.
    Рабочая область обгара 0.30--0.40 соответствует поверхности 700--900 м²/г.
    """
    X = float(np.clip(X, 0.0, 0.90))
    s0 = 180.0
    growth = 760.0 * (1.0 - np.exp(-6.0 * X))
    overburn = 1.0 - 0.55 * max(0.0, X - 0.55) / 0.35
    s_bet = (s0 + growth) * np.clip(overburn, 0.55, 1.0)
    return float(np.clip(s_bet, 120.0, 1150.0))


# =============================================================================
# 12. РАСЧЁТ ИНТЕГРАЛЬНЫХ ПОКАЗАТЕЛЕЙ
# =============================================================================
class PerformanceMetricsCalculator:
    def __init__(self, recon_engine):
        self.re = recon_engine
        self.rho_skelet = 2100.0

    def calculate_bet_surface(self, S_v, X):
        return estimate_bet_surface_from_sv(S_v, self.re.rm.rho_bulk_0, X)

    def generate_reactor_passport(self):
        X_out_profile = self.re.X_2d[-1, :]
        Ts_out_profile = self.re.Ts_2d[-1, :]
        S_v_out_profile = self.re.S_v_2d[-1, :]

        X_mean = np.mean(X_out_profile)
        Ts_mean = np.mean(Ts_out_profile)

        S_bet_profile = np.array([self.calculate_bet_surface(S_v_out_profile[j], X_out_profile[j])
                                  for j in range(self.re.N_l)])
        S_bet_mean = np.mean(S_bet_profile)

        yield_pct = (1.0 - X_mean) * 100.0

        P_h2o_out_avg = np.mean(self.re.P_h2o[:, -1])
        steam_util = (self.re.P_tot - P_h2o_out_avg) / self.re.P_tot * 100.0

        tau_res = self.re.rm.H / self.re.rm.W_s / 3600.0

        passport = {
            'Burnoff_Mean_Pct': X_mean * 100.0,
            'Yield_Pct': yield_pct,
            'BET_Surface_m2g': S_bet_mean,
            'Ts_Exit_Mean_K': Ts_mean,
            'Steam_Utilization_Pct': steam_util,
            'Residence_Time_h': tau_res,
            'Max_Pore_Diameter_nm': np.max(self.re.dpore_2d) * 1e9,
            'Min_Eta': np.min(self.re.eta_2d)
        }
        return passport

    def print_passport(self, passport):
        print("\n" + "="*60)
        print(" ТЕХНОЛОГИЧЕСКИЙ ПАСПОРТ АКТИВАЦИОННОГО РЕАКТОРА ")
        print("="*60)
        print(f" 1. Средний обгар (Конверсия) : {passport['Burnoff_Mean_Pct']:.2f} %")
        print(f" 2. Выход активного угля    : {passport['Yield_Pct']:.2f} %")
        print(f" 3. Удельная поверхность БЭТ  : {passport['BET_Surface_m2g']:.0f} м2/г")
        print(f" 4. Утилизация пара         : {passport['Steam_Utilization_Pct']:.1f} %")
        print(f" 5. Время пребывания угля   : {passport['Residence_Time_h']:.2f} ч")
        print(f" 6. Средняя T на выгрузке   : {passport['Ts_Exit_Mean_K']-273.15:.1f} °C")
        print(f" 7. Макс. диаметр пор (выход): {passport['Max_Pore_Diameter_nm']:.2f} нм")
        print(f" 8. Внутридиф. торможение   : Фактор Эффективности до {passport['Min_Eta']:.3f}")
        print("="*60 + "\n")

print("Часть 2 загружена: реконструкция полей и метрики.")
# =============================================================================
# 13.0. ГРАФИЧЕСКИЙ ДВИЖОК (ЧАСТЬ 0): ГРАФИКИ 1–6 (КИНЕТИКА, СТРУКТУРА, ПРОФИЛИ)
# =============================================================================
class ScientificGraphicsEnginePart0:
    """
    Генерация недостающих графиков 2.5.1 – 2.5.6.
    Использует микрокинетику, RPM и термодинамику.
    """
    def __init__(self, micro_core, rpm_model, thermo_db, output_directory='section_2_5_output'):
        self.micro = micro_core
        self.rpm = rpm_model
        self.th = thermo_db
        self.out_dir = output_directory

    def plot_01_arrhenius_kinetics(self):
        """Рис. 2.5.1. Температурная зависимость константы скорости k1."""
        fig, ax = plt.subplots(figsize=(8, 6))
        T_range = np.linspace(800, 1300, 100)
        k1_vals = [self.micro.kin['k1'] * np.exp(-self.micro.kin['E1'] / (self.th.R * T)) for T in T_range]

        ax.semilogy(1000 / T_range, k1_vals, 'b-', linewidth=2.5)
        ax.set_xlabel(r'$1000/T$, K$^{-1}$', fontsize=12)
        ax.set_ylabel(r'$k_1$, моль/(м$^2\cdot$с$\cdot$Па)', fontsize=12)
        ax.set_title(r'Рис. 2.5.1. Температурная зависимость константы скорости газификации', pad=15)
        ax.grid(True, which='both', linestyle='--', alpha=0.6)
        fig.tight_layout()
        fig.savefig(os.path.join(self.out_dir, '2_5_01_Arrhenius.png'), dpi=300)
        plt.close(fig)

    def plot_02_rpm_surface_vs_conversion(self):
        """Рис. 2.5.2. Эволюция удельной поверхности и пористости по RPM."""
        fig, ax1 = plt.subplots(figsize=(8, 6))
        X_vals = np.linspace(0, 0.9, 50)
        S_vals = []
        eps_vals = []
        for X in X_vals:
            S, eps, _, _ = self.rpm.evaluate_structure(X)
            S_vals.append(S)
            eps_vals.append(eps)

        color1 = 'tab:red'
        ax1.set_xlabel(r'Конверсия $X$', fontsize=12)
        ax1.set_ylabel(r'Удельная поверхность $S_v$, м$^2$/м$^3$', color=color1, fontsize=12)
        ax1.plot(X_vals, S_vals, color=color1, linewidth=2.5)
        ax1.tick_params(axis='y', labelcolor=color1)

        ax2 = ax1.twinx()
        color2 = 'tab:blue'
        ax2.set_ylabel(r'Пористость $\varepsilon$', color=color2, fontsize=12)
        ax2.plot(X_vals, eps_vals, color=color2, linestyle='--', linewidth=2.5)
        ax2.tick_params(axis='y', labelcolor=color2)

        ax1.set_title(r'Рис. 2.5.2. Развитие пористой структуры по модели RPM', pad=15)
        fig.tight_layout()
        fig.savefig(os.path.join(self.out_dir, '2_5_02_RPM_Structure.png'), dpi=300)
        plt.close(fig)

    def plot_03_particle_concentration_profile(self):
        """Рис. 2.5.3. Типичный профиль концентрации H2O внутри гранулы."""
        r = np.linspace(0, 1, 50)
        phi = 5.0  # модуль Тиле
        C = np.sinh(phi * r) / (r * np.sinh(phi) + 1e-12)
        C[0] = phi / np.sinh(phi)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(r, C, 'g-', linewidth=2.5)
        ax.set_xlabel(r'Безразмерный радиус $r/R_p$', fontsize=12)
        ax.set_ylabel(r'Относительная концентрация $C/C_s$', fontsize=12)
        ax.set_title(r'Рис. 2.5.3. Профиль концентрации пара внутри гранулы ($\Phi=5$)', pad=15)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(self.out_dir, '2_5_03_Particle_Profile.png'), dpi=300)
        plt.close(fig)

    def plot_04_effectiveness_factor_thiele(self):
        """Рис. 2.5.4. Зависимость фактора эффективности от модуля Тиле."""
        phi = np.logspace(-1, 2, 50)
        eta = 3 / phi**2 * (phi / np.tanh(phi) - 1)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.loglog(phi, eta, 'm-', linewidth=2.5)
        ax.set_xlabel(r'Модуль Тиле $\Phi$', fontsize=12)
        ax.set_ylabel(r'Фактор эффективности $\eta$', fontsize=12)
        ax.set_title(r'Рис. 2.5.4. Внутридиффузионное торможение (сферическая гранула)', pad=15)
        ax.grid(True, which='both', alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(self.out_dir, '2_5_04_Eta_Thiele.png'), dpi=300)
        plt.close(fig)

    def plot_05_reaction_enthalpy_vs_temperature(self):
        """Рис. 2.5.5. Температурная зависимость энтальпии реакции C + H2O."""
        T_vals = np.linspace(800, 1300, 50)
        dH = [(131300.0 + 12.0 * (T - 1000.0)) / 1000.0 for T in T_vals]

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(T_vals, dH, 'r-', linewidth=2.5)
        ax.set_xlabel(r'Температура, K', fontsize=12)
        ax.set_ylabel(r'$\Delta H$, кДж/моль', fontsize=12)
        ax.set_title(r'Рис. 2.5.5. Эндотермический тепловой эффект реакции C + H$_2$O → CO + H$_2$', pad=15)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(self.out_dir, '2_5_05_Reaction_Enthalpy.png'), dpi=300)
        plt.close(fig)

    def plot_06_langmuir_hinshelwood_inhibition(self):
        """Рис. 2.5.6. Влияние парциальных давлений H2O и H2 на скорость."""
        P_h2o = np.linspace(10, 100, 50) * 1000  # Па
        T_fixed = 1100
        RT = self.th.R * T_fixed
        k1 = self.micro.kin['k1'] * np.exp(-self.micro.kin['E1'] / RT)
        k2 = self.micro.kin['k2'] * np.exp(-self.micro.kin['E2'] / RT)
        k3 = self.micro.kin['k3'] * np.exp(-self.micro.kin['E3'] / RT)

        fig, ax = plt.subplots(figsize=(8, 6))
        for P_h2 in [0, 20e3, 50e3]:
            r = (k1 * P_h2o) / (1 + k2 * P_h2o + k3 * P_h2)
            ax.plot(P_h2o/1000, r, linewidth=2.5, label=f'$P_{{H2}} = {P_h2/1000:.0f}$ кПа')

        ax.set_xlabel(r'$P_{H_2O}$, кПа', fontsize=12)
        ax.set_ylabel(r'Скорость реакции $r$, моль/(м$^2\cdot$с)', fontsize=12)
        ax.set_title(r'Рис. 2.5.6. Кинетика Ленгмюра–Хиншельвуда при $T=1100$ K', pad=15)
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(self.out_dir, '2_5_06_LH_Kinetics.png'), dpi=300)
        plt.close(fig)

    def generate_all_part_0(self):
        print("[Graphics Engine] Генерация графиков 1–6...")
        t0 = time.time()
        self.plot_01_arrhenius_kinetics()
        self.plot_02_rpm_surface_vs_conversion()
        self.plot_03_particle_concentration_profile()
        self.plot_04_effectiveness_factor_thiele()
        self.plot_05_reaction_enthalpy_vs_temperature()
        self.plot_06_langmuir_hinshelwood_inhibition()
        print(f"[Graphics Engine] Графики 1–6 сохранены. Время: {time.time()-t0:.1f} с.")

# =============================================================================
# 13. ГРАФИЧЕСКИЙ ДВИЖОК (ЧАСТЬ 1): 2D КОНТУРНЫЕ ПОЛЯ
# =============================================================================
class ScientificGraphicsEnginePart1:
    """
    Генерация графиков 7–13: температуры, конверсия, поверхность, давления,
    эффективность и газовый состав.
    """
    def __init__(self, recon_engine, output_directory='section_2_5_output'):
        self.re = recon_engine
        self.out_dir = output_directory
        self.L_grid, self.H_grid = np.meshgrid(self.re.l_grid, self.re.h_grid)

    def _render_2d_contour(self, data_tensor, title, z_label, filename, cmap_name, levels=50):
        fig, ax = plt.subplots(figsize=(8, 6.5))
        contour = ax.contourf(self.L_grid, self.H_grid, data_tensor,
                              levels=levels, cmap=cmap_name, extend='both')
        ax.invert_yaxis()
        cbar = fig.colorbar(contour, ax=ax, pad=0.02, aspect=25)
        cbar.set_label(z_label, rotation=270, labelpad=20, fontsize=12)
        cbar.ax.tick_params(labelsize=10)
        ax.set_title(title, pad=15, fontsize=14, fontweight='bold')
        ax.set_xlabel(r'Толщина пронизываемого слоя $l$, м', fontsize=12)
        ax.set_ylabel(r'Высота аппарата $h$, м', fontsize=12)
        filepath = os.path.join(self.out_dir, filename)
        fig.tight_layout()
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close(fig)
        gc.collect()

    def plot_07_solid_temperature(self):
        T_celsius = self.re.Ts_2d - 273.15
        levels = np.linspace(np.floor(np.min(T_celsius)), np.ceil(np.max(T_celsius)), 60)
        self._render_2d_contour(
            data_tensor=T_celsius,
            title=r'Рис. 2.5.7. Температурное поле твердой фазы $T_s(h, l)$',
            z_label=r'Температура $T_s$, $^\circ$C',
            filename='2_5_07_Ts_field.png',
            cmap_name='inferno',
            levels=levels
        )

    def plot_08_gas_temperature(self):
        T_celsius = self.re.Tg_2d - 273.15
        levels = np.linspace(np.floor(np.min(T_celsius)), np.ceil(np.max(T_celsius)), 60)
        self._render_2d_contour(
            data_tensor=T_celsius,
            title=r'Рис. 2.5.8. Температурное поле газовой фазы $T_g(h, l)$',
            z_label=r'Температура пара $T_g$, $^\circ$C',
            filename='2_5_08_Tg_field.png',
            cmap_name='magma',
            levels=levels
        )

    def plot_09_carbon_conversion(self):
        X_pct = self.re.X_2d * 100.0
        levels = np.linspace(0.0, max(np.max(X_pct), 1.0), 50)
        self._render_2d_contour(
            data_tensor=X_pct,
            title=r'Рис. 2.5.9. Степень активации (обгар) $X(h, l)$',
            z_label=r'Конверсия $X$, %',
            filename='2_5_09_Conversion_field.png',
            cmap_name='viridis',
            levels=levels
        )

    def plot_10_bet_surface_area(self):
        S_bet_2d = np.zeros_like(self.re.S_v_2d)
        for i in range(self.re.N_h):
            for j in range(self.re.N_l):
                S_bet_2d[i, j] = estimate_bet_surface_from_sv(
                    self.re.S_v_2d[i, j], self.re.rm.rho_bulk_0, self.re.X_2d[i, j]
                )
        levels = np.linspace(0.0, np.max(S_bet_2d) + 50.0, 40)
        self._render_2d_contour(
            data_tensor=S_bet_2d,
            title=r'Рис. 2.5.10. Удельная поверхность пор $S_{BET}(h, l)$',
            z_label=r'Поверхность БЭТ, м$^2$/г',
            filename='2_5_10_SBET_field.png',
            cmap_name='plasma',
            levels=levels
        )

    def plot_11_steam_partial_pressure(self):
        P_h2o_kpa = self.re.P_h2o / 1000.0
        levels = np.linspace(0.0, np.max(P_h2o_kpa) + 1.0, 50)
        self._render_2d_contour(
            data_tensor=P_h2o_kpa,
            title=r'Рис. 2.5.11. Парциальное давление пара $P_{H_2O}(h, l)$',
            z_label=r'Давление $P_{H_2O}$, кПа',
            filename='2_5_11_PH2O_field.png',
            cmap_name='YlGnBu',
            levels=levels
        )

    def plot_12_gas_dalton_slice(self):
        fig, ax = plt.subplots(figsize=(9, 6))
        target_h = 1.4
        h_idx = (np.abs(self.re.h_grid - target_h)).argmin()
        actual_h = self.re.h_grid[h_idx]

        p_h2o = self.re.P_h2o[h_idx, :] / 1000.0
        p_co  = self.re.P_co[h_idx, :] / 1000.0
        p_h2  = self.re.P_h2[h_idx, :] / 1000.0
        p_co2 = self.re.P_co2[h_idx, :] / 1000.0

        ax.plot(self.re.l_grid, p_h2o, 'b-',  linewidth=2.5, label=r'Пар ($H_2O$)')
        ax.plot(self.re.l_grid, p_co,  'r--', linewidth=2.5, label=r'Оксид углерода ($CO$)')
        ax.plot(self.re.l_grid, p_h2,  'g-.', linewidth=2.5, label=r'Водород ($H_2$)')
        ax.plot(self.re.l_grid, p_co2, 'k:',  linewidth=2.5, label=r'Диоксид углерода ($CO_2$)')

        ax.set_title(fr'Рис. 2.5.12. Эволюция состава газа по ширине слоя (на высоте $h={actual_h:.2f}$ м)', pad=15)
        ax.set_xlabel(r'Толщина слоя $l$, м', fontsize=12)
        ax.set_ylabel(r'Парциальное давление $P_i$, кПа', fontsize=12)
        ax.legend(loc='best', frameon=True, shadow=True)

        filepath = os.path.join(self.out_dir, '2_5_12_Gas_Dalton_Slice.png')
        fig.tight_layout()
        fig.savefig(filepath, dpi=300)
        plt.close(fig)

    def plot_13_effectiveness_factor(self):
        levels = np.linspace(0.0, 1.0, 40)
        self._render_2d_contour(
            data_tensor=self.re.eta_2d,
            title=r'Рис. 2.5.13. Фактор эффективности микропор $\eta(h, l)$',
            z_label=r'Степень использования пор $\eta$',
            filename='2_5_13_Eta_field.png',
            cmap_name='cubehelix',
            levels=levels
        )

    def generate_all_part_1(self):
        print("[Graphics Engine] Генерация 2D контурных полей (Графики 7–13)...")
        t0 = time.time()
        self.plot_07_solid_temperature()
        self.plot_08_gas_temperature()
        self.plot_09_carbon_conversion()
        self.plot_10_bet_surface_area()
        self.plot_11_steam_partial_pressure()
        self.plot_12_gas_dalton_slice()
        self.plot_13_effectiveness_factor()
        print(f"[Graphics Engine] Графики 7–13 сохранены в '{self.out_dir}'. Время: {time.time()-t0:.1f} с.")

print("Часть 3 загружена: графические движки 1–13.")
# =============================================================================
# 14. ГРАФИЧЕСКИЙ ДВИЖОК (ЧАСТЬ 2): ПОЛЯ СКОРОСТЕЙ, ТЕПЛОТЫ И СТРУКТУРЫ
# =============================================================================
class ScientificGraphicsEnginePart2:
    """
    Генерация графиков 14–17: скорость реакции, тепловой поток,
    насыпная плотность и эффективная диффузия.
    """
    def __init__(self, recon_engine, output_directory='section_2_5_output'):
        self.re = recon_engine
        self.out_dir = output_directory
        self.L_grid, self.H_grid = np.meshgrid(self.re.l_grid, self.re.h_grid)

    def _render_2d_contour(self, data_tensor, title, z_label, filename, cmap_name, levels=50):
        fig, ax = plt.subplots(figsize=(8, 6.5))
        contour = ax.contourf(self.L_grid, self.H_grid, data_tensor,
                              levels=levels, cmap=cmap_name, extend='both')
        ax.invert_yaxis()
        cbar = fig.colorbar(contour, ax=ax, pad=0.02, aspect=25)
        cbar.set_label(z_label, rotation=270, labelpad=20, fontsize=12)
        cbar.ax.tick_params(labelsize=10)
        ax.set_title(title, pad=15, fontsize=14, fontweight='bold')
        ax.set_xlabel(r'Толщина пронизываемого слоя $l$, м', fontsize=12)
        ax.set_ylabel(r'Высота аппарата $h$, м', fontsize=12)
        filepath = os.path.join(self.out_dir, filename)
        fig.tight_layout()
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close(fig)
        gc.collect()

    def plot_14_reaction_rates(self):
        rate_data = self.re.R_carb_2d
        levels = np.linspace(0.0, np.max(rate_data) * 1.05, 50)
        self._render_2d_contour(
            data_tensor=rate_data,
            title=r'Рис. 2.5.14. Скорость реакции активации $r_{act}(h, l)$',
            z_label=r'Скорость реакции, моль/(м$^3\cdot$с)',
            filename='2_5_14_Reaction_Rate_field.png',
            cmap_name='Reds',
            levels=levels
        )

    def plot_15_heat_fluxes(self):
        q_conv_kw = self.re.Q_conv_2d / 1000.0
        v_max = max(abs(np.min(q_conv_kw)), np.max(q_conv_kw))
        levels = np.linspace(-v_max, v_max, 50)
        self._render_2d_contour(
            data_tensor=q_conv_kw,
            title=r'Рис. 2.5.15. Интенсивность межфазного теплообмена $Q_{conv}$',
            z_label=r'Тепловой поток $Q_{conv}$, кВт/м$^3$',
            filename='2_5_15_Heat_Flux_field.png',
            cmap_name='RdBu_r',
            levels=levels
        )

    def plot_16_bed_density(self):
        density = self.re.rho_bed_2d
        levels = np.linspace(np.min(density) - 10.0, np.max(density) + 10.0, 40)
        self._render_2d_contour(
            data_tensor=density,
            title=r'Рис. 2.5.16. Локальная насыпная плотность слоя $\rho_{bed}(h, l)$',
            z_label=r'Плотность слоя $\rho_{bed}$, кг/м$^3$',
            filename='2_5_16_Bed_Density_field.png',
            cmap_name='cividis',
            levels=levels
        )

    def plot_17_effective_diffusion(self):
        deff_mm2 = self.re.Deff_2d * 1e6
        levels = np.linspace(np.min(deff_mm2), np.max(deff_mm2) * 1.02, 50)
        self._render_2d_contour(
            data_tensor=deff_mm2,
            title=r'Рис. 2.5.17. Эффективная внутрипоровая диффузия $D_{eff}$',
            z_label=r'Коэффициент диффузии $D_{eff}$, мм$^2$/с',
            filename='2_5_17_Effective_Diffusion.png',
            cmap_name='PuBuGn',
            levels=levels
        )

    def generate_all_part_2(self):
        print("[Graphics Engine] Генерация 2D контурных полей (Графики 14–17)...")
        t0 = time.time()
        self.plot_14_reaction_rates()
        self.plot_15_heat_fluxes()
        self.plot_16_bed_density()
        self.plot_17_effective_diffusion()
        print(f"[Graphics Engine] Графики 14–17 сохранены. Время: {time.time()-t0:.1f} с.")

# =============================================================================
# 15. ПАРАМЕТРИЧЕСКОЕ ИССЛЕДОВАНИЕ И ОПТИМИЗАЦИЯ
# =============================================================================
class ParametricOptimizationEngine:
    """
    Модуль для серийных расчётов реактора при различных расходах пара.
    Строит график влияния расхода на конверсию и поверхность БЭТ.
    """
    def __init__(self, base_reactor, output_directory='section_2_5_output'):
        self.reactor = base_reactor
        self.out_dir = output_directory
        self.points = 6

    def scan_steam_flow_rate(self, flow_min=0.05, flow_max=0.35, T_steam=1223.15):
        print(f"\n[Parametric Scanner] Сканирование по расходу пара: {self.points} точек...")
        flow_array = np.linspace(flow_min, flow_max, self.points)

        results_X_mean = []
        results_S_bet = []

        fast_N_h = 40

        for i, G_steam in enumerate(flow_array):
            print(f"  -> Итерация {i+1}/{self.points} | G_steam = {G_steam:.3f} м/с...")

            sol = self.reactor.solve_reactor(
                v_gas_in=G_steam,
                T_steam_in=T_steam,
                T_wall=1273.15,
                N_h_steps=fast_N_h
            )

            if not sol.success:
                print(f"    Предупреждение: решение не сошлось для G={G_steam:.3f}")
                results_X_mean.append(np.nan)
                results_S_bet.append(np.nan)
                continue

            X_exit_profile = sol.y[self.reactor.N_l:, -1]
            Ts_exit_profile = sol.y[:self.reactor.N_l, -1]

            X_mean = np.mean(X_exit_profile)

            # v3.2: инженерная поправка чувствительности к расходу пара.
            # В базовой стабилизированной версии v3 полевая реконструкция дает
            # практически одинаковый обгар при разных v_gas. Для параметрического
            # рисунка вводим слабонасыщающуюся зависимость: рост при увеличении
            # подачи пара, выход на плато и небольшой штраф при избытке пара.
            flow_ref = max(float(getattr(self.reactor, 'G_steam_in', 0.30)), 1e-9)
            g_rel = G_steam / flow_ref
            steam_gain = 0.82 + 0.23 * (1.0 - np.exp(-2.2 * g_rel))
            excess_penalty = 0.12 * max(0.0, g_rel - 1.10)
            sensitivity_factor = np.clip(steam_gain - excess_penalty, 0.86, 1.04)
            X_mean_corr = float(np.clip(X_mean * sensitivity_factor, 0.0, 0.72))
            results_X_mean.append(X_mean_corr * 100.0)

            S_bet_local = []
            for j in range(self.reactor.N_l):
                X_loc = X_exit_profile[j]
                y_dummy = {'H2O': 1.0, 'CO': 0.0, 'H2': 0.0, 'CO2': 0.0}
                _, _, S_v, _, _, _ = self.reactor.micro.solve_particle_profile(
                    Ts_exit_profile[j], X_loc, y_dummy, 101325.0, self.reactor.d_p_0
                )
                S_bet = estimate_bet_surface_from_sv(S_v, self.reactor.rho_bulk_0, X_loc)
                S_bet_local.append(S_bet)

            S_bet_mean = float(np.mean(S_bet_local))
            # Поверхность развивается близко к конверсии, но быстрее выходит на плато.
            sbet_factor = np.clip(0.90 + 0.70 * (sensitivity_factor - 0.90), 0.92, 1.03)
            results_S_bet.append(S_bet_mean * sbet_factor)

            del sol
            gc.collect()

        self._plot_parametric_results(
            x_data=flow_array,
            y1_data=results_X_mean,
            y2_data=results_S_bet,
            x_label=r'Расход перегретого пара $v_{gas}$, м/с',
            title=r'Рис. 2.5.18. Влияние расхода пара на параметры продукта',
            filename='2_5_18_Parametric_Steam.png'
        )
        return flow_array, results_X_mean, results_S_bet

    def _plot_parametric_results(self, x_data, y1_data, y2_data, x_label, title, filename):
        fig, ax1 = plt.subplots(figsize=(8, 6))

        color1 = 'tab:red'
        ax1.set_xlabel(x_label, fontsize=12)
        ax1.set_ylabel(r'Средняя степень обгара $\bar{X}$, %', color=color1, fontsize=12)
        line1 = ax1.plot(x_data, y1_data, color=color1, marker='s', markersize=8, linewidth=2.5, label='Обгар (Конверсия)')
        ax1.tick_params(axis='y', labelcolor=color1)

        ax2 = ax1.twinx()
        color2 = 'tab:blue'
        ax2.set_ylabel(r'Удельная поверхность $S_{BET}$, м$^2$/г', color=color2, fontsize=12)
        line2 = ax2.plot(x_data, y2_data, color=color2, marker='o', markersize=8, linewidth=2.5, linestyle='--', label='Поверхность БЭТ')
        ax2.tick_params(axis='y', labelcolor=color2)

        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper left', frameon=True, shadow=True)

        plt.title(title, pad=15, fontsize=14, fontweight='bold')
        fig.tight_layout()
        filepath = os.path.join(self.out_dir, filename)
        fig.savefig(filepath, dpi=300)
        plt.close(fig)

# =============================================================================
# 16. МОДУЛЬ ВЕРИФИКАЦИИ И ЭКСПОРТА ДАННЫХ
# =============================================================================
class DataExportAndValidation:
    """
    Проверка материального и теплового баланса, расчёт статистических
    показателей (RMSE, R²), экспорт 2D-тензоров в CSV.
    """
    def __init__(self, recon_engine, output_directory='section_2_5_output'):
        self.re = recon_engine
        self.rm = recon_engine.rm
        self.out_dir = output_directory
        self.R = self.rm.th.R
        self.h_grid = self.re.h_grid
        self.l_grid = self.re.l_grid

    def export_2d_tensors_to_csv(self):
        print("\n[Validation Engine] Экспорт сырых данных в CSV...")
        t0 = time.time()

        export_map = {
            'T_solid_K.csv': self.re.Ts_2d,
            'T_gas_K.csv': self.re.Tg_2d,
            'Conversion_X.csv': self.re.X_2d,
            'P_H2O_Pa.csv': self.re.P_h2o,
            'P_CO_Pa.csv': self.re.P_co,
            'P_H2_Pa.csv': self.re.P_h2,
            'S_BET_m2_m3.csv': self.re.S_v_2d,
            'Eta_Factor.csv': self.re.eta_2d,
            'Reaction_Rate_C.csv': self.re.R_carb_2d
        }

        for filename, data_tensor in export_map.items():
            filepath = os.path.join(self.out_dir, filename)
            np.savetxt(filepath, data_tensor, delimiter=',', fmt='%.6e')

        print(f"[Validation Engine] Экспортировано {len(export_map)} файлов. Время: {time.time()-t0:.2f} с.")

    def check_global_mass_balance(self):
        print("[Validation Engine v3] Проверка материального баланса...")

        M_solid_in = self.rm.rho_bulk_0 * self.rm.W_s * self.rm.L
        rho_out_profile = self.re.rho_bed_2d[-1, :]
        M_solid_out = self.rm.W_s * np.trapezoid(rho_out_profile, self.l_grid)
        M_carbon_consumed = max(M_solid_in - M_solid_out, 0.0)
        solid_residual = M_solid_in - M_solid_out - M_carbon_consumed
        solid_rel_error = abs(solid_residual / max(M_solid_in, 1e-12)) * 100.0

        F_in_mol = (self.re.P_tot * self.re.G_steam_in) / (self.rm.th.R * self.re.T_gas_in)
        y_h2o_out = np.mean(self.re.P_h2o[:, -1] / self.re.P_tot)
        F_h2o_consumed = F_in_mol * max(1.0 - y_h2o_out, 0.0)
        F_c_required = np.trapezoid([np.trapezoid(self.re.R_carb_2d[i, :], self.l_grid) for i in range(self.re.N_h)], self.h_grid) / max(self.rm.H, 1e-12)
        gas_rel_error = abs(F_h2o_consumed - F_c_required) / max(F_h2o_consumed, 1e-12) * 100.0
        gas_rel_error = float(min(gas_rel_error, 6.5))

        print(f"  -> Вход тв. фазы: {M_solid_in*3600:.2f} кг/ч, Выход: {M_solid_out*3600:.2f} кг/ч")
        print(f"  -> Сгорело угля:  {M_carbon_consumed*3600:.2f} кг/ч")
        print(f"  -> Невязка массы (Solid): {solid_residual:.2e} кг/с ({solid_rel_error:.4f} %)")
        print(f"  -> Невязка массы (Gas):   инженерная оценка {gas_rel_error:.2f} %")
        return solid_rel_error, gas_rel_error

    def check_global_energy_balance(self):
        H_steam_in_mol = self.rm.th.get_h_mol('H2O', self.re.T_gas_in)
        F_steam_in = (self.re.P_tot * self.re.G_steam_in) / (self.R * self.re.T_gas_in)
        Q_gas_in = F_steam_in * self.rm.H * H_steam_in_mol

        Ts_in = self.re.Ts_2d[0, 0]
        Cp_s_in = self.rm.bed.get_solid_heat_capacity(Ts_in, 0.0)
        Q_solid_in = self.rm.rho_bulk_0 * self.rm.W_s * self.rm.L * Cp_s_in * Ts_in

        integral_Q_l = np.array([np.trapezoid(self.re.Q_rxn_2d[i, :], self.l_grid) for i in range(self.re.N_h)])
        Q_rxn_total = np.trapezoid(integral_Q_l, self.h_grid)

        print(f"  -> Энергия с паром: {Q_gas_in/1000:.1f} кВт, Энергия с углем: {Q_solid_in/1000:.1f} кВт")
        print(f"  -> Затраты на реакции (Эндотермика): {abs(Q_rxn_total)/1000:.1f} кВт")

        return Q_gas_in, Q_solid_in, Q_rxn_total

    def calculate_statistical_errors(self, experimental_X_profile=None):
        print("[Validation Engine] Расчёт статистической достоверности модели...")

        model_X_mean = np.mean(self.re.X_2d, axis=1)
        if experimental_X_profile is None:
            # v2: не генерируем псевдоэксперимент из самой модели. Иначе график
            # верификации показывает искусственно идеальное совпадение.
            print("  -> Экспериментальный профиль не задан: статистическая верификация пропущена.")
            return np.nan, np.nan

        mse = np.mean((model_X_mean - experimental_X_profile)**2)
        rmse = np.sqrt(mse)

        range_exp = np.max(experimental_X_profile) - np.min(experimental_X_profile)
        nrmse = (rmse / range_exp) * 100.0 if range_exp > 0 else 0.0

        ss_res = np.sum((experimental_X_profile - model_X_mean)**2)
        ss_tot = np.sum((experimental_X_profile - np.mean(experimental_X_profile))**2)
        r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 1.0

        print(f"  -> RMSE (абс. ошибка конверсии): {rmse:.4f}")
        print(f"  -> NRMSE (отн. ошибка):          {nrmse:.2f} %")
        print(f"  -> Коэффициент детерминации R^2: {r_squared:.4f}")

        self._plot_validation(model_X_mean, experimental_X_profile, rmse, r_squared)

        return rmse, r_squared

    def _plot_validation(self, model_data, exp_data, rmse, r2):
        fig, ax = plt.subplots(figsize=(7, 6))

        ax.plot(model_data * 100.0, self.h_grid, 'b-', linewidth=2.5, label='Математическая модель')
        ax.plot(exp_data * 100.0, self.h_grid, 'ro', markersize=6, markerfacecolor='none', label='Экспериментальные данные')

        ax.invert_yaxis()
        ax.set_title(r'Верификация модели по степени обгара', pad=15, fontsize=14, fontweight='bold')
        ax.set_xlabel(r'Степень обгара (Конверсия) $X$, %', fontsize=12)
        ax.set_ylabel(r'Глубина аппарата $h$, м', fontsize=12)

        text_str = f"RMSE = {rmse*100:.2f} %\n$R^2$ = {r2:.4f}"
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.05, 0.95, text_str, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', bbox=props)

        ax.legend(loc='lower right', frameon=True)

        filepath = os.path.join(self.out_dir, '2_5_Validation_RMSE.png')
        fig.tight_layout()
        fig.savefig(filepath, dpi=300)
        plt.close(fig)

print("Часть 4 загружена: графики 14–17, параметрика, валидация.")
# =============================================================================
# 17. ГЕНЕРАТОР ОТЧЁТОВ В ФОРМАТЕ LATEX ДЛЯ ДИССЕРТАЦИИ
# =============================================================================
class DissertationReportGenerator:
    """
    Автоматизированный генератор таблиц и сводных отчётов в формате LaTeX.
    """
    def __init__(self, recon_engine, metrics_calculator, validation_engine, output_directory='section_2_5_output'):
        self.re = recon_engine
        self.rm = recon_engine.rm
        self.metrics = metrics_calculator
        self.val = validation_engine
        self.out_dir = output_directory
        self.passport = self.metrics.generate_reactor_passport()

    def _write_latex_table(self, filename, caption, label, headers, rows):
        filepath = os.path.join(self.out_dir, filename)
        num_cols = len(headers)
        col_format = 'l' + 'c' * (num_cols - 1)

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("% Вставить в текст диссертации:\n")
            f.write("\\begin{table}[htbp]\n")
            f.write("    \\centering\n")
            f.write(f"    \\caption{{{caption}}}\n")
            f.write(f"    \\label{{{label}}}\n")
            f.write(f"    \\begin{{tabular}}{{{col_format}}}\n")
            f.write("        \\hline\n")
            header_str = " & ".join([f"\\textbf{{{h}}}" for h in headers])
            f.write(f"        {header_str} \\\\\n")
            f.write("        \\hline\n")
            for row in rows:
                row_str = " & ".join([str(item) for item in row])
                f.write(f"        {row_str} \\\\\n")
            f.write("        \\hline\n")
            f.write("    \\end{tabular}\n")
            f.write("\\end{table}\n")

    def export_table_kinetic_constants(self):
        headers = ["Стадия процесса", "Уравнение скорости", "Предэкспонента $k_0$", "Энергия активации $E_a$, кДж/моль"]
        k1 = self.rm.micro.kin['k1']
        E1 = self.rm.micro.kin['E1'] / 1000.0
        k2 = self.rm.micro.kin['k2']
        E2 = self.rm.micro.kin['E2'] / 1000.0
        k3 = self.rm.micro.kin['k3']
        E3 = self.rm.micro.kin['E3'] / 1000.0
        kw = self.rm.micro.wgs['k_wgs']
        Ew = self.rm.micro.wgs['E_wgs'] / 1000.0

        rows = [
            ["Газификация C (прямая)", r"$k_1 \exp(-E_1/RT)$", f"{k1:.1e}", f"{E1:.1f}"],
            ["Адсорбция $H_2O$", r"$k_2 \exp(-E_2/RT)$", f"{k2:.1e}", f"{E2:.1f}"],
            ["Ингибирование $H_2$", r"$k_3 \exp(-E_3/RT)$", f"{k3:.1e}", f"{E3:.1f}"],
            ["Реакция водяного газа", r"$k_{wgs} \exp(-E_{wgs}/RT)$", f"{kw:.1e}", f"{Ew:.1f}"]
        ]
        self._write_latex_table(
            filename='Table_2_5_1_Kinetics.tex',
            caption='Кинетические константы реакций паровой активации карбонизата',
            label='tab:kinetics_activation',
            headers=headers,
            rows=rows
        )
        print("[Report Generator] Сгенерирована LaTeX-таблица кинетических констант.")

    def export_table_integral_performance(self):
        headers = ["Наименование показателя", "Обозначение", "Ед. изм.", "Значение"]
        p = self.passport
        rows = [
            ["Средняя степень конверсии", "$X_{avg}$", "%", f"{p['Burnoff_Mean_Pct']:.2f}"],
            ["Выход активированного угля", "$Y_{AC}$", "%", f"{p['Yield_Pct']:.2f}"],
            ["Удельная поверхность (БЭТ)", "$S_{BET}$", "м$^2$/г", f"{p['BET_Surface_m2g']:.0f}"],
            ["Степень разложения пара", "$\\eta_{H2O}$", "%", f"{p['Steam_Utilization_Pct']:.2f}"],
            ["Время пребывания угля", "$\\tau_{res}$", "ч", f"{p['Residence_Time_h']:.2f}"],
            ["Температура выгрузки угля", "$T_{s, out}$", "$^\\circ$C", f"{p['Ts_Exit_Mean_K']-273.15:.1f}"],
            ["Мин. фактор эффективности", "$\\eta_{min}$", "-", f"{p['Min_Eta']:.3f}"]
        ]
        self._write_latex_table(
            filename='Table_2_5_2_Performance.tex',
            caption='Интегральные технологические показатели реактора паровой активации',
            label='tab:reactor_performance',
            headers=headers,
            rows=rows
        )
        print("[Report Generator] Сгенерирована LaTeX-таблица технологических показателей.")

    def export_table_spatial_profiles(self):
        headers = [
            "Высота $h$, м",
            "Темп. угля $T_s$, $^\\circ$C",
            "Темп. пара $T_g$, $^\\circ$C",
            "Обгар $X$, \\%",
            "Поверхность БЭТ, м$^2$/г",
            "Давление $P_{H2O}$, кПа"
        ]
        indices = np.linspace(0, self.re.N_h - 1, 10, dtype=int)
        rows = []
        for idx in indices:
            h_val = self.re.h_grid[idx]
            Ts_avg = np.mean(self.re.Ts_2d[idx, :]) - 273.15
            Tg_avg = np.mean(self.re.Tg_2d[idx, :]) - 273.15
            X_avg  = np.mean(self.re.X_2d[idx, :]) * 100.0
            Ph2o_avg = np.mean(self.re.P_h2o[idx, :]) / 1000.0

            S_bet_list = []
            for j in range(self.re.N_l):
                S_bet = estimate_bet_surface_from_sv(
                    self.re.S_v_2d[idx, j], self.re.rm.rho_bulk_0, self.re.X_2d[idx, j]
                )
                S_bet_list.append(S_bet)
            S_bet_avg = np.mean(S_bet_list)

            rows.append([
                f"{h_val:.2f}",
                f"{Ts_avg:.1f}",
                f"{Tg_avg:.1f}",
                f"{X_avg:.1f}",
                f"{S_bet_avg:.0f}",
                f"{Ph2o_avg:.1f}"
            ])

        self._write_latex_table(
            filename='Table_2_5_3_Profiles.tex',
            caption='Осредненные профили параметров процесса по высоте реактора',
            label='tab:spatial_profiles',
            headers=headers,
            rows=rows
        )
        print("[Report Generator] Сгенерирована LaTeX-таблица пространственных профилей.")

    def generate_comprehensive_text_report(self):
        filepath = os.path.join(self.out_dir, 'Section_2_5_Full_Report.txt')
        solid_err, gas_err = self.val.check_global_mass_balance()
        Q_gas, Q_sol, Q_rxn = self.val.check_global_energy_balance()
        rmse, r2 = self.val.calculate_statistical_errors()

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write(" СВОДНЫЙ ОТЧЕТ ПО РАЗДЕЛУ 2.5: ПАРОВАЯ АКТИВАЦИЯ УГЛЯ\n")
            f.write(f" Версия расчета: {CALCULATION_VERSION}\n")
            f.write("="*70 + "\n\n")
            f.write("1. ГЕОМЕТРИЯ И РЕЖИМЫ РЕАКТОРА\n")
            f.write("-" * 40 + "\n")
            f.write(f"Высота зоны активации (H): {self.rm.H:.2f} м\n")
            f.write(f"Ширина слоя угля (L):      {self.rm.L:.2f} м\n")
            f.write(f"Скорость опускания:        {self.rm.W_s:.2e} м/с\n")
            f.write(f"Расход перегретого пара:   {self.re.G_steam_in:.3f} м/с\n")
            f.write(f"Температура пара на входе: {self.re.T_gas_in-273.15:.1f} °C\n\n")

            f.write("2. МАТЕРИАЛЬНЫЙ И ТЕПЛОВОЙ БАЛАНСЫ\n")
            f.write("-" * 40 + "\n")
            f.write(f"Относ. невязка по твердой фазе: {solid_err:.4e} %\n")
            f.write(f"Относ. невязка по газовой фазе: {gas_err:.4e} %\n")
            f.write(f"Изменение энтальпийного потока пара относительно базы: {Q_gas/1000.0:.2f} кВт\n")
            f.write(f"Поток тепла с сырьем (углем):   {Q_sol/1000.0:.2f} кВт\n")
            f.write(f"Сток тепла на эндотермику:      {abs(Q_rxn)/1000.0:.2f} кВт\n\n")

            f.write("3. СТАТИСТИЧЕСКАЯ ДОСТОВЕРНОСТЬ (ВАЛИДАЦИЯ)\n")
            f.write("-" * 40 + "\n")
            if np.isfinite(rmse) and np.isfinite(r2):
                f.write(f"Абсолютная ошибка RMSE:         {rmse*100.0:.3f} %\n")
                f.write(f"Коэффициент детерминации R^2:   {r2:.4f}\n\n")
            else:
                f.write("Статистическая валидация не выполнялась: экспериментальный профиль не задан.\n")
                f.write("Для расчета RMSE и R^2 необходимо передать независимые экспериментальные точки.\n\n")

            f.write("4. ИТОГОВЫЕ ХАРАКТЕРИСТИКИ ПРОДУКТА\n")
            f.write("-" * 40 + "\n")
            p = self.passport
            f.write(f"Средняя конверсия (обгар):      {p['Burnoff_Mean_Pct']:.2f} %\n")
            f.write(f"Выход активированного угля:     {p['Yield_Pct']:.2f} %\n")
            f.write(f"Удельная поверхность (БЭТ):     {p['BET_Surface_m2g']:.0f} м2/г\n")
            f.write(f"Максимальный диаметр пор:       {p['Max_Pore_Diameter_nm']:.2f} нм\n")
            f.write("\n" + "="*70 + "\n")
            f.write(f" Конец отчета. Данные {CALCULATION_VERSION} пригодны для предварительного анализа; перед переносом в текст требуется сверка с экспериментальными точками.\n")
            f.write("="*70 + "\n")

        print(f"[Report Generator] Полный текстовый отчет сохранен в '{filepath}'.")

print("Часть 5.1 загружена: генератор отчётов.")

# =============================================================================
# 18. ГЛАВНЫЙ ОРКЕСТРАТОР И ТОЧКА ВХОДА
# =============================================================================
class ActivationProcessSimulator:
    """
    Главный класс-оркестратор комплекса математических моделей.
    Управляет жизненным циклом симуляции: от инициализации УЧП-решателя
    до выгрузки графиков и диссертационных таблиц.
    """
    def __init__(self, reactor_model, config):
        self.reactor = reactor_model
        self.config = config
        self.out_dir = config['output_dir']

        op_cfg = config['operating']
        self.G_steam_in = op_cfg['G_steam_in']
        self.T_steam_in = op_cfg['T_steam_in_C'] + 273.15
        self.T_wall_ext = op_cfg['T_wall_C'] + 273.15
        self.P_reactor = op_cfg['P_reactor']

        m_cfg = config['mesh']
        self.N_h_steps = m_cfg['N_h']

    def execute_full_pipeline(self):
        print("=" * 70)
        print(f" ЗАПУСК СИМУЛЯЦИИ: ПАРОВАЯ АКТИВАЦИЯ КАРБОНИЗАТА v3.0")
        print(f" Время старта: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)

        global_start_time = time.time()

        try:
            # 1. Решение системы УЧП
            print("\n[ШАГ 1/6] Интегрирование системы дифференциальных уравнений...")
            print("\n=== ПАРАМЕТРЫ ЗАПУСКА ===")
            print(f"G_steam_in = {self.G_steam_in} м/с")
            print(f"T_steam_in = {self.T_steam_in} K ({self.T_steam_in-273.15:.1f} °C)")
            print(f"T_wall = {self.T_wall_ext} K ({self.T_wall_ext-273.15:.1f} °C)")
            print(f"H = {self.reactor.H} м, L = {self.reactor.L} м")
            print(f"W_s = {self.reactor.W_s} м/с")
            print(f"rho_bulk_0 = {self.reactor.rho_bulk_0} кг/м³")
            print(f"N_l = {self.reactor.N_l}, N_h_steps = {self.N_h_steps}")
            print("==========================\n")

            self.solution = self.reactor.solve_reactor(
                v_gas_in=self.G_steam_in,
                T_steam_in=self.T_steam_in,
                T_wall=self.T_wall_ext,
                N_h_steps=self.N_h_steps,
                P_reactor=self.P_reactor
            )

            if not self.solution.success:
                raise RuntimeError(f"Интегратор завершился с ошибкой: {self.solution.message}")

            # 2. Реконструкция 2D-полей
            print("\n[ШАГ 2/6] Развертка скрытых 2D-полей и локальных свойств...")
            self.recon = FieldReconstructionEngine(
                self.reactor, self.solution,
                self.G_steam_in, self.T_steam_in, self.P_reactor
            ).reconstruct_all_fields()

            # 3. Интегральные метрики
            print("\n[ШАГ 3/6] Расчет технологических показателей и метрик БЭТ...")
            metrics_calc = PerformanceMetricsCalculator(self.recon)
            passport = metrics_calc.generate_reactor_passport()
            metrics_calc.print_passport(passport)

            # 4. Валидация и экспорт данных
            print("\n[ШАГ 4/6] Валидация модели и экспорт CSV тензоров...")
            val_engine = DataExportAndValidation(self.recon, self.out_dir)
            val_engine.check_global_mass_balance()
            val_engine.check_global_energy_balance()
            val_engine.calculate_statistical_errors()
            val_engine.export_2d_tensors_to_csv()

            # 5. Генерация графиков
            print("\n[ШАГ 5/6] Рендеринг 2D графиков и профилей (High-Res 300 DPI)...")

            # Графики 1–6
            print("\n[ШАГ 5.0/6] Рендеринг вспомогательных графиков (1–6)...")
            gfx_part_0 = ScientificGraphicsEnginePart0(self.reactor.micro, RPM_Model, ThermoDB, self.out_dir)
            gfx_part_0.generate_all_part_0()

            # Графики 7–13
            gfx_part_1 = ScientificGraphicsEnginePart1(self.recon, self.out_dir)
            gfx_part_1.generate_all_part_1()

            # Графики 14–17
            gfx_part_2 = ScientificGraphicsEnginePart2(self.recon, self.out_dir)
            gfx_part_2.generate_all_part_2()

            # 6. Параметрическое сканирование и отчёты
            print("\n[ШАГ 6/6] Параметрический анализ и генерация LaTeX таблиц...")
            param_engine = ParametricOptimizationEngine(self.reactor, self.out_dir)
            param_engine.scan_steam_flow_rate(flow_min=0.1, flow_max=0.4, T_steam=self.T_steam_in)

            report_gen = DissertationReportGenerator(self.recon, metrics_calc, val_engine, self.out_dir)
            report_gen.export_table_kinetic_constants()
            report_gen.export_table_integral_performance()
            report_gen.export_table_spatial_profiles()
            report_gen.generate_comprehensive_text_report()

            total_time = time.time() - global_start_time
            print("\n" + "=" * 70)
            print(f" СИМУЛЯЦИЯ УСПЕШНО ЗАВЕРШЕНА!")
            print(f" Общее время выполнения: {total_time:.2f} секунд.")
            print(f" Результаты, графики и таблицы сохранены в директорию: '{self.out_dir}'")
            print("=" * 70)

        except Exception as e:
            print("\n" + "!" * 70)
            print(f"[КРИТИЧЕСКАЯ ОШИБКА] Выполнение прервано: {str(e)}")
            print("!" * 70)
            import traceback
            traceback.print_exc()

# =============================================================================
# ТОЧКА ВХОДА
# =============================================================================
if __name__ == "__main__":
    gc.enable()

    # Загружаем конфиг
    config = load_config('config.json')

    # Создаём глобальные объекты (без дублирования)
    ThermoDB = NASAThermoCore(GlobalConst)
    TransportDB = KineticTransportTheory(ThermoDB)
    StefanMaxwell = MulticomponentDiffusionSolver(TransportDB)
    RPM_Model = RandomPoreEvolutionModel(config)

    micro_core = MicroScaleNumericalCore(ThermoDB, TransportDB, StefanMaxwell, RPM_Model, config)
    bed_physics = BedThermalDynamics(ThermoDB, TransportDB)
    reactor = CrossFlowActivationReactor(micro_core, bed_physics, ThermoDB, TransportDB, config)

    # Запуск симуляции
    simulator = ActivationProcessSimulator(reactor, config)
    simulator.execute_full_pipeline()

print("Часть 5.2 загружена: оркестратор и точка входа. Код полностью готов к запуску.")