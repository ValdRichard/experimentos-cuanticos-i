import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import constants as const
from scipy.optimize import brentq

# ==========================================
# DATOS EXPERIMENTALES Y CONSTANTES
# ==========================================


df = pd.DataFrame({
    "cristal": ["LiF", "NaCl", "NaF", "KCl"],
    "lambda_nm": [259.41, 461.26, 339.32, 558.3],
    "sigma_lambda_nm": [0.02, 0.09, 0.05, 0.5],
    "a_nm": [0.4028, 0.564, 0.4634, 0.6894],
})

M_madelung = 1.7476
hbar = const.hbar
m_e = const.m_e
e = const.e
eps0 = const.epsilon_0
hc_eV_nm = const.h * const.c / e * 1e9

def V0_madelung(a_nm):
    a_m = a_nm * 1e-9 / 2.0
    V0_J = M_madelung * e**2 / (4 * np.pi * eps0 * a_m)
    return V0_J / e

# ==========================================
# SOLUCIÓN DEL POZO CÚBICO FINITO (SEPARABLE)
# ==========================================

def solver_cubico(a_nm, V0_eV):
    R = a_nm * 1e-9 / 2
    V0_J = V0_eV * e
    rho = R * np.sqrt(2 * m_e * V0_J) / hbar

    def f_par(z):
        return z * np.tan(z) - np.sqrt(rho**2 - z**2)

    def f_impar(z):
        return -z / np.tan(z) - np.sqrt(rho**2 - z**2)

    # Raíz del estado fundamental (n=1)
    lo_1 = 1e-8
    hi_1 = min(rho, np.pi/2) - 1e-8
    if hi_1 <= lo_1 or f_par(lo_1) * f_par(hi_1) > 0:
        E1 = np.nan
    else:
        z1 = brentq(f_par, lo_1, hi_1)
        E1 = V0_eV * (z1 / rho)**2

    # Raíz del primer excitado (n=2)
    # En el pozo 1D, el estado impar existe si rho > pi/2
    if rho <= np.pi / 2:
        E2 = np.nan
    else:
        lo_2 = np.pi/2 + 1e-8
        hi_2 = min(rho, np.pi) - 1e-8
        if hi_2 <= lo_2 or f_impar(lo_2) * f_impar(hi_2) > 0:
            E2 = np.nan
        else:
            z2 = brentq(f_impar, lo_2, hi_2)
            E2 = V0_eV * (z2 / rho)**2

    return E1, E2

def lambda_cubico(a_nm, V0_eV):
    E1, E2 = solver_cubico(a_nm, V0_eV)
    if np.isnan(E1) or np.isnan(E2):
        return np.nan
    return hc_eV_nm / (E2 - E1)

# ==========================================
# EVALUACIÓN DE MADELUNG Y AJUSTE INVERSO
# ==========================================

def ajustar_V0_cubico(a_nm, lambda_exp):
    def residuo(V0):
        lam = lambda_cubico(a_nm, V0)
        return lam - lambda_exp if np.isfinite(lam) else np.nan

    V_grid = np.geomspace(0.1, 1000, 1000)
    validos = [residuo(V) for V in V_grid]
    
    for i in range(len(V_grid) - 1):
        if np.isfinite(validos[i]) and np.isfinite(validos[i+1]) and validos[i] * validos[i+1] < 0:
            return brentq(residuo, V_grid[i], V_grid[i+1])
    return np.nan

V0_M = []
lambda_M = []
V0_fit = []
lambda_fit = []

for _, row in df.iterrows():
    V0 = V0_madelung(row["a_nm"])
    V0_M.append(V0)
    lambda_M.append(lambda_cubico(row["a_nm"], V0))
    
    V_adj = ajustar_V0_cubico(row["a_nm"], row["lambda_nm"])
    V0_fit.append(V_adj)
    lambda_fit.append(lambda_cubico(row["a_nm"], V_adj))

df["V0_Madelung_eV"] = V0_M
df["lambda_Madelung_nm"] = lambda_M
df["V0_ajustado_eV"] = V0_fit
df["lambda_ajustada_nm"] = lambda_fit

print("--- RESULTADOS POZO CÚBICO FINITO ---")
print(df[["cristal", "a_nm", "lambda_nm", "lambda_Madelung_nm", "V0_Madelung_eV", "V0_ajustado_eV"]].to_string(index=False))

# ==========================================
# GRÁFICOS: DISPERSIÓN DE PUNTOS
# ==========================================

fig, ax = plt.subplots(figsize=(9, 6))

ax.plot(df["a_nm"], df["lambda_nm"], marker='o', markersize=8, linestyle='', 
        color='royalblue', label="Medida (Experimental)", zorder=3)

ax.plot(df["a_nm"], df["lambda_ajustada_nm"], marker='s', markersize=6, linestyle='', 
        color='red', label="Obtenida (V0 Ajustado)", zorder=4)

ax.plot(df["a_nm"], df["lambda_Madelung_nm"], marker='^', markersize=10, linestyle='', 
        color='darkorange', label="Obtenida (Madelung puro)", zorder=5)

for i, row in df.iterrows():
    ax.annotate(row["cristal"], (row["a_nm"], row["lambda_nm"]),
                xytext=(-10, 15), textcoords="offset points", 
                fontsize=11, fontweight='bold', color="black")

ax.set_xlabel(r"Constante de celda $a$ (nm)", fontsize=12)
ax.set_ylabel(r"Longitud de onda $\lambda$ (nm)", fontsize=12)
ax.set_title(r"Pozo Cúbico: Longitud de onda vs Constante de red (Transición $111 \rightarrow 211$)", fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()
plt.show()