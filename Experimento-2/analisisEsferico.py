
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import constants as const
from scipy.optimize import brentq, minimize_scalar
from scipy.special import spherical_jn

# ==========================================
# 1. DATOS EXPERIMENTALES
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

# ==========================================
# 2. POTENCIAL DE MADELUNG
# ==========================================

def V0_madelung(a_nm):
    """Profundidad propuesta por Madelung, en eV."""
    # La constante M = 1.7476 requiere la distancia al primer vecino (R = a/2)
    R_m = (a_nm * 1e-9) / 2.0 
    V0_J = M_madelung * e**2 / (4 * np.pi * eps0 * R_m)
    return V0_J / e


# ==========================================
# 3. POZO ESFERICO FINITO
# ==========================================

def estados_sp(a_nm, V0_eV):
    """
    Resuelve los estados s y p más bajos de un pozo esférico.

    Las energías devueltas son positivas respecto del fondo
    del pozo. Para obtener las energías respecto del exterior,
    se resta V0_eV.

    Devuelve None para un estado que no esté ligado.
    """

    R = a_nm * 1e-9 / 2
    V0_J = V0_eV * e

    rho = R * np.sqrt(2 * m_e * V0_J) / hbar

    def resolver_l(l):
        # Para l=0, el estado ligado requiere rho > pi/2.
        # Para l=1, el primer estado ligado requiere rho > pi.
        umbral = np.pi / 2 if l == 0 else np.pi

        if rho <= umbral:
            return None

        def ecuacion(z):
            q2 = rho**2 - z**2

            if q2 <= 0:
                return np.nan

            q = np.sqrt(q2)

            if l == 0:
                # Empalme de u=sin(z) dentro y exp(-q) fuera
                return z / np.tan(z) + q

            # l=1: empalme de u=z*j_1(z) con la solución
            # radial decreciente exterior, proporcional a
            # exp(-q)*(1 + 1/q).
            j1 = spherical_jn(1, z)
            dj1 = spherical_jn(1, z, derivative=True)

            if abs(j1) < 1e-12:
                return np.nan

            return 1 + z * dj1 / j1 + q + 1 / (q + 1)

        if l == 0:
            lo = np.pi / 2 + 1e-8
            hi = min(rho, np.pi) - 1e-8
        else:
            # Primera raíz p, antes del primer cero de j1.
            lo = np.pi + 1e-8
            hi = min(rho, 4.4934094579) - 1e-8

        if hi <= lo:
            return None

        f_lo = ecuacion(lo)
        f_hi = ecuacion(hi)

        if not np.isfinite(f_lo) or not np.isfinite(f_hi):
            return None

        if f_lo * f_hi >= 0:
            return None

        z = brentq(ecuacion, lo, hi)

        # Energía respecto del fondo del pozo
        E_interior = V0_eV * (z / rho)**2

        # Energía respecto del potencial exterior V=0
        E_ligado = E_interior - V0_eV

        return {
            "z": z,
            "E_interior_eV": E_interior,
            "E_ligado_eV": E_ligado,
        }

    return {
        "rho": rho,
        "s": resolver_l(0),
        "p": resolver_l(1),
    }


def lambda_teorica(a_nm, V0_eV):
    """Longitud de onda de la transición s to p, en nm."""

    estados = estados_sp(a_nm, V0_eV)

    if estados["s"] is None or estados["p"] is None:
        return np.nan

    E_s = estados["s"]["E_interior_eV"]
    E_p = estados["p"]["E_interior_eV"]

    delta_E = E_p - E_s

    if delta_E <= 0:
        return np.nan

    return hc_eV_nm / delta_E


# ==========================================
# 4. CASO A: V0 FIJADO POR MADELUNG
# ==========================================

V0_M = []
rho_M = []
E_s_M = []
E_p_M = []
lambda_M = []

for _, row in df.iterrows():
    V0 = V0_madelung(row["a_nm"])
    estados = estados_sp(row["a_nm"], V0)

    V0_M.append(V0)
    rho_M.append(estados["rho"])

    if estados["s"] is None:
        E_s_M.append(np.nan)
    else:
        E_s_M.append(estados["s"]["E_ligado_eV"])

    if estados["p"] is None:
        E_p_M.append(np.nan)
    else:
        E_p_M.append(estados["p"]["E_ligado_eV"])

    lambda_M.append(lambda_teorica(row["a_nm"], V0))

df["V0_Madelung_eV"] = V0_M
df["rho_Madelung"] = rho_M
df["E_s_Madelung_eV"] = E_s_M
df["E_p_Madelung_eV"] = E_p_M
df["lambda_Madelung_nm"] = lambda_M

print("\n--- POTENCIAL DE MADELUNG FIJO ---")
print(df[[
    "cristal", "V0_Madelung_eV", "rho_Madelung",
    "E_s_Madelung_eV", "E_p_Madelung_eV",
    "lambda_Madelung_nm"
]].to_string(index=False))


# ==========================================
# 5. CASO B: INFERIR V0 PARA CADA CRISTAL
# ==========================================

def ajustar_V0_individual(a_nm, lambda_exp):
    """
    Busca el V0 que reproduce lambda_exp.
    Primero localiza intervalos válidos y después
    resuelve lambda_teorica(V0) - lambda_exp = 0.
    """

    def residuo(V0):
        lam = lambda_teorica(a_nm, V0)
        if not np.isfinite(lam):
            return np.nan
        return lam - lambda_exp

    # Barrido inicial para localizar soluciones.
    V_grid = np.geomspace(0.1, 1000, 2500)
    validos = []

    for V in V_grid:
        r = residuo(V)
        validos.append(r)

    raices = []

    for i in range(len(V_grid) - 1):
        r1, r2 = validos[i], validos[i + 1]

        if not np.isfinite(r1) or not np.isfinite(r2):
            continue

        if r1 * r2 < 0:
            raiz = brentq(
                residuo, V_grid[i], V_grid[i + 1]
            )
            raices.append(raiz)

    if not raices:
        return np.nan

    # Si aparecen varias soluciones, conservar la menor
    # dentro del intervalo explorado y reportarla explícitamente.
    return min(raices)


V0_fit = []
lambda_fit = []

for _, row in df.iterrows():
    V_fit = ajustar_V0_individual(
        row["a_nm"], row["lambda_nm"]
    )

    V0_fit.append(V_fit)

    if np.isfinite(V_fit):
        lambda_fit.append(
            lambda_teorica(row["a_nm"], V_fit)
        )
    else:
        lambda_fit.append(np.nan)

df["V0_ajustado_eV"] = V0_fit
df["lambda_ajustada_nm"] = lambda_fit
df["factor_V0"] = (
    df["V0_ajustado_eV"] / df["V0_Madelung_eV"]
)

print("\n--- POTENCIALES AJUSTADOS POR CRISTAL ---")
print(df[[
    "cristal", "lambda_nm", "lambda_Madelung_nm",
    "V0_Madelung_eV", "V0_ajustado_eV",
    "factor_V0", "lambda_ajustada_nm"
]].to_string(index=False))


# # ==========================================
# # 6. GRAFICOS COMPARATIVOS
# # ==========================================

# fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# # Longitudes de onda
# ax = axes[0]
# x = np.arange(len(df))
# ancho = 0.25

# ax.bar(
#     x - ancho, df["lambda_nm"], ancho,
#     label="Experimental"
# )

# ax.bar(
#     x, df["lambda_Madelung_nm"], ancho,
#     label="Madelung fijo"
# )

# ax.bar(
#     x + ancho, df["lambda_ajustada_nm"], ancho,
#     label="V0 ajustado"
# )

# ax.set_xticks(x)
# ax.set_xticklabels(df["cristal"])
# ax.set_ylabel("Longitud de onda (nm)")
# ax.set_title("Transición óptica s $\to$ p")
# ax.legend()
# ax.grid(axis="y", alpha=0.3)

# # Potenciales
# ax = axes[1]

# ax.plot(
#     df["cristal"], df["V0_Madelung_eV"],
#     "o-", label="Madelung"
# )

# ax.plot(
#     df["cristal"], df["V0_ajustado_eV"],
#     "s--", label="Ajustado"
# )

# ax.set_ylabel(r"$V_0$ (eV)")
# ax.set_title("Comparación de potenciales")
# ax.legend()
# ax.grid(alpha=0.3)

# plt.tight_layout()
# plt.show()

# ==========================================
# 6. GRÁFICO: LONGITUD DE ONDA VS CONSTANTE DE RED (a)
# ==========================================

fig, ax = plt.subplots(figsize=(9, 6))

# Graficar datos experimentales (Círculos azules, sin líneas)
ax.plot(df["a_nm"], df["lambda_nm"], 
        marker='o', markersize=8, linestyle='', color='royalblue',
        label="Medida (Experimental)", zorder=3)

# Graficar ajuste de V0 (Cuadrados rojos, sin líneas)
ax.plot(df["a_nm"], df["lambda_ajustada_nm"], 
        marker='s', markersize=6, linestyle='', color='red',
        label="Obtenida (V0 Ajustado)", zorder=4)

# Graficar modelo Madelung puro (Triángulos naranjas, sin líneas)
# Al tener NaN en LiF, NaCl y NaF, matplotlib solo dibujará el punto válido del KCl
ax.plot(df["a_nm"], df["lambda_Madelung_nm"], 
        marker='^', markersize=10, linestyle='', color='darkorange', 
        label="Obtenida (Madelung puro)", zorder=5)

# Etiquetar los cristales junto a cada punto experimental
for i, row in df.iterrows():
    ax.annotate(row["cristal"], (row["a_nm"], row["lambda_nm"]),
                xytext=(-10, 15), textcoords="offset points", 
                fontsize=11, fontweight='bold', color="black")

ax.set_xlabel(r"Constante de celda $a$ (nm)", fontsize=12)
ax.set_ylabel(r"Longitud de onda $\lambda$ (nm)", fontsize=12)
ax.set_title(r"Longitud de onda vs Constante de red (Transición $s \rightarrow p$)", fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()
plt.show()