from functools import partial
import uncertainties
import numpy as np
import pandas as pd
import scipy.odr as odr
import scipy.optimize as opt
import matplotlib.pyplot as plt
# Se utiliza el modelo de pozo finito, seguiré cada una de sus definiciones 

# tan(theta) = srqt(theta_0^2 /theta^2 -1 ) Pares 
# -cot(theta) = sqrt(theta_0^2/theta^2 -1 ) Impares

# donde theta(E,L) = k(E) L/2 = sqrt(2m_e E) L / 2hbar



# Universal constants
hbar = 6.582119569e-16 # eV * s
c = 299792458 # m/s
m_e = 510.998e3 / c**2 # eV / c^2

def swap_ev_and_nm(v: float) -> float:
    # Convert an energy in eV to a wavelength in nm, and vice-versa
    return 1240 / v


# Obtener el k(E).
def get_fsw_k(e: float) -> float:
    '''Energy in eV'''
    return np.sqrt(2 * m_e * e) / hbar


# Obtener el theta(E,L), el l se tiene que pasar en nanometros, por como está definido. 
def get_fsw_theta(e: float, l: float) -> float:
    return get_fsw_k(e) * (l * 1e-9 / 2)

# Obtener la energía, para esto invierte la relación que tengo con el theta. 
def get_fsw_energy(theta: float, l: float) -> float:
    '''Invert the theta definition'''
    return 2 * hbar**2 * theta**2 / (m_e * (l * 1e-9)**2)

# Obtener el ancho de la cadena. En particular usa 0.134 y esta en nm. Es coherente con nuestro modelo. 
def get_chain_length(p: int) -> float:
    '''Get chain length in nm from the number of C-C links'''
    l_cc = 0.134 # [nm], length of C-C links
    return (p+2) * l_cc

# Encuentra el HOMO, sabe cuantos electroes hay, lo divide entre dos, y ahí lo seleccióna. Hace un redondeo de esa division y le resta uno. Si tengo 5 electrones 5//2 = 3, entonces el HOMO es el 2. 
def get_homo_e(energies: list[float], p: int) -> float:
    '''Get the HOMO (Highest Occupied Molecular Orbital) energy from the list of energy levels'''
    n_electrons = p + 3
    if len(energies) < n_electrons / 2: return np.nan
    return sorted(energies)[(n_electrons // 2) - 1]

# Lo mismo que el de arriba pero al revés jajaja. 
def get_lumo_e(energies: list[float], p: int) -> float:
    '''Get the LUMO (Lowest Unoccupied Molecular Orbital) energy from the list of energy levels'''
    n_electrons = p + 3
    if len(energies) < n_electrons / 2 + 1: return np.nan
    return sorted(energies)[(n_electrons // 2 + 1) - 1] # Esto esta al re pedo, no sé porque lo hizo así. 

# Tachan, el que te interesa. 
def get_homo_lumo_transition(energies: list[float], p: int) -> float:
    '''Get the HOMO-LUMO transition energy from the list of energy levels'''
    return get_lumo_e(energies, p) - get_homo_e(energies, p)



# Truco ingenioso, elimina por completo los que sean negativos, para que se optimice. Si hay valores negativos directamente los pone como nan, que es un tipo que será ignorado por el análisis. 
def negative_to_nan(v: float | np.ndarray) -> float | np.ndarray:
    '''
    Turn negative values to np.nan.
    
    Implementing it this way significantly speeds up the numeric methods
    vs vectorizing a single function.
    '''
    if isinstance(v, float):
        return np.nan if v < 0 else v
    elif isinstance(v, np.ndarray):
        arr = v.copy()
        arr[arr < 0] = np.nan
        return arr
    
    raise TypeError('Input must be a float or a numpy array')

# Elimina los estados negativos de energía de la parte par 
def fsw_even_fn(x: float | np.ndarray) -> float | np.ndarray:
    return negative_to_nan(np.tan(x))

# Elimina los estados negativos de energía de la parte impar
def fsw_odd_fn(x: float | np.ndarray) -> float | np.ndarray:
    return negative_to_nan(-1/np.tan(x))

# Parte compartida por las soluciones impares o pares, que depende si del otro lado hay un tg o -cotg
def fsw_g(theta: float, theta0: float) -> float:
    return np.sqrt((theta0 / theta)**2 - 1)

# Encuentra las raices, método entendible, habrá que ver como define el min y max. 
def find_all_roots(f, x_min: float, x_max: float, step: float | None = None, n_steps: int | None = None) -> list[float]:
    '''
    Use a sliding window approach to find all roots of the function f in the interval [x_min, x_max].
    '''
    if step and n_steps:
        raise ValueError('Cannot specify both step and n_steps')
    if not step:
        if not n_steps: n_steps = 1000
        step = (x_max - x_min)/n_steps
    xs = np.arange(x_min, x_max, step)

    roots = []
    # Se declararon dos variables, de manera que esto va de dos caminos en dos caminos, elimina el ultimo de xs para que lo recorra x1 y elimina elprimero para que lo recorra x2. 
    for x1, x2 in zip(xs[:-1], xs[1:]):
        y1, y2 = f(x1), f(x2)
        
        # Check if function crosses zero in [x1, x2]
        if y1 == 0:
            # Exact zero at grid point
            roots.append(x1)
        elif y2 == 0:
            # Exact zero at grid point
            roots.append(x2)
        elif y1 * y2 < 0:
            # There is a root within the open interval
            try:
                # We'll find it with brentq
                roots.append(opt.brentq(f, x1, x2))
            except ValueError:
                # No root despite sign change (rare, numerical issue)
                pass

    # Round roots to overcome limits in the numerical precision,
    # and remove duplicates
    return np.unique(np.round(roots, 12))

# Ahora sí, la resolución natural de los estados ligados. 
def get_cyanines_fsw_bound_states(v0: float, p: int) -> tuple[list[float], list[float]]:
    '''
    Solve the finite square well equation numerically, and get the bound states' energy levels.

    @param v0: Depth of the well in eV.
    @param p: Number of C-C links in the cyanine chain.

    @return: A tuple with two lists, the first one containing the even states' theta values, and the second one the odd states' theta values.
    '''
    theta0 = get_fsw_theta(e=v0, l=get_chain_length(p))

    # We want to get the zeros of:
    # Even modes: tan(theta) - g(theta),
    # Odd modes: -cot(theta) - g(theta).

    # First, find the range in which we'll look for intersections between the curves.
    search_min = 1e-4 # Avoid the singularities at 0
    # The maximum number of bound states will be when theta == theta0.
    search_max = theta0

    even_zeros_fn = lambda theta: fsw_even_fn(theta) - fsw_g(theta, theta0)
    even_zeros = find_all_roots(even_zeros_fn, search_min, search_max)

    odd_zeros_fn = lambda theta: fsw_odd_fn(theta) - fsw_g(theta, theta0)
    odd_zeros = find_all_roots(odd_zeros_fn, search_min, search_max)

    return list(even_zeros), list(odd_zeros)

# Acá las vamos a ordenar para asi poder saber que transiciones son. 
def get_cyanines_fsw_energies(v0: float, p: int) -> list[float]:
    '''
    Solve the finite square well equation numerically, and get the bound states' energy levels.

    @param v0: Depth of the well in eV.
    @param p: Number of C-C links in the cyanine chain.

    @return: A list with all bound states' energy levels in eV.
    '''

    even_states, odd_states = get_cyanines_fsw_bound_states(v0, p)
    return sorted(get_fsw_energy(np.array(even_states + odd_states), get_chain_length(p)))

# Plotear. 
def plot_cyanines_fsw_bound_states(v0: float = 10, p: int = 5, save: str | None = None):
    '''
    Solve the finite square well equation graphically, and get the bound states' energy levels.

    @param v0: Depth of the well in eV.
    @param p: Number of C-C links in the cyanine chain.
    @param save: If specified, save the plot to this file.
    '''
    l = get_chain_length(p)
    theta0 = get_fsw_theta(e=v0, l=l)

    # X-axis values
    theta = np.linspace(0.001, theta0, 400)

    # Clear the previous plot (in case we're using an interactive backend)
    plt.cla()

    # Draw the function
    plt.plot(theta, fsw_g(theta, theta0), label=r'$\sqrt{\frac{\theta_0^2}{\theta^2}-1}$')
    plt.plot(theta, fsw_even_fn(theta), label=r'$\tan(\theta)$ (even states)', linestyle='--')
    plt.plot(theta, fsw_odd_fn(theta), label=r'$-\cot(\theta)$ (odd states)', linestyle='--')

    plt.axvline(theta0, linestyle=':', color='gray')
    plt.xlabel(r'$\theta$')
    plt.ylabel('y')
    plt.ylim(-0.5, 10)

    plt.legend(loc='upper right')


    # Now solve the equation numerically
    even_solutions, odd_solutions = get_cyanines_fsw_bound_states(v0, p)

    energies = get_fsw_energy(np.array(even_solutions + odd_solutions), l)
    homo_e = get_homo_e(energies, p)
    lumo_e = get_lumo_e(energies, p)

    def process_solutions(solutions: list[float], fn) -> tuple[list[float], list[float]]:
        x_lst = []
        y_lst = []
        for x in solutions:
            if x is np.nan: continue
            y = fn(x)

            x_lst.append(x)
            y_lst.append(y)

            if y > 10: continue

            e = get_fsw_energy(x, get_chain_length(p))

            sfx = ''
            if homo_e != np.nan and round(e - homo_e, 5) == 0: sfx = '\nHOMO'
            elif lumo_e != np.nan and round(e - lumo_e, 5) == 0: sfx = '\nLUMO'

            plt.text(x + theta0/40, y + 0.1, f'{e:.2f} eV' + sfx, horizontalalignment='left')

        return x_lst, y_lst

    es_x, es_y = process_solutions(even_solutions, fsw_even_fn)
    plt.plot(es_x, es_y, 'o', color='purple', linestyle='None')

    os_x, os_y = process_solutions(odd_solutions, fsw_odd_fn)
    plt.plot(os_x, os_y, 'o', color='purple', linestyle='None')

    if save:
        plt.savefig(save)
1
    # plt.show()

# CASO de prueba, asume que ya sabemos el v0, y que estamos trabajando con el p5. 
plot_cyanines_fsw_bound_states(v0=10, p=5, save='./Experimento-1/output/fsw_solutions_v10_p5.jpeg')

# 1. Definir tus datos originales
datos_OXO = [
    {'p': 5, 'lam_exp': 486.56, 'err': 0.02},
    {'p': 7, 'lam_exp': 584.31, 'err': 0.03},
    {'p': 9, 'lam_exp': 688.23, 'err': 0.03}
]

datos_TIO = [
    {'p': 5, 'lam_exp': 559.76, 'err': 0.05},
    {'p': 7, 'lam_exp': 655.27, 'err': 0.02},
    {'p': 9, 'lam_exp': 761.63, 'err': 0.02}
]

# 2. Armar spectrums_per_p iterando sobre tus listas
spectrums_per_p = {
    'Oxo': {dato['p']: dato['lam_exp'] for dato in datos_OXO},
    'Tia': {dato['p']: dato['lam_exp'] for dato in datos_TIO}
}

# Intenta minimizar los datos que tengo experimentales y el modelo predicho. 
def get_v0_from_homo_lumo_transition(e: float, p: int) -> float:
    '''Get the finite square well potential V0 from the HOMO-LUMO transition energy'''
    
    # Minimize the difference between the experimental observation and the model prediction
    # to get the per-cyanine V0
    def fn(v0: float) -> float:
        model_e = get_homo_lumo_transition(get_cyanines_fsw_energies(v0, p), p)
        return model_e - e
    # Esta buscando valores desde el 0ev hasta 20ev, y los esta revisando todos. Como no puse una configuración dividira el intervalo en 1000 partes y ahí va chequeando tal que la diferencia sea cero, de manera que evalua los distintos saltos posibles. 
    roots = find_all_roots(fn, 0, 20)
    if len(roots) != 1:
        raise ValueError(f'Did not find a unique solution for {e=} and {p=}')
    
    return roots[0]

# NOTE: This takes ~1m to run, be patient!
oxo_v0s = {
    p: get_v0_from_homo_lumo_transition(swap_ev_and_nm(lam_exp), p)
    for p, lam_exp in spectrums_per_p['Oxo'].items()
}

tia_v0s = {
    p: get_v0_from_homo_lumo_transition(swap_ev_and_nm(lam_exp), p)
    for p, lam_exp in spectrums_per_p['Tia'].items()
}

plt.plot(oxo_v0s.keys(), oxo_v0s.values(), 'o', label='Oxo-cyanines', linestyle='None')
plt.plot(tia_v0s.keys(), tia_v0s.values(), 'o', label='Tia-cyanines', linestyle='None')

plt.legend()
plt.xlabel('Number of carbon bonds ($p$)')
plt.ylabel('Finite square well potential $V_0$ [eV]')

plt.xticks(ticks=list(oxo_v0s.keys()), labels=list(oxo_v0s.keys()))

plt.savefig('Experimento-1/output/fsw_v0s_fit.jpeg')

# plt.show()

# QUEEEEEEE, esto es buenisimo, esta por graficarme el grafico del chi2 para que yo vea literalmente si encontré un mínimo. 
def _fsw_chi2(v0: float, spectrums: dict) -> float:
    '''Calculate the chi-squared value for a set of spectra's energies, and the FSW predictions'''
    total = 0

    for p, lam_exp in spectrums.items():
        # model_e calcula la longitud de onda (en nm) teórica
        model_e = swap_ev_and_nm(get_homo_lumo_transition(get_cyanines_fsw_energies(v0, p), p))
        # Se resta directamente contra tu dato experimental
        total += (lam_exp - model_e) ** 2

    return total

# We need to vectorize this function, so that the libraries can use it over an np.ndarray
fsw_chi2 = np.vectorize(_fsw_chi2, excluded=['spectrums'])


v0s = np.linspace(5, 11, 200)

plt.plot(v0s, fsw_chi2(v0s, spectrums=spectrums_per_p['Oxo']), label='Oxo-cyanines')
plt.plot(v0s, fsw_chi2(v0s, spectrums=spectrums_per_p['Tia']), label='Tia-cyanines')
plt.legend()
plt.xlabel('Finite square well potential $V_0$ [eV]')
plt.ylabel(r'$\chi^2$')
plt.yscale('log')

plt.savefig('Experimento-1/output/fsw_chi2.jpeg')

# plt.show()


def find_v0_common(spectrums: dict):
    """Find the common V0 value for a series of spectra."""

    # Función escalar: recibe un V0 y devuelve un único chi2
    def objective(v0):
        v0 = float(np.asarray(v0).item())
        return float(_fsw_chi2(v0, spectrums))

    # Buscar el mínimo
    roots = opt.minimize_scalar(
        objective,
        bounds=(1, 20),  
        method='bounded'
    )

    if not roots.success:
        raise RuntimeError(roots.message)

    print(roots)

    # Calcular la segunda derivada del chi2 en el mínimo
    import numdifftools as nd

    hess = nd.Hessian(objective, step=1e-5)
    hess_value = float(np.asarray(hess(roots.x)).item())

    # Incertidumbre estadística aproximada
    sigma = np.sqrt(2 / hess_value)

    return uncertainties.ufloat(float(roots.x), float(sigma))

def run_fsw_fit(series: str) -> dict[int, float | uncertainties.UFloat]:
    print(f'Fitting {series}-cyanines V0')
    v0 = find_v0_common(spectrums_per_p[series])
    print(f'Found V0 = {v0}\n')
    # Guardamos el V0 directamente en la función para usarlo en el gráfico
    if series == 'Oxo':
        run_fsw_fit.last_oxo = v0
    else:
        run_fsw_fit.last_tio = v0
    ret = {}
    for p in [5, 7, 9]:
        # We will manually propapate the uncertainty in V0 to the energy levels,
        # by calculating the energies at V0, V0 + 1 sigma, and V0 - 1 sigma.
        # This is an approximation, but it's good enough for our purposes.
        energies_nom = get_cyanines_fsw_energies(v0.n, p)
        energies_p1sigma = get_cyanines_fsw_energies(v0.n + v0.std_dev, p)
        energies_m1sigma = get_cyanines_fsw_energies(v0.n - v0.std_dev, p)

        energies = []
        for i, e_nom in enumerate(energies_nom):
            # We first need to check that we have an ith state when we calculated the energy levels with V0 +/- 1 sigma
            unc_p1sigma = abs(energies_p1sigma[i] - e_nom) if len(energies_p1sigma) >= i else 0
            unc_m1sigma = abs(energies_m1sigma[i] - e_nom) if len(energies_m1sigma) >= i else 0

            # Since the uncertainties package does not support asymmetric uncertainties,
            # we will symmetrize them by taking the largest uncertainty
            energies.append(uncertainties.ufloat(e_nom, max(unc_p1sigma, unc_m1sigma)))

        ret[p] = swap_ev_and_nm(get_homo_lumo_transition(energies, p))

    return ret
def plot_all(models: dict, y_range: tuple = None, save: str = None):
    """
    Grafica la longitud de onda vs el número de carbonos (p).
    Compara los datos experimentales directos con los modelos teóricos calculados.
    """
    plt.figure(figsize=(9, 6))
    ax = plt.gca()

    # 1. Extraemos y graficamos tus datos experimentales (con sus errores)
    p_oxo = [dato['p'] for dato in datos_OXO]
    lam_oxo = [dato['lam_exp'] for dato in datos_OXO]
    err_oxo = [dato['err'] for dato in datos_OXO]

    p_tio = [dato['p'] for dato in datos_TIO]
    lam_tio = [dato['lam_exp'] for dato in datos_TIO]
    err_tio = [dato['err'] for dato in datos_TIO]

    plt.errorbar(p_oxo, lam_oxo, yerr=err_oxo, fmt='ks', markersize=7, 
                 label='Experimental Oxo', capsize=4)
    plt.errorbar(p_tio, lam_tio, yerr=err_tio, fmt='k^', markersize=7, 
                 label='Experimental Tia', capsize=4)

    # 2. Graficamos los modelos teóricos que pasaste en el diccionario
    colores = ['blue', 'red']
    
    for i, (nombre_modelo, datos_modelo) in enumerate(models.items()):
        p_vals = list(datos_modelo.keys())
        y_vals = []
        y_errs = []
        
        for p in p_vals:
            val = datos_modelo[p]
            # Extraemos el valor nominal y la desviación estándar del paquete uncertainties
            if hasattr(val, 'n') and hasattr(val, 'std_dev'):
                y_vals.append(val.n)
                y_errs.append(val.std_dev)
            else:
                y_vals.append(val)
                y_errs.append(0)
                
        plt.errorbar(p_vals, y_vals, yerr=y_errs, fmt='--o', 
                     color=colores[i % len(colores)], label=nombre_modelo, capsize=4)

    # 3. Formato del gráfico
    plt.xlabel('Número de enlaces (p)', fontsize=12)
    plt.ylabel(r'Longitud de onda $\lambda$ [nm]', fontsize=12)
    plt.title('Comparación: Modelo de Pozo Finito vs Experimental', fontsize=14)
    
    if y_range:
        plt.ylim(y_range)
        
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    
    if save:
        plt.savefig(save)
        print(f"Gráfico guardado en: {save}")

    # Textos con los V0 reales obtenidos por el ajuste
    v0_oxo = getattr(run_fsw_fit, 'last_oxo', 'N/A')
    v0_tio = getattr(run_fsw_fit, 'last_tio', 'N/A')

    info_text = (
        "Parámetros del Modelo\n"
        "---------------------\n"
        f"Oxo $V_0$: {v0_oxo:.4f} eV\n"
        f"Tia $V_0$: {v0_tio:.4f} eV"
    )

    # Propiedades de la caja con fondo blanco limpio
    box_properties = dict(
        boxstyle="round,pad=0.5",  
        facecolor="white",         # Fondo blanco
        edgecolor="black",         # Borde negro simple
        alpha=0.9                  
    )

    # Colocar la caja de texto en el gráfico
    ax.text(
        0.05, 0.95,               
        info_text, 
        transform=ax.transAxes,   
        fontsize=10, 
        verticalalignment='top',  
        bbox=box_properties       
    )
        
    # plt.show()
# Tu llamada final:
plot_all({
    'Pozo finito (Oxo-cianinas)': run_fsw_fit('Oxo'),
    'Pozo finito (Tia-cianinas)': run_fsw_fit('Tia'),
}, y_range=(400, 800), save='Experimento-1/output/all_models.jpeg')


# Error porcentual entre lambda experimental y lambda teórica

for serie, datos in [('Oxo', datos_OXO), ('Tia', datos_TIO)]:
    print(f'\n{serie}-cianinas')

    modelo = run_fsw_fit(serie)

    for dato in datos:
        p = dato['p']
        lam_exp = dato['lam_exp']
        lam_teo = modelo[p].n if hasattr(modelo[p], 'n') else modelo[p]

        error_porcentual = abs(lam_teo - lam_exp) / lam_exp * 100

        print(
            f'p = {p}: '
            f'λ_exp = {lam_exp:.2f} nm, '
            f'λ_teo = {lam_teo:.2f} nm, '
            f'error = {error_porcentual:.2f} %'
        )

# Distancia en sigmas entre lambda experimental y lambda teórica

for serie, datos in [('Oxo', datos_OXO), ('Tia', datos_TIO)]:
    print(f'\n{serie}-cianinas')

    modelo = run_fsw_fit(serie)

    for dato in datos:
        p = dato['p']

        lam_exp = dato['lam_exp']
        sigma_exp = dato['err']

        # Valor teórico y su incertidumbre
        lam_teo = modelo[p].n
        sigma_teo = modelo[p].std_dev

        # Distancia en sigmas
        n_sigma = abs(lam_teo - lam_exp) / np.sqrt(
            sigma_exp**2 + sigma_teo**2
        )

        print(
            f'p = {p}: '
            f'λ_exp = {lam_exp:.2f} ± {sigma_exp:.2f} nm, '
            f'λ_teo = {lam_teo:.2f} ± {sigma_teo:.2f} nm, '
            f'Δ = {n_sigma:.2f} σ'
        )

# Guardar comparación experimental vs teórica en un archivo .txt

ruta_txt = 'Experimento-1/output/comparacion_lambdas.txt'

with open(ruta_txt, 'w', encoding='utf-8') as f:
    for serie, datos in [('Oxo', datos_OXO), ('Tia', datos_TIO)]:
        f.write(f'{serie}-cianinas\n')
        f.write('=' * 50 + '\n')

        modelo = run_fsw_fit(serie)

        for dato in datos:
            p = dato['p']

            lam_exp = dato['lam_exp']
            sigma_exp = dato['err']

            lam_teo = modelo[p].n
            sigma_teo = modelo[p].std_dev

            # Error porcentual
            error_porcentual = (
                abs(lam_teo - lam_exp) / lam_exp * 100
            )

            # Distancia en sigmas
            n_sigma = abs(lam_teo - lam_exp) / np.sqrt(
                sigma_exp**2 + sigma_teo**2
            )

            f.write(
                f'p = {p}\n'
                f'  λ experimental = {lam_exp:.2f} ± {sigma_exp:.2f} nm\n'
                f'  λ teórica      = {lam_teo:.2f} ± {sigma_teo:.2f} nm\n'
                f'  Error porcentual = {error_porcentual:.2f} %\n'
                f'  Distancia       = {n_sigma:.2f} σ\n'
                f'\n'
            )

        f.write('\n')

print(f'Comparación guardada en: {ruta_txt}')