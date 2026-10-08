from typing import Callable
from functools import partial

import numpy as np
import pandas as pd
import scipy.odr as odr
import scipy.optimize as opt
import matplotlib.pyplot as plt
import uncertainties

from adjustText import adjust_text

# Universal constants
hbar = 6.582119569e-16 # eV * s
c = 299792458 # m/s
m_e = 510.998e3 / c**2 # eV / c^2

# Lo dice el nombre, convierte de eV en longitud de onda en nanometros. 
def swap_ev_and_nm(v: float) -> float:
    # Convert an energy in eV to a wavelength in nm, and vice-versa
    return 1240 / v

# Esta haciendo una función de resonancia, si traduzco es basicamente una gaussiana, b0 es la amplitd b1 es el centroid y b2 es el sigma. b3 es la constante para que se levante. 
def resonance(B: list[float], x: float) -> float:
    return B[0] * np.exp(-((x - B[1])**2) / (2 * B[2]**2)) + B[3]
resonance_model = odr.Model(resonance)

# ¿Porque se llamará FZ? Más alla de eso, esto es nuevo para la gran mayoria de mis amigos, está definiendo un objeto, que es relacionada de manera teórica a Programación Orientada a Objetos (POO), en criollo, es una especie de molde, que permitirá crear la cantidad que queramos de estos bichos que vendran y mantendran sus propiedades. Piensenlo como un personaje. 
class FZSpectrum:
    def __init__(self, name: str, filename: str, x_range: tuple[float, float] | None = None):
        self.name = name

        # Load the spectrums
        df = pd.read_csv(filename, header=0)
        # Esto seguramente no funcione para mis archivos, por como estan exportados. 
        # Check if the series is in the file
        if f'I({name})' not in df.columns:
            raise ValueError(f'Series I({name}) not found in file {filename}. Available series: {", ".join([col for col in df.columns if col.startswith("I(")])}')
        # Ley de Lambert - Beer, absorbancia! 
        # Calculate the absorbance
        df['Absorbance'] = -np.log10(df[f'I({name})'] / df['I(0)'])
        # Tira las filas que tengan datos vacios
        # Drop rows with NaN values
        df = df.dropna(subset=['Wavelength', 'Absorbance'])
        # TOCA INVESTIGAR  que tipo de sensor tenemos y su error, este tipo dice que no impacta pero lo tenemos que hacer! 
        # We don't know the spectrometer that was used, so we'll assume some common uncertainties
        # The impact on the fits is minimal, so it's not critical if we're not more precise
        df['Wavelength unc'] = [0.1 for _ in df['Wavelength']]
        df['Absorbance unc'] = [0.003 for _ in df['Absorbance']]
        # Se queda con las columnas que le importan, me parece que este código no va a servir para nosotros. Veremos. 
        # Keep only the relevant columns
        self.df = df[['Wavelength', 'Wavelength unc', 'Absorbance', 'Absorbance unc']]

        self.fit_result: odr.Output | None = None
        self.x_range = x_range if x_range is not None else (self.df['Wavelength'].min(), self.df['Wavelength'].max())
    # Acá ya hay algo importante, esta función esta definida dentro de una clase! eso significa que le pertenece a este bicho, para invocarla podriamos poner el nombre de la clase, por ejemplo, NaCl y si queremos su fit ponemos NaCl.fit()
    def fit(
            self, 
            x_range: tuple[float, float] | None = None, 
            initial_mu: float | None = None, 
            initial_sigma: float | None = None, 
            initial_a0: float | None = None,
            initial_y0: float | None = None,
            fix_y0: bool = False,
    ) -> odr.Output:
        # Notar que todo usa el 'self' esto es, las variables definidas sobre el objeto, son de acceso libre por cada una de sus funciones. 
        if x_range is None: x_range = self.x_range

        x_data_mask = (self.df['Wavelength'] >= x_range[0]) & (self.df['Wavelength'] <= x_range[1])
        fit_data = odr.RealData(
            self.df['Wavelength'][x_data_mask], 
            self.df['Absorbance'][x_data_mask], 
            sx=self.df['Wavelength unc'][x_data_mask],
            sy=self.df['Absorbance unc'][x_data_mask],
        )

        # Pone condiciones iniciales manuales. 
        if initial_mu is None:
            max_absorbance_index = self.df['Absorbance'].idxmax()
            initial_mu = self.df['Wavelength'].iloc[max_absorbance_index]
        if initial_sigma is None: initial_sigma = 50
        if initial_a0 is None: initial_a0 = self.df['Absorbance'].iloc[self.df['Wavelength'].searchsorted(initial_mu)]
        if initial_y0 is None: initial_y0 = 0
        initial_params = [initial_a0, initial_mu, initial_sigma, initial_y0]

        # List of 0/1 to indicate which parameters are fixed (0) or free (1), following ODRPACK's convention
        fix_params = [1, 1, 1, int(not fix_y0)]

        fit = odr.ODR(fit_data, resonance_model, beta0=initial_params, ifixb=fix_params)
        self.fit_result = fit.run()
        print(f'Fit stop: {self.fit_result.stopreason}')

        return self.fit_result
    # Acá vemos las propiedades, bastante prolijo para poder saber que dato estoy pidiendo, aprendes un montón! 
    @property
    def a0(self) -> uncertainties.UFloat | None:
        if self.fit_result is None: return None
        return uncertainties.ufloat(self.fit_result.beta[0], self.fit_result.sd_beta[0])

    @property
    def mu(self) -> uncertainties.UFloat | None:
        if self.fit_result is None: return None
        return uncertainties.ufloat(self.fit_result.beta[1], self.fit_result.sd_beta[1])

    @property
    def sigma(self) -> uncertainties.UFloat | None:
        if self.fit_result is None: return None
        return uncertainties.ufloat(self.fit_result.beta[2], self.fit_result.sd_beta[2])

    @property
    def y_0(self) -> uncertainties.UFloat | None:
        if self.fit_result is None: return None
        return uncertainties.ufloat(self.fit_result.beta[3], self.fit_result.sd_beta[3])
    # Función libre que me permitira obtener todos los parametros! 
    def get_fit_params_df(self) -> pd.DataFrame | None:
        if self.fit_result is None: return None
        df = pd.DataFrame({
            'mu': [self.mu],
            'sigma': [self.sigma],
            'A': [self.a0],
            'y_0': [self.y_0],
        })
        df.index = [self.name]
        return df

    def plot(self, mask: bool = False, x_range: tuple[float, float] | None = None, save: str | None = None):
        if mask or x_range:
            if not x_range: x_range = self.x_range
            mask = (self.df['Wavelength'] >= x_range[0]) & (self.df['Wavelength'] <= x_range[1])
        else:
            mask = [True for _ in self.df['Wavelength']]

        plt.errorbar(
            x=self.df['Wavelength'][mask], 
            y=self.df['Absorbance'][mask], 
            xerr=self.df['Wavelength unc'][mask], 
            yerr=self.df['Absorbance unc'][mask],
            label=self.name,
        )
        if self.fit_result is not None:
            x_fit = np.linspace(self.df['Wavelength'][mask].min(), self.df['Wavelength'][mask].max(), 100)
            plt.plot(x_fit, resonance(self.fit_result.beta, x_fit), color='red', label='Fit')

        plt.legend()

        plt.xlabel('Wavelength [nm]')
        plt.ylabel('Absorbance [a.u.]')

        if save:
            plt.savefig(save)


def plot_all_spectrums(spectrums: list[FZSpectrum], plot_fits: bool = False, apply_show_masks: bool = False, shift_to_zero: bool = False, norm: bool = False, norm_to_max: bool = False, save: str | None = None):
    for i, spectrum in enumerate(spectrums):
        if apply_show_masks:
            mask = (spectrum.df['Wavelength'] >= spectrum.x_range[0]) & (spectrum.df['Wavelength'] <= spectrum.x_range[1])
        else:
            mask = np.array([True for _ in spectrum.df['Wavelength']])

        # Shift to zero
        shift = 0
        if shift_to_zero:
            shift = spectrum.df['Absorbance'][mask].min()

        scale = 1.0
        if norm:
            scale = 1/sum(spectrum.df['Absorbance'][mask] - shift)
        elif norm_to_max:
            scale = 1/(spectrum.df['Absorbance'][mask].max() - shift)

        plt.errorbar(
            x=spectrum.df['Wavelength'][mask], 
            y=(spectrum.df['Absorbance'][mask] - shift) * scale, 
            xerr=spectrum.df['Wavelength unc'][mask], 
            yerr=spectrum.df['Absorbance unc'][mask] * scale, 
            color=f'C{i}', 
            label=spectrum.name
        )
        if plot_fits and spectrum.fit_result is not None:
            x_fit = np.linspace(spectrum.df['Wavelength'][mask].min(), spectrum.df['Wavelength'][mask].max(), 100)
            plt.plot(x_fit, (resonance(spectrum.fit_result.beta, x_fit) - shift) * scale, color=f'C{i}', linestyle='dashed')

    plt.legend()

    plt.xlabel('Wavelength [nm]')
    plt.ylabel('Absorbance [a.u.]')

    if save:
        plt.savefig(save)


def get_all_params(spectrums: list[FZSpectrum], mu_only: bool = False) -> pd.DataFrame:
    df_list = []
    for spectrum in spectrums:
        params_df = spectrum.get_fit_params_df()
        if params_df is not None: df_list.append(params_df)
    if df_list:
        df = pd.concat(df_list)

        if mu_only:
            return df[['mu']]
        else:
            return df
    else:
        return pd.DataFrame()