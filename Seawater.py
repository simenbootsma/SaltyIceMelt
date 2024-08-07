import numpy as np


class Seawater:
    def __init__(self, temperature, salinity):
        """ temperature in degrees Celsius and salinity in g/kg at atmospheric pressure. """
        self.t = temperature
        self.s = salinity

    def density(self, t=None, s=None):
        return self.density_millero(t, s)

    def density_derivative_t(self, t=None, s=None):
        return self.density_derivative_t_millero(t, s)

    def density_derivative_s(self, t=None, s=None):
        return self.density_derivative_s_millero(t, s)

    def density_sharqawy(self, t=None, s=None):
        if t is None:
            t = self.t
        if s is None:
            s = self.s

        """
        Computes density of seawater in kg/m^3.
        Function taken from Eq. 8 in Sharqawy2010.
        Valid in the range 0 < t < 180 degC and 0 < sal < 160 g/kg.
        Accuracy: 0.1%
        """
        s = s * 1e-3  # g/kg -> kg/kg
        a = [9.999e2, 2.034e-2, -6.162e-3, 2.261e-5, -4.657e-8]
        b = [8.02e2, -2.001, 1.677e-2, -3.06e-5, -1.613e-5]

        rho_sw = np.sum([a[i] * t**i for i in range(5)], axis=0) \
                 + np.sum([b[i] * t**i * s for i in range(4)], axis=0) \
                 + b[4] * t ** 2 * s**2
        return rho_sw

    def density_millero(self, t=None, s=None):
        if t is None:
            t = self.t
        if s is None:
            s = self.s

        """
        Computes density of seawater in kg/m^3.
        Function taken from Eq. 6 in Sharqawy2010.
        Valid in the range 0 < t < 40 degC and 0.5 < sal < 43 g/kg.
        Accuracy: 0.01%
        """
        t68 = t / (1 - 2.5e-4)  # inverse of Eq. 4 in Sharqawy2010
        sp = s / 1.00472  # inverse of Eq. 3 in Sharqawy2010

        rho_0 = 999.842594 + 6.793952e-2 * t68 - 9.095290e-3 * t68 ** 2 + 1.001685e-4 * t68 ** 3 - 1.120083e-6 * t68 ** 4 + 6.536336e-9 * t68 ** 5
        A = 8.24493e-1 - 4.0899e-3 * t68 + 7.6438e-5 * t68 ** 2 - 8.2467e-7 * t68 ** 3 + 5.3875e-9 * t68 ** 4
        B = -5.72466e-3 + 1.0227e-4 * t68 - 1.6546e-6 * t68 ** 2
        C = 4.8314e-4
        rho_sw = rho_0 + A * sp + B * sp ** (3 / 2) + C * sp ** 2
        return rho_sw

    def density_derivative_t_sharqawy(self, t=None, s=None):
        """
        Computes partial derivative of density of seawater to temperature in kg/m^3/K.
        Function taken from Eq. 6 in Sharqawy2010.
        Valid in the range 0 < t < 40 degC and 0.5 < sal < 43 g/kg.
        Accuracy: 0.01%
        """

        if t is None:
            t = self.t
        if s is None:
            s = self.s

        # # Function taken from Eq. 6 in Sharqawy2010, valid up to 40 degrees C and 43 g/kg
        # t68 = t / (1 - 2.5e-4)  # inverse of Eq. 4 in Sharqawy2010
        #
        # drho0_dt68 = 6.793952e-2 - 2 * 9.095290e-3 * t68 + 3 * 1.001685e-4 * t68 ** 2 - 4 * 1.120083e-6 * t68 ** 3 + 5 * 6.536336e-9 * t68 ** 4
        # drho_dt = drho0_dt68 / (1 - 2.5e-4)

        # Eq. 8 from Sharqawy for S up to 160 g/kg
        s = s * 1e-3  # g/kg -> kg/kg
        a = [9.999e2, 2.034e-2, -6.162e-3, 2.261e-5, -4.657e-8]
        b = [8.02e2, -2.001, 1.677e-2, -3.06e-5, -1.613e-5]

        drho_dt = np.sum([i * a[i] * t ** (i-1) for i in range(1, 5)], axis=0) \
                  + np.sum([i * b[i] * t ** (i-1) * s for i in range(1, 4)], axis=0) \
                  + 2 * b[4] * t * s ** 2
        return drho_dt

    def density_derivative_s_sharqawy(self, t=None, s=None):
        """
        Computes partial derivative of density of seawater to salinity in kg/m^3/(g/kg).
        Function taken from Eq. 6 in Sharqawy2010.
        Valid in the range 0 < t < 40 degC and 0.5 < sal < 43 g/kg.
        Accuracy: 0.01%
        """

        if t is None:
            t = self.t
        if s is None:
            s = self.s

        # # Function taken from Eq. 6 in Sharqawy2010, valid up to 40 degrees C and 43 g/kg
        # t68 = t / (1 - 2.5e-4)  # inverse of Eq. 4 in Sharqawy2010
        # sp = s / 1.00472  # inverse of Eq. 3 in Sharqawy2010
        #
        # A = 8.24493e-1 - 4.0899e-3 * t68 + 7.6438e-5 * t68 ** 2 - 8.2467e-7 * t68 ** 3 + 5.3875e-9 * t68 ** 4
        # B = -5.72466e-3 + 1.0227e-4 * t68 - 1.6546e-6 * t68 ** 2
        # C = 4.8314e-4
        # drho_dsp = A + (3 / 2) * B * sp ** (1 / 2) + 2 * C * sp
        # drho_ds = drho_dsp / 1.00472

        # Eq. 8 from Sharqawy for S up to 160 g/kg
        s = s * 1e-3  # g/kg -> kg/kg
        b = [8.02e2, -2.001, 1.677e-2, -3.06e-5, -1.613e-5]
        drho_ds = np.sum([b[i] * t ** i for i in range(4)], axis=0) + 2 * b[4] * t ** 2 * s
        drho_ds *= 1e-3  # back to per g/kg
        return drho_ds

    def density_derivative_t_millero(self, t=None, s=None):
        """
        Computes partial derivative of density of seawater to temperature in kg/m^3/K.
        Function taken from Eq. 6 in Sharqawy2010.
        Valid in the range 0 < t < 40 degC and 0.5 < sal < 43 g/kg.
        Accuracy: 0.01%
        """

        if t is None:
            t = self.t
        if s is None:
            s = self.s

        # Function taken from Eq. 6 in Sharqawy2010, valid up to 40 degrees C and 43 g/kg
        t68 = t / (1 - 2.5e-4)  # inverse of Eq. 4 in Sharqawy2010
        sp = s / 1.00472  # inverse of Eq. 3 in Sharqawy2010

        drho0_dt68 = 6.793952e-2 - 2 * 9.095290e-3 * t68 + 3 * 1.001685e-4 * t68 ** 2 - 4 * 1.120083e-6 * t68 ** 3 + 5 * 6.536336e-9 * t68 ** 4
        dA_dt68 = - 4.0899e-3 + 2 * 7.6438e-5 * t68 - 3 * 8.2467e-7 * t68 ** 2 + 4 * 5.3875e-9 * t68 ** 3
        dB_dt68 = 1.0227e-4 - 2 * 1.6546e-6 * t68
        drho_dt = (drho0_dt68 + dA_dt68 * sp + dB_dt68 * sp**(3/2)) / (1 - 2.5e-4)
        return drho_dt

    def density_derivative_s_millero(self, t=None, s=None):
        """
        Computes partial derivative of density of seawater to salinity in kg/m^3/(g/kg).
        Function taken from Eq. 6 in Sharqawy2010.
        Valid in the range 0 < t < 40 degC and 0.5 < sal < 43 g/kg.
        Accuracy: 0.01%
        """

        if t is None:
            t = self.t
        if s is None:
            s = self.s

        # Function taken from Eq. 6 in Sharqawy2010, valid up to 40 degrees C and 43 g/kg
        t68 = t / (1 - 2.5e-4)  # inverse of Eq. 4 in Sharqawy2010
        sp = s / 1.00472  # inverse of Eq. 3 in Sharqawy2010

        A = 8.24493e-1 - 4.0899e-3 * t68 + 7.6438e-5 * t68 ** 2 - 8.2467e-7 * t68 ** 3 + 5.3875e-9 * t68 ** 4
        B = -5.72466e-3 + 1.0227e-4 * t68 - 1.6546e-6 * t68 ** 2
        C = 4.8314e-4
        drho_dsp = A + (3 / 2) * B * sp ** (1 / 2) + 2 * C * sp
        drho_ds = drho_dsp / 1.00472
        return drho_ds

    def density_ratio(self):
        return -self.density_derivative_t() * self.t / (self.density_derivative_s() * self.s)

    def specific_heat(self):
        """
        Computes specific heat of seawater in J/kg/K.
        Function taken from Eq. 9 in Sharqawy2010.
        Valid in the range 0 < t < 120 degC and 0 < sal < 180 g/kg.
        Accuracy: 0.28%
        """

        T68 = 273.15 + self.t / (1 - 2.5e-4)  # inverse of Eq. 4 in Sharqawy2010
        sp = self.s / 1.00472  # inverse of Eq. 3 in Sharqawy2010

        A = 5.328 - 9.76e-2 * sp + 4.04e-4 * sp**2
        B = -6.913e-3 + 7.351e-4 * sp - 3.15e-6 * sp**2
        C = 9.6e-6 - 1.927e-6 * sp + 8.23e-9 * sp**2
        D = 2.5e-9 + 1.666e-9 * sp - 7.125e-12 * sp**2
        c_sw = A + B * T68 + C * T68**2 + D * T68**3
        return c_sw * 1e3

    def thermal_conductivity(self):
        # """
        # Computes thermal conductivity of seawater in W/m/K.
        # Function taken from Eq. 14 in Sharqawy2010.
        # Valid in the range 0 < t < 60 degC and 0 < sal < 60 g/kg.
        # Accuracy: 0.5%
        # """
        #
        # t68 = self.t / (1 - 2.5e-4)  # inverse of Eq. 4 in Sharqawy2010
        # sp = self.s / 1.00472  # inverse of Eq. 3 in Sharqawy2010
        # p = 1.0  # [MPa], atmospheric pressure
        # k_sw = 0.5715 * (1 + 0.003 * t68 - 1.025e-5 * t68**2 + 6.53e-3 * p - 0.00029 * sp)
        # return k_sw

        """
        Computes thermal conductivity of seawater in W/m/K.
        Function taken from Eq. 13 in Sharqawy2010.
        Valid in the range 0 < t < 180 degC and 0 < sal < 160 g/kg.
        Accuracy: 3%
        """

        t68 = self.t / (1 - 2.5e-4)  # inverse of Eq. 4 in Sharqawy2010
        sp = self.s / 1.00472  # inverse of Eq. 3 in Sharqawy2010
        logk = np.log10(240 + 0.0002 * sp) + 0.434 * (2.3 - (343.5 + 0.037*sp)/(t68 + 273.15)) * (1 - (t68 + 273.15)/(647 + 0.03*sp)) ** 0.333
        k_sw = 1e-3 * 10**logk
        return k_sw

    def dynamic_viscosity(self):
        # """
        # Computes dynamic viscosity of seawater in kg/m/s.
        # Function taken from Eq. 18 in Sharqawy2010.
        # Valid in the range 5 < t < 25 degC and 0 < sal < 40 g/kg.
        # Accuracy: 0.5%
        # """
        #
        # t68 = self.t / (1 - 2.5e-4)  # inverse of Eq. 4 in Sharqawy2010
        # sp = self.s / 1.00472  # inverse of Eq. 3 in Sharqawy2010
        # Cl = self.density() * sp / 1806.55  # between Eqs. 18 and 19 in Sharqaqy2010
        # # Cl = self.s / 1.815068  # inverse of Eq. 2 in Sharqawy2010
        # A = 5.185e-5 * t68 + 1.0675e-4
        # B = 3.3e-5 * t68 + 2.591e-3
        # mu_w = 1.002e-3 * 10**((1.1709*(20-t68) - 0.001827 * (t68-20)**2)/(t68+89.93))
        # mu_sw = mu_w * (1 + A * Cl**(1/2) + B * Cl)
        # return mu_sw

        """
        Computes dynamic viscosity of seawater in kg/m/s.
        Function taken from Eq. 22 in Sharqawy2010.
        Valid in the range 0 < t < 180 degC and 0 < sal < 150 g/kg.
        Accuracy: 1.5%
        """

        s = self.s * 1e-3
        A = 1.541 + 1.998e-2 * self.t - 9.52e-5 * self.t**2
        B = 7.974 - 7.561e-2 * self.t + 4.742e-4 * self.t**2
        mu_w = 4.2844e-5 + (0.157*(self.t + 64.993)**2 - 91.296)**(-1)  # from IAPWS 2008
        mu_sw = mu_w * (1 + A*s + B*s**2)
        return mu_sw

    def kinematic_viscosity(self):
        return self.dynamic_viscosity() / self.density()

    def freezing_temperature(self):
        Kf = 1.86  # [K.kg/mol] cryoscopic constant of water
        M = 58.44  # [g/mol] molar mass of NaCl
        i = 2      # [-] Van 't Hoff constant of NaCl
        return 0.0 - Kf*i*self.s/M  # Bagden's law for freezing depression

    def transition_temperature(self):
        """
        Temperature at which transition between melting and dissolution occurs.
        See Wells2011 Eq. 6.1.
         """
        D = 1.68e-9  # diffusivity of NaCl [m^2/s]
        L = 3.34e5  # latent heat of fusion [J/kg]
        return np.sqrt(D/self.thermal_diffusivity()) * L / self.specific_heat()

    def thermal_diffusivity(self):
        """ alpha = k/(rho * cp) """
        return self.thermal_conductivity() / (self.density() * self.specific_heat())

    def saline_diffusivity(self):
        """ From Caldwell 1974 """
        return (62 + 3.63 * self.t) * 1e-11  # m^2/s

    def prandtl_number(self):
        """ Pr = nu/alpha """
        return self.dynamic_viscosity() / (self.thermal_diffusivity() * self.density())


def example():
    T = np.linspace(0, 20, 10)   # Temperature in degrees Celsius
    S = np.ones(T.size) * 35    # Salinity in g/kg

    sw = Seawater(T, S)
    rho = sw.density()
    mu = sw.dynamic_viscosity()

    print("Density of water at T = {:.1f} degrees Celsius and S = {:.1f} g/kg is: {:.1f} kg/m3".format(T[0], S[0], rho[0]))
    print("Dynamic viscosity of water at T = {:.1f} degrees Celsius and S = {:.1f} g/kg is: {:.1f} kg/m3".format(T[0], S[0],
                                                                                                       mu[0]))

