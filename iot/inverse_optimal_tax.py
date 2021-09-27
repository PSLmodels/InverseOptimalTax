import numpy as np
import pandas as pd
import scipy.stats as st
from scipy.interpolate import UnivariateSpline
from iot.utils import wavg


class IOT:
    """
    Constructor for the IOT class.

    This IOT class can be used to compute the social welfare weights
    across the income distribution given data, tax policy parametesr,
    and behavioral parameters.

    Args:
        data (Pandas DataFrame): micro data representing tax payers.
            Must include the following columns: income_measure,
            weight_var, mtr
        income_measure (str): name of income measure from data to use
        weight_var (str): name of weight measure from data to use
        inc_elast (scalar): compensated elasiticy of taxable income
            w.r.t. the marginal tax rate
        bandwidth (scalar): size of income bins in units of income
        lower_bound (scalar): minimum income to consider
        upper_bound (scalar): maximum income to consider
        dist_type (None or str): type of distribution to use if
            parametric, if None, then non-parametric bin weights
        mtr_smoother (None or str): method used to smooth our mtr
            function, if None, then use bin average mtrs

    Returns:
        class instance: IOT
    """

    def __init__(
        self,
        data,
        income_measure="expanded_income",
        weight_var="s006",
        inc_elast=0.25,
        bandwidth=1000,
        lower_bound=0,
        upper_bound=500000,
        dist_type="log_normal",
        mtr_smoother="cubic_spline",
    ):

        # clean data based on upper and lower bounds
        data = data[
            (data[income_measure] > lower_bound)
            & (data[income_measure] <= upper_bound)
        ]
        # create bins for analysis
        bins = np.arange(
            start=lower_bound, stop=upper_bound + bandwidth, step=bandwidth
        )
        data["z_bin"] = pd.cut(data[income_measure], bins)
        self.inc_elast = inc_elast
        self.z, self.f, self.f_prime = self.compute_income_dist(
            data, income_measure, weight_var, dist_type
        )
        self.mtr, self.mtr_prime = self.compute_mtr_dist(
            data, weight_var, mtr_smoother
        )
        self.theta_z = 1 + ((self.z * self.f_prime) / self.f)
        self.g_z = self.sw_weights()

    def df(self):
        """
        Return all vector attributs in a dataframe format

        Args:
            None

        Returns:
            df (Pandas DataFrame): DataFrame with all inputs/outputs
                for each income bin
        """
        dict_out = {
            "z": self.z,
            "f": self.f,
            "f_prime": self.f_prime,
            "mtr": self.mtr,
            "mtr_prime": self.mtr_prime,
            "theta_z": self.theta_z,
            "g_z": self.g_z,
        }
        df = pd.DataFrame.from_dict(dict_out)
        return df

    def compute_mtr_dist(self, data, weight_var, mtr_smoother):
        """
        Compute marginal tax rates over the income distribution and
        their derivative.

        Args:
            data (Pandas DataFrame): micro data representing tax payers.
                Must include the following columns: income_measure,
                weight_var, mtr
            weight_var (str): name of weight measure from data to use
            mtr_smoother (None or str): method used to smooth our mtr
            function, if None, then use bin average mtrs

        Returns:
            mtr (array_like): mean marginal tax rate for each income bin
            mtr_prime (array_like): rate of change in marginal tax rates
                for each income bin
        """
        data_group = (
            data[["mtr", "z_bin", weight_var]]
            .groupby(["z_bin"])
            .apply(wavg, "mtr", weight_var)
        )
        if mtr_smoother == "cubic_spline":
            spl = UnivariateSpline(self.z, data_group.values)
            mtr = spl(self.z)
        else:
            mtr = data_group.values

        mtr_prime = np.diff(mtr) / np.diff(self.z)
        mtr_prime = np.append(mtr_prime, mtr_prime[-1])

        return mtr, mtr_prime

    def compute_income_dist(self, data, income_measure, weight_var, dist_type):
        """
        Compute the distribution of income (parametrically or not) from
        the raw data.

        This method computes the probability density function and its
        derivative.

        Args:
            data (Pandas DataFrame): micro data representing tax payers.
                Must include the following columns: income_measure,
                weight_var, mtr
            income_measure (str): name of income measure from data to
                use
            weight_var (str): name of weight measure from data to use
            dist_type (None or str): type of distribution to use if
                parametric, if None, then non-parametric bin weights

        Returns:
            tuple: z (array_like): mean income at each bin in the income
            distribution

            f (array_like): density for income bin z

            f_prime (array_like): slope of the density function for
            income bin z
        """
        data_group = (
            data[[income_measure, "z_bin", weight_var]]
            .groupby(["z_bin"])
            .apply(wavg, income_measure, weight_var)
        )
        z = data_group.values

        if dist_type == "log_normal":
            mu = (
                np.log(data[income_measure]) * data[weight_var]
            ).sum() / data[weight_var].sum()
            sigmasq = (
                (((np.log(data[income_measure]) - mu) ** 2) * data[weight_var])
                / data[weight_var].sum()
            ).sum()
            f = st.lognorm.pdf(z, s=(sigmasq) ** 0.5, scale=np.exp(mu))
            f = f / f.sum()
        else:
            f = (
                data[[weight_var, "z_bin"]].groupby("z_bin").sum()
                / data[weight_var].sum()
            ).s006.values

        # Compute rate of change in pdf
        f_prime = np.diff(f) / np.diff(z)
        # assume diff between last bin and next is zero
        f_prime = np.append(f_prime, 0)

        return z, f, f_prime

    def sw_weights(self):
        r"""
        Returns the social welfare weights for a given tax policy.

        See Jacobs, Jongen, and Zoutman (2017)

        .. math::
            g_{z} = 1 + \theta_z \varepsilon^{c}\frac{T'(z)}{(1-T'(z))} +
            \varepsilon^{c}\frac{zT''(z)}{(1-T''(z))^{2}}

        Args:
            None

        Returns:
            array_like: vector of social welfare weights across
            the income distribution
        """
        g_z = (
            1
            + ((self.theta_z * self.inc_elast * self.mtr) / (1 - self.mtr))
            + ((self.inc_elast * self.z * self.mtr_prime) / (1 - self.mtr) ** 2)
         )

        return g_z
