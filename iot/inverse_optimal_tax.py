import numpy as np
import pandas as pd
import scipy.stats as st
from scipy.interpolate import UnivariateSpline
from statsmodels.nonparametric.kernel_regression import KernelReg
from sklearn.kernel_ridge import KernelRidge

class IOT:
    """
    Constructor for the IOT class.

    This IOT class can be used to compute the social welfare weights
    across the income distribution given data, tax policy parameters,
    and behavioral parameters.

    Args:
        data (Pandas DataFrame): micro data representing tax payers.
            Must include the following columns: income_measure,
            weight_var, mtr
        income_measure (str): name of income measure from data to use
        weight_var (str): name of weight measure from data to use
        inc_elast (scalar): compensated elasticity of taxable income
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
        dist_type="kde_full",
        kde_bw=None,
        mtr_smoother="spline",
        mtr_smooth_param=4,
    ):
        # keep the original data intact
        self.data_original = data.copy()
        # clean data based on upper and lower bounds
        # data = data[
        #     (data[income_measure] >= lower_bound)
        #     & (data[income_measure] <= upper_bound)
        # ]
        # create bins for analysis
        bins = np.arange(
            start=lower_bound, stop=upper_bound + bandwidth, step=bandwidth
        )
        data.loc[:, ["z_bin"]] = pd.cut(
            data[income_measure], bins, include_lowest=True
        )
        self.inc_elast = inc_elast
        self.z, self.f, self.f_prime = self.compute_income_dist(
            data, income_measure, weight_var, dist_type, kde_bw
        )
        self.mtr, self.mtr_prime = self.compute_mtr_dist(
            data, weight_var, income_measure, mtr_smoother, mtr_smooth_param
        )
        self.theta_z = 1 + ((self.z * self.f_prime) / self.f)
        self.g_z = self.sw_weights()

    def df(self):
        """
        Return all vector attributes in a DataFrame format

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

    def compute_mtr_dist(
        self, data, weight_var, income_measure, mtr_smoother, mtr_smooth_param
    ):
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
            tuple:
                * mtr (array_like): mean marginal tax rate for each income bin
                * mtr_prime (array_like): rate of change in marginal tax rates
                    for each income bin
        """
        # sort dataframe on income
        data = data.sort_values(by=income_measure)
        if mtr_smoother == "spline":
            spl = UnivariateSpline(
                data[income_measure], data["mtr"], w=data[weight_var], k=mtr_smooth_param
            )
            mtr = spl(self.z)
        elif mtr_smoother == "kr":
            krr = KernelRidge(alpha=1.0)
            krr.fit(data[income_measure], data["mtr"], sample_weight=data[weight_var])
            mtr = krr.predict(self.z)
        else:
            pass
        mtr_prime = np.gradient(mtr, edge_order=2)

        return mtr, mtr_prime

    def compute_income_dist(
        self, data, income_measure, weight_var, dist_type, kde_bw=None
    ):
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
            tuple:
                * z (array_like): mean income at each bin in the income
                    distribution
                * f (array_like): density for income bin z
                * f_prime (array_like): slope of the density function for
                    income bin z
        """
        z_line = np.linspace(1, 100000, 100000)

        if dist_type == "log_normal":
            mu = (
                np.log(data[income_measure] + 1) * data[weight_var]
            ).sum() / data[weight_var].sum()
            sigmasq = (
                (
                    ((np.log(data[income_measure] + 1) - mu) ** 2)
                    * data[weight_var]
                ).values
                / data[weight_var].sum()
            ).sum()
            print("Type mu = ", mu)
            print("Type sigma = ", np.sqrt(sigmasq))
            f = st.lognorm.pdf(z_line, s=(sigmasq) ** 0.5, scale=np.exp(mu))
        elif dist_type == "kde_full":
            # uses the original full data for kde estimation
            f_function = st.gaussian_kde(
                self.data_original[income_measure].values,
                bw_method=kde_bw,
                weights=self.data_original[weight_var].values,
            )
            f = f_function.pdf(z_line)
        elif dist_type == "kde_subset":
            # uses the subsetted data for kde estimation
            f_function = st.gaussian_kde(
                data[income_measure],
                bw_method=kde_bw,
                weights=data[weight_var],
            )
            f = f_function.pdf(z_line)
        else:
            assert False
        # normalize f
        f = f / np.sum(f)
        f_prime = np.gradient(f, edge_order=2)  # this works a bit better than finite differences, but still not great

        return z_line, f, f_prime

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
            + (
                (self.inc_elast * self.z * self.mtr_prime)
                / (1 - self.mtr) ** 2
            )
        )

        return g_z


def wm(value, weight):
    """
    Weighted mean function that allows for zero division

    Args:
        value (array_like): values to be averaged
        weight (array_like): weights for each value

    Returns:
        scalar: weighted average
    """
    try:
        return np.average(value, weights=weight)
    except ZeroDivisionError:
        return np.nan
