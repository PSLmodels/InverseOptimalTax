import numpy as np
import pandas as pd
import scipy.stats as st
import scipy
from statsmodels.nonparametric.kernel_regression import KernelReg
from scipy.interpolate import UnivariateSpline
from scipy.linalg import lstsq


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
        eti (scalar): compensated elasticity of taxable income
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
        income_measure="e00200",
        weight_var="s006",
        eti=0.25,
        dist_type="log_normal",
        kde_bw=None,
        mtr_smoother="kreg",
        mtr_smooth_param=3,
    ):
        # keep the original data intact
        self.data_original = data.copy()
        # clean data based on upper and lower bounds
        # data = data[
        #     (data[income_measure] >= lower_bound)
        #     & (data[income_measure] <= upper_bound)
        # ]
        # Get income distribution
        self.z, self.F, self.f, self.f_prime = self.compute_income_dist(
            data, income_measure, weight_var, dist_type, kde_bw
        )
        # see if eti is a scalar
        if isinstance(eti, float):
            self.eti = eti
        else:  # if not, then it should be a dict with keys containing lists as values
            # check that same number of ETI values as knot points
            assert len(eti["knot_points"]) == len(eti["eti_values"])
            # want to interpolate across income distribution with knot points
            # assume that eti can't go beyond 1 (or the max of the eti_values provided)
            if len(eti["knot_points"]) > 3:
                spline_order = 3
            else:
                spline_order = 1
            eti_spl = UnivariateSpline(
                eti["knot_points"], eti["eti_values"], k=spline_order, s=0
            )
            self.eti = eti_spl(self.z)
        # compute marginal tax rate schedule
        self.mtr, self.mtr_prime = self.compute_mtr_dist(
            data, weight_var, income_measure, mtr_smoother, mtr_smooth_param
        )
        # compute theta_z, the elasticity of the tax base
        self.theta_z = 1 + ((self.z * self.f_prime) / self.f)
        # compute the social welfare weights
        self.g_z, self.g_z_numerical = self.sw_weights()

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
            "F": self.F,
            "f_prime": self.f_prime,
            "mtr": self.mtr,
            "mtr_prime": self.mtr_prime,
            "theta_z": self.theta_z,
            "g_z": self.g_z,
            "g_z_numerical": self.g_z_numerical,
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

        if mtr_smoother == "kreg":
            bins = 1000  # number of equal-width bins
            data.loc[:, ["z_bin"]] = pd.cut(
                data[income_measure], bins, include_lowest=True
            )
            binned_data = pd.DataFrame(
                data[["mtr", income_measure, "z_bin", weight_var]]
                .groupby(["z_bin"], observed=False)
                .apply(lambda x: wm(x[["mtr", income_measure]], x[weight_var]))
            )
            # make column 0 into two columns
            binned_data[["mtr", income_measure]] = pd.DataFrame(
                binned_data[0].tolist(), index=binned_data.index
            )
            binned_data.drop(columns=0, inplace=True)
            binned_data.reset_index(inplace=True)
            mtr_function = KernelReg(
                binned_data["mtr"].dropna(),
                binned_data[income_measure].dropna(),
                var_type="c",
                reg_type="ll",
                bw=[mtr_smooth_param * 40_000],
            )
            mtr, _ = mtr_function.fit(self.z)
            mtr_prime = np.gradient(mtr, edge_order=2)
        elif mtr_smoother == "HSV":
            # estimate the HSV function on mtrs via weighted least squares
            # DATA CLEANING
            # drop rows with missing or inf mtr
            data = data[~data["mtr"].isna()]
            data = data[~data["mtr"].isin([np.inf, -np.inf])]
            # drop if MTR > 100%
            data = data[data["mtr"] < 1]
            # drop rows with missing, inf, or zero income
            data = data[data[income_measure] > 0]
            # drop rows with missing, inf or negative weights
            data = data[~data[weight_var].isna()]
            data = data[~data[weight_var].isin([np.inf, -np.inf])]
            data = data[data[weight_var] > 0]
            # ESTIMATION
            X = np.log(data[income_measure].values)
            X = np.column_stack((np.ones(len(X)), X))
            w = np.array(data[weight_var].values)
            w_sqrt = np.sqrt(w)
            y = np.log(1 - data["mtr"].values)
            X_weighted = X * w_sqrt[:, np.newaxis]
            y_weighted = y * w_sqrt
            coef, _, _, _ = lstsq(X_weighted, y_weighted)
            tau = -coef[1]
            lambda_param = np.exp(coef[0]) / (1 - tau)
            mtr = 1 - lambda_param * (1 - tau) * self.z ** (-tau)
            mtr_prime = lambda_param * tau * (1 - tau) * self.z ** (-tau - 1)
        else:
            print("Please enter a value mtr_smoother method")
            assert False

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
        z_line = np.linspace(100, 1000000, 100000)
        # drop zero income observations
        data = data[data[income_measure] > 0]
        if dist_type == "log_normal":
            mu = (
                np.log(data[income_measure]) * data[weight_var]
            ).sum() / data[weight_var].sum()
            sigmasq = (
                (
                    ((np.log(data[income_measure]) - mu) ** 2)
                    * data[weight_var]
                ).values
                / data[weight_var].sum()
            ).sum()
            # F = st.lognorm.cdf(z_line, s=(sigmasq) ** 0.5, scale=np.exp(mu))
            # f = st.lognorm.pdf(z_line, s=(sigmasq) ** 0.5, scale=np.exp(mu))
            # f = f / np.sum(f)
            # f_prime = np.gradient(f, edge_order=2)

            # analytical derivative of lognormal
            sigma = np.sqrt(sigmasq)
            F = (1 / 2) * (
                1
                + scipy.special.erf(
                    (np.log(z_line) - mu) / (np.sqrt(2) * sigma)
                )
            )
            f = (
                (1 / (sigma * np.sqrt(2 * np.pi)))
                * np.exp(-((np.log(z_line) - mu) ** 2) / (2 * sigma**2))
                * (1 / z_line)
            )
            f_prime = (
                -1
                * np.exp(-((np.log(z_line) - mu) ** 2) / (2 * sigma**2))
                * (
                    (np.log(z_line) + sigma**2 - mu)
                    / (z_line**2 * sigma**3 * np.sqrt(2 * np.pi))
                )
            )
        elif dist_type == "kde":
            # uses the original full data for kde estimation
            f_function = st.gaussian_kde(
                data[income_measure],
                # bw_method=kde_bw,
                weights=data[weight_var],
            )
            f = f_function.pdf(z_line)
            F = np.cumsum(f)
            f_prime = np.gradient(f, edge_order=2)
        elif dist_type == "Pln":

            def pln_pdf(y, mu, sigma, alpha):
                x1 = alpha * sigma - (np.log(y) - mu) / sigma
                phi = st.norm.pdf((np.log(y) - mu) / sigma)
                R = (1 - st.norm.cdf(x1)) / (st.norm.pdf(x1) + 1e-15)
                # 1e-15 to avoid division by zero
                pdf = alpha / y * phi * R
                return pdf

            def neg_weighted_log_likelihood(params, data, weights):
                mu, sigma, alpha = params
                likelihood = np.sum(
                    weights * np.log(pln_pdf(data, mu, sigma, alpha) + 1e-15)
                )
                # 1e-15 to avoid log(0)
                return -likelihood

            def fit_pln(data, weights, initial_guess):
                bounds = [(None, None), (0.01, None), (0.01, None)]
                result = scipy.optimize.minimize(
                    neg_weighted_log_likelihood,
                    initial_guess,
                    args=(data, weights),
                    method="L-BFGS-B",
                    bounds=bounds,
                )
                return result.x

            mu_initial = (
                np.log(data[income_measure]) * data[weight_var]
            ).sum() / data[weight_var].sum()
            sigmasq = (
                (
                    ((np.log(data[income_measure]) - mu_initial) ** 2)
                    * data[weight_var]
                ).values
                / data[weight_var].sum()
            ).sum()
            sigma_initial = np.sqrt(sigmasq)
            # Initial guess for m, sigma, alpha
            initial_guess = np.array([mu_initial, sigma_initial, 1.5])
            mu, sigma, alpha = fit_pln(
                data[income_measure], data[weight_var], initial_guess
            )

            def pln_cdf(y, mu, sigma, alpha):
                x1 = alpha * sigma - (np.log(y) - mu) / sigma
                R = (1 - st.norm.cdf(x1)) / (st.norm.pdf(x1) + 1e-12)
                CDF = (
                    st.norm.cdf((np.log(y) - mu) / sigma)
                    - st.norm.pdf((np.log(y) - mu) / sigma) * R
                )
                return CDF

            def pln_dpdf(y, mu, sigma, alpha):
                x = (np.log(y) - mu) / sigma
                R = (1 - st.norm.cdf(alpha * sigma - x)) / (
                    st.norm.pdf(alpha * sigma - x) + 1e-15
                )
                left = (1 + x / sigma) * pln_pdf(y, mu, sigma, alpha)
                right = (
                    alpha
                    * st.norm.pdf(x)
                    * ((alpha * sigma - x) * R - 1)
                    / (sigma * y)
                )
                return -(left + right) / y

            f = pln_pdf(z_line, mu, sigma, alpha)
            F = pln_cdf(z_line, mu, sigma, alpha)
            f_prime = pln_dpdf(z_line, mu, sigma, alpha)
        else:
            print("Please enter a valid value for dist_type")
            assert False

        return z_line, F, f, f_prime

    def sw_weights(self):
        r"""
        Returns the social welfare weights for a given tax policy.

        See Jacobs, Jongen, and Zoutman (2017) and
        Lockwood and Weinzierl (2016) for details.

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
            + ((self.theta_z * self.eti * self.mtr) / (1 - self.mtr))
            + ((self.eti * self.z * self.mtr_prime) / (1 - self.mtr) ** 2)
        )
        integral = np.trapz(g_z * self.f, self.z)
        g_z = g_z / integral

        # use Lockwood and Weinzierl formula, which should be equivalent but using numerical differentiation
        bracket_term = (
            1
            - self.F
            - (self.mtr / (1 - self.mtr)) * self.eti * self.z * self.f
        )
        d_dz_bracket = np.gradient(bracket_term, edge_order=2)
        # d_dz_bracket = np.diff(bracket_term) / np.diff(self.z)
        # d_dz_bracket = np.append(d_dz_bracket, d_dz_bracket[-1])
        g_z_numerical = -(1 / self.f) * d_dz_bracket
        integral = np.trapz(g_z_numerical * self.f, self.z)
        g_z_numerical = g_z_numerical / integral

        return g_z, g_z_numerical


def find_eti(iot1, iot2, g_z_type="g_z"):
    """
    This function solves for the ETI that would result in the
    policy represented via MTRs in iot2 be consistent with the
    social welfare function inferred from the policies of iot1.

    .. math::
            \varepsilon_{z} = \frac{(1-T'(z))}{T'(z)}\frac{(1-F(z))}{zf(z)}\int_{z}^{\infty}\frac{1-g_{\tilde{z}}{1-F(y)}dF(\tilde{z})

    Args:
        iot1 (IOT): IOT class instance representing baseline policy
        iot2 (IOT): IOT class instance representing reform policy
        g_z_type (str): type of social welfare function to use
            Options are:
            * 'g_z' for the analytical formula
            * 'g_z_numerical' for the numerical approximation

    Returns:
        eti_beliefs (array-like): vector of ETI beliefs over z
    """
    if g_z_type == "g_z":
        g_z = iot1.g_z
    else:
        g_z = iot1.g_z_numerical
    # The equation below is a simplication of the above to make the integration easier
    eti_beliefs_lw = ((1 - iot2.mtr) / (iot2.z * iot2.f * iot2.mtr)) * (
        1 - iot2.F - (g_z.sum() - np.cumsum(g_z))
    )
    # derivation from JJZ analytical solution that doesn't involved integration
    eti_beliefs_jjz = (g_z - 1) / (
        (iot2.theta_z * (iot2.mtr / (1 - iot2.mtr)))
        + (iot2.z * (iot2.mtr_prime / (1 - iot2.mtr) ** 2))
    )

    return eti_beliefs_lw, eti_beliefs_jjz


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
        return np.average(value, weights=weight, axis=0)
    except ZeroDivisionError:
        return [np.nan, np.nan]
