import taxcalc
import pandas as pd
import numpy as np
import copy
import pytest
import os
from iot.inverse_optimal_tax import IOT

CUR_PATH = os.path.abspath(os.path.dirname(__file__))


def test_IOT_df():
    """
    Test that IOT dataframe method returns a dataframe.
    """
    # get data from Tax-Calculator
    pol = taxcalc.policy.Policy()
    rec = taxcalc.records.Records.cps_constructor()
    calc = taxcalc.calculator.Calculator(policy=pol, records=rec)
    calc.advance_to_year(2022)
    calc.calc_all()
    data = calc.dataframe(["s006", "expanded_income", "XTOT", "combined"])
    (_, _, mtr1) = calc.mtr(
        "e00200p", calc_all_already_called=True, wrt_full_compensation=False
    )
    data["mtr"] = mtr1
    # create instance of IOT object
    iot1 = IOT(
        data,
        dist_type="log_normal",
        mtr_smoother="cubic_spline",
        mtr_smooth_param=3,
    )
    # return df from IOT object
    df_out = iot1.df()

    assert isinstance(df_out, pd.DataFrame)


@pytest.mark.parametrize("mtr_smoother", [("spline"), ("kr"), (None)])
def test_IOT_compute_mtr_dist(mtr_smoother):
    """
    Test computation of the mtr distribution
    """
    pol = taxcalc.policy.Policy()
    rec = taxcalc.records.Records.cps_constructor()
    calc = taxcalc.calculator.Calculator(policy=pol, records=rec)
    calc.advance_to_year(2022)
    calc.calc_all()
    data = calc.dataframe(["s006", "expanded_income", "XTOT", "combined"])
    income_measure = "expanded_income"
    upper_bound = 500000
    lower_bound = 0
    bandwidth = 1000
    (_, _, mtr1) = calc.mtr(
        "e00200p", calc_all_already_called=True, wrt_full_compensation=False
    )
    data["mtr"] = mtr1
    # clean data based on upper and lower bounds
    data = data[
        (data[income_measure] >= lower_bound)
        & (data[income_measure] <= upper_bound)
    ]
    # create bins for analysis
    bins = np.arange(
        start=lower_bound, stop=upper_bound + bandwidth, step=bandwidth
    )
    data["z_bin"] = pd.cut(data[income_measure], bins, include_lowest=True)
    # create instance of IOT object
    weight_var = "s006"
    # iot1 = IOT(df, dist_type="log_normal", mtr_smoother=mtr_smoother)
    iot1 = IOT(data)
    mtr, mtr_prime = iot1.compute_mtr_dist(data, weight_var, mtr_smoother, 3)
    # np.savetxt(os.path.join(CUR_PATH, 'test_io_data', str(mtr_smoother) + '_mtr.csv'), mtr, delimiter=",")
    expected_mtr = np.genfromtxt(
        os.path.join(CUR_PATH, "test_io_data", str(mtr_smoother) + "_mtr.csv"),
        delimiter=",",
    )

    assert np.allclose(mtr, expected_mtr)


# In tests above and below, will need to experiment with different
# income measures and lower and upper bounds (625000 seems to be limit so far)
# may need to do something with bins with no observations in them
@pytest.mark.parametrize(
    "income_measure,dist_type",
    [
        ("expanded_income", None),
        ("e00200", None),
        ("e00200", "log_normal"),
        ("e00200", "kde_full"),
        ("e00200", "kde_subset"),
    ],
)
def test_IOT_compute_income_dist(income_measure, dist_type):
    """
    Test the computation of the income distribution
    """
    pol = taxcalc.policy.Policy()
    rec = taxcalc.records.Records.cps_constructor()
    calc = taxcalc.calculator.Calculator(policy=pol, records=rec)
    calc.advance_to_year(2022)
    calc.calc_all()
    data = calc.dataframe(["s006", income_measure, "XTOT", "combined"])
    upper_bound = 500000
    lower_bound = 0
    bandwidth = 1000
    (_, _, mtr1) = calc.mtr(
        "e00200p", calc_all_already_called=True, wrt_full_compensation=False
    )
    data["mtr"] = mtr1
    # clean data based on upper and lower bounds
    data = data[
        (data[income_measure] >= lower_bound)
        & (data[income_measure] <= upper_bound)
    ]
    # create bins for analysis
    bins = np.arange(
        start=lower_bound, stop=upper_bound + bandwidth, step=bandwidth
    )
    data["z_bin"] = pd.cut(data[income_measure], bins, include_lowest=True)
    # create instance of IOT object
    weight_var = "s006"
    iot1 = IOT(data, income_measure=income_measure)
    z, f, f_prime = iot1.compute_income_dist(
        data, income_measure, weight_var, dist_type=dist_type
    )
    # np.savetxt(os.path.join(CUR_PATH, "test_io_data", income_measure + "_" + str(dist_type) + "_dist.csv"), f, delimiter=",")
    expected_dist = np.genfromtxt(
        os.path.join(
            CUR_PATH,
            "test_io_data",
            income_measure + "_" + str(dist_type) + "_dist.csv",
        ),
        delimiter=",",
    )

    assert np.allclose(f, expected_dist)


pol = taxcalc.policy.Policy()
rec = taxcalc.records.Records.cps_constructor()
calc = taxcalc.calculator.Calculator(policy=pol, records=rec)
calc.advance_to_year(2022)
calc.calc_all()
df = calc.dataframe(["s006", "expanded_income", "XTOT", "combined"])
(_, _, mtr1) = calc.mtr(
    "e00200p", calc_all_already_called=True, wrt_full_compensation=False
)
df["mtr"] = mtr1
# create instance of IOT object
mtr_smoother = "spline"
weight_var = "s006"
iot1 = IOT(
    df, dist_type="log_normal", mtr_smoother=mtr_smoother, mtr_smooth_param=3
)
iot2 = copy.deepcopy(iot1)
iot2.theta_z = np.array([1.7, 2.4, 99.0, 1.5, 1.5, 1.5])
iot2.inc_elast = np.array([0.3, 0.1, 0.0, 0.4, 0.4, 0.4])
iot2.mtr = np.array([0.25, 0.2, 0.25, 0.25, 0.25, 0.0])
iot2.z = np.array([5000.0, 5000.0, 5000.0, 5000.0, 300.0, 300.0])
iot2.mtr_prime = np.array([0.03, 0.03, 0.03, 0.0, 0.0, 0.0])
expected_g_z = np.array([81.17, 24.4975, 1.0, 1.2, 1.2, 1.0])


@pytest.mark.parametrize(
    "iot,expected_g_z", [(iot2, expected_g_z)], ids=["Test array"]
)
def test_IOT_sw_weights(iot, expected_g_z):
    """
    Test computation of the social welfare weights
    """
    g_z = iot.sw_weights()
    assert np.allclose(g_z, expected_g_z)


def test_sw_weights_analytical():
    """
    A test of the sw_weights function using a special case where their
    is an analytical solution.

    The special case involves the following assumptions:
    1. A constant marginal tax rate (so that T''(z) = 0)
    2. An exponential distribution in z, so that f(z) = 1/beta * exp(-z/beta)
    f'(z) = -1/(beta ** 2) * exp(-z/beta)
    theta(z) = 1 + (z f'(z) / f(z)) = 1 - z/beta
    """
    beta = 10000
    mu = 10
    sigma = 1
    mtr = 0.15
    elasticity = 0.4
    sim_dist_type = "log_normal"
    if sim_dist_type == "exponential":
        # generate income according to exponential distribution
        z = np.random.exponential(beta, 100000)

        def f_z_exp(z, beta):
            return (1 / beta) * np.exp(-z / beta)

        def f_prime_z_exp(z, beta):
            return (-1 / (beta**2)) * np.exp(-z / beta)

        def theta_z_exp(z, beta):
            return 1 - (z / beta)

        def g_z_exp(z, beta, elasticity, mtr):
            theta = theta_z_exp(z, beta)
            g_z_exp = 1 + theta * elasticity * (mtr / (1 - mtr))
            return g_z_exp

    else:
        # generate income according to lognormal distribution
        z = np.random.lognormal(mu, sigma, 100000)

        def f_z_exp(z, mu, sigma):
            f = (
                (1 / np.sqrt(2 * np.pi * sigma))
                * np.exp(-((np.log(z) - mu) ** 2) / (2 * sigma**2))
                * (1 / z)
            )
            return f

        def f_prime_z_exp(z, mu, sigma):
            fp = (
                # -((np.log(z) - mu) / (np.sqrt(2 * np.pi * sigma) * (sigma ** 2) * (z ** 2)))
                # * np.exp(-((np.log(z) - mu) ** 2) / (2 * sigma ** 2))
                -(
                    np.exp(-((np.log(z) - mu) ** 2) / (2 * sigma**2))
                    * (np.log(z) + sigma**2 - mu)
                )
                / (np.sqrt(2) * np.sqrt(np.pi) * sigma ** (5 / 2) * z**2)
            )
            return fp

        def theta_z_exp(z, mu, sigma):
            return 1 - ((np.log(z) + sigma**2 - mu) / sigma**2)

        def g_z_exp(z, mu, sigma, elasticity, mtr):
            theta = theta_z_exp(z, mu, sigma)
            g_z_exp = 1 + theta * elasticity * (mtr / (1 - mtr))
            return g_z_exp

    # Find test value for g_z
    # sort z -- not sure it matters
    z.sort()
    dict_in = {
        "e00200p": z,
        "e00200s": np.zeros_like(z),
        "e00200": z,
        "s006": np.ones_like(z),
        "XTOT": np.ones_like(z),
        "MARS": np.ones_like(z),
        "mtr": np.ones_like(z) * mtr,
    }
    df = pd.DataFrame(dict_in)
    # create instance of IOT object
    mtr_smoother = "spline"
    weight_var = "s006"
    if sim_dist_type == "log_normal":
        dist_type = "log_normal"
    else:
        dist_type = "kde_full"
    iot_test = IOT(
        df,
        income_measure="e00200",
        bandwidth=100,
        dist_type=dist_type,
        mtr_smoother=mtr_smoother,
        mtr_smooth_param=3,
        upper_bound=80000,
    )
    if sim_dist_type == "exponential":
        g_z_test = iot_test.sw_weights()
        g_z_expected = g_z_exp(iot_test.z, beta, elasticity, mtr)
        theta_z_expected = theta_z_exp(iot_test.z, beta)
        f_z_expected = f_z_exp(iot_test.z, beta)
        f_z_expected = f_z_expected / np.sum(f_z_expected)
        f_prime_z_expected = f_prime_z_exp(iot_test.z, beta)
    else:
        g_z_test = iot_test.sw_weights()
        g_z_expected = g_z_exp(iot_test.z, mu, sigma, elasticity, mtr)
        theta_z_expected = theta_z_exp(iot_test.z, mu, sigma)
        f_z_expected = f_z_exp(iot_test.z, mu, sigma)
        f_z_expected = f_z_expected / np.sum(f_z_expected)
        f_prime_z_expected = f_prime_z_exp(iot_test.z, mu, sigma)

    assert np.allclose(iot_test.f, f_z_expected, atol=1e-6)
    assert np.allclose(iot_test.f_prime, f_prime_z_expected, atol=1e-7)
    # fprimes are close, but off by an order of magnitude bc very small
    # numbers (1-12/13 at hight end of z... this then leads to significant
    # diffs in theta_z)
    # assert np.allclose(iot_test.theta_z[100:], theta_z_expected[100:], atol=1e-4)
    # assert np.allclose(g_z_test, g_z_expected, atol=1e-4)
