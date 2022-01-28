import taxcalc
import pandas as pd
import numpy as np
import copy
import pytest
from iot.inverse_optimal_tax import IOT


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
    df = calc.dataframe(["s006", "expanded_income", "XTOT", "combined"])
    (_, _, mtr1) = calc.mtr(
        "e00200p", calc_all_already_called=True, wrt_full_compensation=False
    )
    df["mtr"] = mtr1
    # create instance of IOT object
    iot1 = IOT(df, dist_type="log_normal", mtr_smoother="cubic_spline")
    # return df from IOT object
    df_out = iot1.df()

    assert isinstance(df_out, pd.DataFrame)


# def test_IOT_compute_mtr_dist():
#     """
#     Test computation of the mtr distribution
#     """
#     pol = taxcalc.policy.Policy()
#     rec = taxcalc.records.Records.cps_constructor()
#     calc = taxcalc.calculator.Calculator(policy=pol, records=rec)
#     calc.advance_to_year(2022)
#     calc.calc_all()
#     df = calc.dataframe(["s006", "expanded_income", "XTOT", "combined"])
#     (_, _, mtr1) = calc.mtr(
#         "e00200p", calc_all_already_called=True, wrt_full_compensation=False
#     )
#     df["mtr"] = mtr1
#     # create instance of IOT object
#     mtr_smoother = "cubic_spline"
#     weight_var = "s006"
#     iot1 = IOT(df, dist_type="log_normal", mtr_smoother=mtr_smoother)
#     mtr, mtr_prime = iot1.compute_mtr_dist(df, weight_var, mtr_smoother)

#     assert np.allclose(mtr, expected_mtr)


# def test_IOT_compute_income_dist():
#     """
#     Test the computation of the income distribution
#     """


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
mtr_smoother = "cubic_spline"
weight_var = "s006"
iot1 = IOT(df, dist_type="log_normal", mtr_smoother=mtr_smoother)
iot2 = copy.deepcopy(iot1)
iot2.theta_z = np.array([1.7, 2.4, 99.0, 1.5, 1.5, 1.5])
iot2.inc_elast = np.array([0.3, 0.1, 0.0, 0.4, 0.4, 0.4])
iot2.mtr = np.array([0.25, 0.2, 0.25, 0.25, 0.25, 0.0])
iot2.z = np.array([5000.0, 5000.0, 5000.0, 5000.0, 300.0, 300.0])
iot2.mtr_prime = np.array([0.03, 0.03, 0.03, 0.0, 0.0, 0.0])
expected_g_z = np.array([81.17, 24.4975, 1.0, 1.2, 1.2, 1.0])


@pytest.mark.parametrize('iot,expected_g_z', [(iot2, expected_g_z)], ids=['Test array'])
def test_IOT_sw_weights(iot, expected_g_z):
    """
    Test computation of the social welfare weights
    """
    g_z = iot.sw_weights()
    assert np.allclose(g_z, expected_g_z)
