import taxcalc
import pandas as pd
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
