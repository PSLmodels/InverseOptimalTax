from email.mime import base
import taxcalc as tc
import numpy as np
import pandas as pd


def gen_microdata(
    year=2022,
    data="CPS",
    baseline_policy=None,
    reform={},
    mtr_wrt="e00200p",
    income_measure="expanded_income",
    weight_var="s006",
):
    """
    Uses taxcalc to generate microdata used for social welfare analysis.

    Args:
        year (int): year for analysis, see
            taxcalc.Calculator.advance_to_year
        data (str): 'CPS' for Current Population Survey or
            'PUF' for IRS Public Use File (must have puf.csv in cd)
        baseline_policy (Tax-Calculator Policy Object): baseline policy
            upon which to layer reform
        reform (dict or str): a dictionary which specifies
            policy changes or a string which points to a JSON file
        mtr_wrt (str): specifies variable with which to calculate
            marginal tax rates
        income_measure (str): specifies income measure to use
        weight_var (str): specifies variable to use for weighting

    Returns:
        df (Pandas DataFrame): microdata generated by taxcalc
            including  weight_var, income_measure, 'XTOT', 'combined'
            and 'mtr' for each individual
    """
    if data == "CPS":
        recs = tc.Records.cps_constructor()
    elif data == "PUF":
        try:
            # looks for 'puf.csv' in cd
            recs = tc.Records()
        except:
            print("PUF data not found")
            return
    elif data is None:
        income_vec = np.linspace(0, 1000000, 10000)
        wgts = np.ones_like(income_vec)
        MARS = np.ones_like(income_vec)
        df = pd.DataFrame({"e00200p": income_vec, "s006": wgts, "MARS": MARS})
        df.reset_index(inplace=True)
        df.rename(columns={"index": "REDID"}, inplace=True)
        df["e00200s"] = 0
        df["e00200p"] = df["e00200p"] + df["e00200s"]
        recs = tc.Records(df)
    if baseline_policy:
        print("Using baseline passed in")
        pol1 = baseline_policy
    else:
        pol1 = tc.Policy()

    if isinstance(reform, str):
        reform = tc.Policy.read_json_reform(reform)

    pol1.implement_reform(reform, print_warnings=False, raise_errors=False)

    print('RRC phaseout = ', dict(pol1.items())['RRC_ps'])

    calc1 = tc.Calculator(policy=pol1, records=recs)
    calc1.advance_to_year(year)
    calc1.calc_all()

    df = calc1.dataframe([weight_var, income_measure, "XTOT", "combined"])

    (_, _, mtr1) = calc1.mtr(
        mtr_wrt, calc_all_already_called=True, wrt_full_compensation=False
    )
    # use other mtr options? expanded income or other concept?
    df["mtr"] = mtr1
    return df