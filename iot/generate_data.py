import taxcalc as tc


def gen_microdata(
    year=2022,
    data="CPS",
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
        reform (dict or str): a dictionary which specifies
            policy changes or a string which points to a JSON file
        mtr_wrt (str): specifies variable with which to calculate
            marginal tax rates

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
    pol1 = tc.Policy()

    if isinstance(reform, str):
        reform = tc.Policy.read_json_reform(reform)

    pol1.implement_reform(reform, print_warnings=False, raise_errors=False)

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
