# %%
from iot.inverse_optimal_tax import IOT
from iot.generate_data import gen_microdata
# import plotly.io as pio
import plotly.express as px


# %%
class iot_comparison:
    """
    Uses gen_microdata to generate tax data for each policy and
    uses IOT to generate class instances which can be used
    to compare different tax policies for social welfare analysis

    Args:
        year (int): year for analysis, see
            taxcalc.Calculator.advance_to_year
        policies (list): list of dicts or json files denoting policy
            parameter changes
        labels (list): list of string labels for each policy
        data (str): 'CPS' for Current Population Survey or 'PUF'
            for IRS Public Use File (must have 'puf.csv' in cd)
        compare_default(boolean): True to include default policy
            False to not include default policy
        mtr_wtr (str): name of income source to compute MTR on
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
        iot_comparison class instance
    """

    def __init__(
        self,
        year=2022,
        policies=[],
        labels=[],
        data="CPS",
        compare_default=True,
        mtr_wrt="e00200p",
        income_measure="expanded_income",
        weight_var="s006",
        inc_elast=0.25,
        bandwidth=1000,
        lower_bound=0,
        upper_bound=500000,
        dist_type="kde_full",
        mtr_smoother="cubic_spline",
    ):
        df = []
        self.iot = []
        # inititalize list of dataframes and
        # IOT class objects for each polciy
        self.labels = labels

        for v in policies:
            df.append(gen_microdata(year=year, data=data, reform=v))
        # creates dataframes for each policy given as argument
        if compare_default:
            df.append(
                gen_microdata(
                    year=year,
                    mtr_wrt=mtr_wrt,
                    income_measure=income_measure,
                    weight_var=weight_var,
                )
            )
            self.labels.append("Current Law")
            # adds the defaults to the list
        for j in df:
            self.iot.append(
                IOT(
                    j,
                    income_measure=income_measure,
                    weight_var=weight_var,
                    inc_elast=inc_elast,
                    bandwidth=bandwidth,
                    lower_bound=lower_bound,
                    upper_bound=upper_bound,
                    dist_type=dist_type,
                    mtr_smoother=mtr_smoother,
                )
            )

    def plot(self, var="g_z"):
        """
        Used to plot the attributes of the IOT class objects
        for each policy.
        Args:
            var (str): variable to plot against income
                Variable options are:
                 * 'f' for distribution of income
                 * 'f_prime' for approximate derivative of income distribution
                 * 'mtr' for marginal tax rates
                 * 'mtr_prime' for derivative of marginal tax rate
                 * 'theta_z' for elasticity of the tax base
                 * 'g_z' for social welfare weights

        Note:
            `f`, `f_prime`, and `theta_z` are common between all
                policies, and therefore are not comparative.

        Returns:
            fig (plotly.express figure)
        """
        if var in ["f", "f_prime", "theta_z"]:
            fig = px.line(x=self.iot[0].df().z, y=self.iot[0].df()[var])
            fig.data[0].hovertemplate = (
                "z=%{x}<br>" + var + "=%{y}<extra></extra>"
            )
        else:
            y = []
            for i in self.iot:
                y.append(i.df()[var])
            fig = px.line(x=self.iot[0].df().z, y=y)
            for j in range(len(self.labels)):
                fig.data[j].name = self.labels[j]
                fig.data[j].hovertemplate = (
                    "Policy="
                    + self.labels[j]
                    + "<br>z=%{x}<br>"
                    + var
                    + "=%{y}<extra></extra>"
                )
            fig.update_layout(legend_title="Policy")
        fig.update_layout(
            xaxis_title=r"$z$",
            yaxis_title=var,
        )
        return fig
