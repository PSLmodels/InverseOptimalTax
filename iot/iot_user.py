from iot.inverse_optimal_tax import IOT
from iot.generate_data import gen_microdata
from iot.constants import CURRENT_YEAR, OUTPUT_LABELS

# import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd


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
        years=[CURRENT_YEAR],
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
        kde_bw=None,
        mtr_smoother="spline",
        mtr_smooth_param=4,
    ):
        self.income_measure = income_measure
        self.weight_var = weight_var
        self.upper_bound = upper_bound
        df = []
        self.iot = []
        # inititalize list of dataframes and
        # IOT class objects for each polciy
        self.labels = labels

        for i, v in enumerate(policies):
            df.append(
                gen_microdata(
                    year=years[i],
                    data=data,
                    reform=v,
                    mtr_wrt=mtr_wrt,
                    income_measure=income_measure,
                    weight_var=weight_var,
                )
            )
        # create results for current law policy
        if compare_default:
            df.append(
                gen_microdata(
                    data=data,
                    year=CURRENT_YEAR,
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
                    kde_bw=kde_bw,
                    mtr_smoother=mtr_smoother,
                    mtr_smooth_param=mtr_smooth_param
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
                OUTPUT_LABELS[self.income_measure]
                + "=%{x:$,.2f}<br>"
                + OUTPUT_LABELS[var]
                + "=%{y:.3f}<extra></extra>"
            )
        else:
            y = []
            for i in self.iot:
                y.append(i.df()[var])
            fig = px.line(x=self.iot[0].df().z, y=y)
            for j in enumerate(self.labels):
                fig.data[j[0]].name = j[1]
                fig.data[j[0]].hovertemplate = (
                    "Policy="
                    + j[1]
                    + "<br>"
                    + OUTPUT_LABELS[self.income_measure]
                    + "=%{x:$,.2f}<br>"
                    + OUTPUT_LABELS[var]
                    + "=%{y:.3f}<extra></extra>"
                )
            fig.update_layout(legend_title="Policy")
        fig.update_layout(
            xaxis_title=OUTPUT_LABELS[self.income_measure],
            yaxis_title=OUTPUT_LABELS[var],
        )
        return fig

    def SaezFig2(self, DS2011=False, upper_bound=None):
        df = self.iot[0].data_original.copy()
        df.sort_values(by=[self.income_measure], inplace=True)
        df["zm"] = (
            sum(df[self.income_measure] * df[self.weight_var])
            - np.cumsum(df[self.income_measure] * df[self.weight_var])
        ) / (sum(df[self.weight_var]) - np.cumsum(df[self.weight_var]))
        if DS2011:  # Diamond and Saez (2011)
            df["y_var"] = df.zm / (df.zm - df[self.income_measure])
            lower_bound = 0
            y_string = r"$z_m / z$"
        else:  # Saez (2001)
            df["y_var"] = df.zm / df[self.income_measure]
            lower_bound = 10000
            y_string = r"$z_m / z$"
        if upper_bound is None:
            upper_bound = self.upper_bound
        df.drop(df[df[self.income_measure] < lower_bound].index, inplace=True)
        df.drop(df[df[self.income_measure] > upper_bound].index, inplace=True)
        fig = px.line(x=df[self.income_measure], y=df.y_var)
        fig.data[0].hovertemplate = (
            OUTPUT_LABELS[self.income_measure]
            + "=%{x:$,.2f}<br>"
            + y_string
            + "=%{y:.3f}<extra></extra>"
        )
        fig.update_layout(
            xaxis_title=OUTPUT_LABELS[self.income_measure],
            yaxis_title=y_string,
        )
        return fig

    def JJZFig4(self, policy="Current Law"):
        k = self.labels.index(policy)
        df = self.iot[k].df()
        # g1 with mtr_prime = 0
        g1 = (
            0
            + ((df.theta_z * self.iot[k].inc_elast * df.mtr) / (1 - df.mtr))
            + ((self.iot[k].inc_elast * df.z * 0) / (1 - df.mtr) ** 2)
        )
        # g2 with theta_z = 0
        g2 = (
            0
            + ((0 * self.iot[k].inc_elast * df.mtr) / (1 - df.mtr))
            + (
                (self.iot[k].inc_elast * df.z * df.mtr_prime)
                / (1 - df.mtr) ** 2
            )
        )
        plot_df = pd.DataFrame(
            {
                self.income_measure: df.z,
                "Overall weight": df.g_z,
                "Tax Base Elasticity": df.g_z - g1,
                "Nonconstant MTRs": df.g_z - g1 - g2,
            }
        )
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=plot_df[self.income_measure],
            y=plot_df["Overall weight"],
            fill=None,
            mode='lines',
            name='Overall weight'
            ))
        fig.add_trace(go.Scatter(
            x=plot_df[self.income_measure],
            y=plot_df["Tax Base Elasticity"],
            fill='tonexty',  # fill area between trace0 and trace1
            mode='lines',
            name='Tax Base Elasticity'))
        fig.add_trace(go.Scatter(
            x=plot_df[self.income_measure],
            y=plot_df["Nonconstant MTRs"],
            fill='tonexty',  # fill area between trace1 and trace2
            mode='lines',
            name='Nonconstant MTRs'))
        fig.update_layout(
            xaxis_title=OUTPUT_LABELS[self.income_measure],
            yaxis_title=r"$g_z$",
        )
        return fig
