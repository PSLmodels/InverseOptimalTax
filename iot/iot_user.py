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
        baseline_policies (Tax-Calculator Policy object): baseline
            policy upon which reform policies are layered
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
        iot_comparison class instance
    """

    def __init__(
        self,
        years=[CURRENT_YEAR],
        baseline_policies=[None],
        policies=[],
        labels=[],
        data="CPS",
        compare_default=False,
        mtr_wrt="e00200p",
        income_measure="e00200",
        weight_var="s006",
        eti=0.25,
        bandwidth=1000,
        lower_bound=0,
        upper_bound=500000,
        dist_type="log_normal",
        kde_bw=None,
        mtr_smoother="kreg",
        mtr_smooth_param=1000,
        kreg_bw=[120_000],
    ):
        self.income_measure = income_measure
        self.weight_var = weight_var
        self.upper_bound = upper_bound
        df = []
        self.iot = []
        # inititalize list of dataframes and
        # IOT class objects for each policy
        self.labels = labels
        for i, v in enumerate(policies):
            df.append(
                gen_microdata(
                    year=years[i],
                    data=data,
                    baseline_policy=baseline_policies[i],
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
                    eti=eti,
                    # bandwidth=bandwidth,
                    # lower_bound=lower_bound,
                    # upper_bound=upper_bound,
                    dist_type=dist_type,
                    kde_bw=kde_bw,
                    mtr_smoother=mtr_smoother,
                    mtr_smooth_param=mtr_smooth_param,
                    kreg_bw=kreg_bw,
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
            if var == "g_z_numerical":
                start_idx = 10  # numerical approximation not great near 0
            else:
                start_idx = 0
            y = []
            for i in self.iot:
                y.append(i.df()[var][start_idx:])
            fig = px.line(x=self.iot[0].df().z[start_idx:], y=y)
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

    def JJZFig4(self, policy="Current Law", var="g_z", upper_bound=500_000):
        """
        Function to plot a decomposition of the political weights, `g_z`

        Args:
            policy (str): policy to plot
            var (str): variable to plot against income
                Variable options are:
                 * 'g_z' for analytically derived weights
                 * 'g_z_numeric' for numerically derived weights

        Returns:
            fig (plotly.express figure): figure with the decomposition
        """
        k = self.labels.index(policy)
        df = self.iot[k].df()
        if var == "g_z":
            g_weights = df.g_z
        else:
            g_weights = df.g_z_numerical

        # g1 with mtr_prime = 0
        g1 = (
            ((df.theta_z * self.iot[k].eti * df.mtr) / (1 - df.mtr))
            + ((self.iot[k].eti * df.z * 0) / (1 - df.mtr) ** 2)
        )
        # g2 with theta_z = 0
        g2 = (
            ((0 * self.iot[k].eti * df.mtr) / (1 - df.mtr))
            + ((self.iot[k].eti * df.z * df.mtr_prime) / (1 - df.mtr) ** 2)
        )
        integral = np.trapz(g1, df.z)
        # g1 = g1 / integral
        integral = np.trapz(g2, df.z)
        # g2 = g2 / integral
        plot_df = pd.DataFrame(
            {
                self.income_measure: df.z,
                "Overall Weight": g_weights,
                "Tax Base Elasticity": 1 + g1,
                "Nonconstant MTRs": 1 + g1 + g2 + np.abs(g1) * (np.sign(g1) != np.sign(g2))
            }
        )

        fig = go.Figure()
        # add a line at y = 1
        fig.add_trace(
            go.Scatter(
                x=[
                    plot_df[self.income_measure].min(),
                    plot_df[self.income_measure].max(),
                ],
                y=[1, 1],
                mode="lines",
                line=dict(color="black", width=1, dash="dash"),
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=plot_df[self.income_measure],
                y=plot_df["Tax Base Elasticity"],
                fill="tonexty",  # fill area from prior trace to this one
                # fill="tozeroy",
                mode="lines",
                fillcolor="rgba(4,40,145,0.5)",
                name="Tax Base Elasticity",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[
                    plot_df[self.income_measure].min(),
                    plot_df[self.income_measure].max(),
                ],
                y=[1, 1],
                mode="lines",
                line=dict(color="black", width=1, dash="dash"),
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=plot_df[self.income_measure],
                y=plot_df["Nonconstant MTRs"],
                fill="tonexty",  # fill area from prior trace to this one
                mode="lines",
                fillcolor="rgba(229,0,0,0.5)",
                name="Nonconstant MTRs",
            )
        )
        # Add black line for overall weight
        fig.add_trace(
            go.Scatter(
                x=plot_df[self.income_measure],
                y=plot_df["Overall Weight"],
                mode="lines",
                line=dict(color="black", width=2, dash="solid"),
                name="Overall Weight",
                showlegend=True,
            )
        )
        # # add a line at y=0
        # fig.add_trace(
        #     go.Scatter(
        #         x=[
        #             plot_df[self.income_measure].min(),
        #             plot_df[self.income_measure].max(),
        #         ],
        #         y=[0, 0],
        #         mode="lines",
        #         line=dict(color="black", width=1, dash="dash"),
        #     )
        # )
        fig.update_layout(
            xaxis_title=OUTPUT_LABELS[self.income_measure],
            yaxis_title=r"$g_z$",
        )
        fig.update_xaxes(range=[0, upper_bound])
        return fig
