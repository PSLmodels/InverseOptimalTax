# %%
import taxcalc
from inverse_optimal_tax import IOT
import plotly.io as pio
import plotly.express as px

pio.templates.default = "plotly_white"


# %%
pol = taxcalc.policy.Policy()
rec = taxcalc.records.Records.cps_constructor()
calc = taxcalc.calculator.Calculator(policy=pol, records=rec)

# %%
calc.advance_to_year(2022)
calc.calc_all()

# %%
df = calc.dataframe(["s006", "expanded_income", "XTOT", "combined"])
(_, _, mtr1) = calc.mtr(
    "e00200p", calc_all_already_called=True, wrt_full_compensation=False
)
df["mtr"] = mtr1

# %%
iot1 = IOT(df, dist_type="log_normal", mtr_smoother="cubic_spline")

df_out = iot1.df()

# plot
fig = px.line(df_out, x="z", y="g_z")
fig.show()

fig = px.line(df_out, x="z", y="mtr")
fig.show()

fig = px.line(df_out, x="z", y="theta_z")
fig.show()

fig = px.line(df_out, x="z", y="f")
fig.show()

fig = px.line(df_out, x="z", y="f_prime")
fig.show()
# %%
fig = px.line(df_out, x="z", y="mtr_prime")
fig.show()
