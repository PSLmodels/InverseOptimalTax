# %%
from iot_user import iot_comparison

# %%
iot1 = iot_comparison(
    policies=[
        "https://raw.githubusercontent.com/PSLmodels/examples/main/psl_examples/taxcalc/2017_law.json",
        "https://raw.githubusercontent.com/PSLmodels/examples/main/psl_examples/taxcalc/Biden2020.json",
    ],
    labels=["2017 Law", "Biden 2020"],
    years=[2017, 2020]
)

iot2 = iot_comparison(
    policies=[
        "https://raw.githubusercontent.com/PSLmodels/examples/main/psl_examples/taxcalc/2017_law.json",
        "https://raw.githubusercontent.com/PSLmodels/examples/main/psl_examples/taxcalc/Biden2020.json",
    ],
    labels=["2017 Law", "Biden 2020"],
    years=[2017, 2020],
    inc_elast=2,
)

# %%
iot1.iot[-1].df().head()

# %%
gzplot1 = iot1.plot()
gzplot2 = iot2.plot()
fplot = iot1.plot(var="f")
mtrplot1 = iot1.plot(var="mtr")
thetaplot1 = iot1.plot(var="theta_z")

# %%
fplot.show()
mtrplot1.show()
thetaplot1.show()
gzplot1.show()
gzplot2.show()
