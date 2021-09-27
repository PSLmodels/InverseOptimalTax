(Chap_Params)=
# Parameters
Equation 10 of {cite}JJZ:2017 summarizes the equation to determine the social welfare weights.  The social welfare weight on a taxpayer with income $z$ is:

$$
g_z = 1 + \theta_z \varepsilon^c \frac{T'(z)}{(1-T'(z))} + \varepsilon^c \frac{zT''(z)}{(1-T'(z))^2}
$$

The parameters/functions that are necessary for computing these social welfare weights are summarized below:

| Parameter/Function | Description | Source |
| ------------------ | ----------- | ------ |
| $\varepsilon^c$      | Compensated elasticity of taxable income       |  Economics literature |
| $\theta_z$   | Elasticity of local tax base w.r.t. income $z$, $\theta_{z}\equiv 1 + \frac{zf'(z)}{f(z)}$        |  See JJZ (2017), will need Tax-Calculator + estimates from economics literature |
| $T'(z)$ | Marginal tax rate for each taxpayer | Tax-Calculator |
| $T''(z)$ | Derivative of the MTR for each taxpayer | Tax-Calculator (but will need to write a new function) |
| $z$ | Pre-tax earnings of each taxpayer | Tax-Calculator/TaxData |
| $f(z)$ | pdf of the distribution of earnings | Tax-Calculator/TaxData |
| $f'(z)$ | derivative of the pdf of earnings | Tax-Calculator/TaxData |


Some notes:
* To start, maybe take a "sufficient statistics" approach
  * Keep elasticities, the income distribution, and employment constant ({cite}JJZ:2017, pg. 86)
  * Otherwise, may need a lot more structure on the model  (see {cite}Hendren:2014 and {cite}LW:2016?)
  * Do we need to smooth out MTRs?
    * Maybe just some average effective rate for each point in the income distribution

