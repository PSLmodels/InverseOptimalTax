# Model

Based Saez (2001), Diamond (1998), Saez, Slemrod, Giertz (2012), lecture notes by Rishabh Kirpalani.

We derive the Diamond-Saez-Mirrlees optimal tax formula, and invert it following Lockwood and Weinzierl (2016) and Jacobs, Jongen and Zoutman (2017)

## Environment

Suppose households have an ability type, $\theta \in \Theta$ with distribution $f(\theta)$, which is private information. The production technology is such that type $\theta$ has production (or taxable income) $z(\theta) = \theta * l(\theta)$. The planner or policy maker can observe income / production $z$ for each type, but not labor $l$. The government has exogenous expenditures $E$, which must be funded by labor of households or (in the decentralized problem) taxes on labor.

 An allocation is a set of consumption and output $\{ c(\theta), z(\theta) \}_{\forall \theta \in \Theta}$. The household has preferences given by 

 $$ U(\theta) = u(c(\theta)) - v(l(\theta)) = u(c(\theta)) - v\left (\frac{z(\theta)}{\theta}\right ),$$

where $u', v' > 0$, and $u'' \leq 0 \leq v''$. The marginal rate of substitution for the household between $c$ and $z$ is 

 $$MRS_{c, z} = -\theta\frac{u'(c)}{v'(\frac{z}{\theta})}.$$


Now, note that 

$$\frac{\partial}{\partial \theta} MRS_{c, z} = -u'(c) \left[\frac{1}{v'(\frac{z}{\theta})}+ \frac{zv''(\frac{z}{\theta})}{\theta v'(\frac{z}{\theta})}\right] < 0 $$

by assumptions on $u$ and $v$. Thus, the single crossing property holds. 
## Incentive Compatibility

Since labor effort and productivity type are unobservable, the planner or policy maker cannot achieve the first best outcome in which policy can be a function of ability directly. By the revelation principle, a decentralized equilibrium in which agents truthfully reveal their type is equivalent to planner's problem with incentive compatibility constraints. Thus, policy makers must design the tax system to ensure that agents truthfully reveal their productivity through their income choice. An allocation is incentive compatible (globally) if

$$ u(c(\theta)) - v\left (\frac{z(\theta)}{\theta}\right ) > c(\hat \theta) - v\left (\frac{z(\hat \theta)}{\theta}\right ), \forall \theta, \hat{\theta}.$$

However, it can be shown that this global incentive compatibility constraint can be replaced by the following conditions for local incentive compatibility:

1. $U'(\theta) = \frac{z(\theta)}{\theta^2}v'\left (\frac{z(\theta)}{\theta}\right )$
2. $z(\theta)$ increasing

Mechanism design literature drops condition 2, to be verified ex-post.

## Constrained Social Planner's Problem

The constrained social planner wants to maximize welfare subject to incentive compatibility and resource constraints. The constrained social planner's problem is:

$$\begin{align*}
\max_{c(\theta),z(\theta)} & \int_{\Theta} G(U(\theta)) dF(\theta) \\
\text{subject to} & \\
& \int_{\Theta} (z(\theta) - c(\theta)) dF(\theta) \geq E \quad (\text{RC}) \\
& U'(\theta) = \frac{z(\theta)}{\theta^2} v'\left(\frac{z(\theta)}{\theta}\right) \quad \forall \theta \in \Theta \quad (\text{LIC}) \\
& U(\theta) = u(c(\theta)) - v\left(\frac{z(\theta)}{\theta}\right) \quad \forall \theta \in \Theta \\
& z(\theta) \text{  increasing in } \theta
\end{align*}$$

Where G is a weighting function for the social planner which determines their redistributive preferences. We will drop the monotonicity condition to be verified later. The Lagrangian for the social planner's problem is:

$$
\begin{align*}
L =& \int_{\Theta} G(U(\theta)) + \lambda[z(\theta) - c(\theta) - E] dF(\theta) \\
& + \int_{\Theta} \gamma(\theta) \left [u(c(\theta)) - v\left(\frac{y(\theta)}{\theta}\right) - U(\theta)\right ] + \mu(\theta)\left [U'(\theta)- \frac{z(\theta)}{\theta^2} v'\left(\frac{z(\theta)}{\theta}\right)\right]d\theta\\
\end{align*}
$$


Where $\lambda, \gamma(\theta), \mu(\theta)$ are Lagrange multipliers. There is an equivalent social planner's problem in which the planner has an exogenous value of raising public funds; another interpretation of $\lambda$ is the marginal value of public funds. 


Using integration by parts on $\int_{\Theta} \mu(\theta)U'(\theta)d\theta$, we have:

$$
\begin{align*}
L =& \int_{\Theta} G(U(\theta)) + \lambda[z(\theta) - c(\theta)- E] dF(\theta) \\
& + \int_{\Theta} \gamma(\theta) \left [u(c(\theta)) - v\left(\frac{y(\theta)}{\theta}\right) - U(\theta)\right ] - \mu(\theta)\left [\frac{z(\theta)}{\theta^2} v'\left(\frac{z(\theta)}{\theta}\right)\right]d\theta\\ \\
&+ \mu(\bar{\theta}) U(\bar{\theta}) - \mu(\underline{\theta}) U(\underline{\theta})
\end{align*}
$$

It must be the case that $\mu(\underline\theta) = \mu(\bar\theta) = 0$. Otherwise, the planner would like to set $U(\barθ)  = ∞$ $(U (\underlineθ) = −∞)$ which would violate incentive constraints. 

Taking first order conditions:

$$
\begin{align*}
U(\theta): \quad & G'(U(\theta)) f(\theta) - \gamma(\theta) - \mu'(\theta) = 0 \\
c(\theta): \quad & \gamma(\theta) u'(c(\theta)) - \lambda f(\theta) = 0 \\  
z(\theta): \quad & -\gamma(\theta) \frac{1}{\theta} v'\left(\frac{z(\theta)}{\theta}\right) + \lambda f(\theta) - \mu(\theta) \left[\frac{1}{\theta^2} v'\left(\frac{z(\theta)}{\theta}\right) + \frac{z(\theta)}{\theta^3} v''\left(\frac{z(\theta)}{\theta}\right)\right] = 0
\end{align*}
$$

Boundary conditions: $\mu(\bar{\theta}) = \mu(\underline{\theta}) = 0$.

Using the first order conditions for $U(\theta)$ and $c(\theta)$, we have:

$$
\mu(\theta) = \int_\theta^{\bar\theta} \left[\frac{\lambda f(y)}{u'(c(y))} - G'(U(y)) f(y)\right] dy
$$

Substituting this into the first order condition for $z(\theta)$ yields:

$$
\frac{1}{\frac{1}{\theta} v'\left(\frac{z(\theta)}{\theta}\right)} - \frac{1}{u'(c(\theta))} = \frac{1-F(\theta)}{\theta f(\theta)} \left[1 + \frac{z(\theta)}{\theta} \frac{v''\left(\frac{z(\theta)}{\theta}\right)}{v'\left(\frac{z(\theta)}{\theta}\right)}\right] \int_\theta^{\bar{\theta}} \left[\frac{1}{u'(c(y))} - \frac{G'(U(y))}{\lambda}\right] \frac{dF(y)}{1-F(\theta)}
$$

This, togethether with the resource constraint characterizes optimal allocations. 

## The Diamond-Saez-Mirrlees Optimal Tax Formula

The policy maker wants to implement the constrained planner's problem with a tax policy $T(z)$. The household's problem is:

$$\begin{align*}
\max_{c, z} \quad &u(c) - v(\frac{z}{\theta}) \\
\text{subject to} & \\
& c = z - T(z)
\end{align*}$$

Assuming households do not optimize with respect to a highly nonlinear tax code, the first order condtion of the household is $\frac{1}{\theta} v'(\frac{z}{\theta}) = u'(c)(1-T'(z))$. 
Letting $T(z)$ be the tax function that implements the efficient allocation and substituting the household FOC, the formula becomes:

$$
\frac{T'(\theta)}{1-T'(\theta)} = u'(c(\theta)) \frac{1-F(\theta)}{\theta f(\theta)} \left[1 + \frac{z(\theta)}{\theta} \frac{v''\left(\frac{z(\theta)}{\theta}\right)}{v'\left(\frac{z(\theta)}{\theta}\right)}\right] \int_\theta^{\bar{\theta}} \left[\frac{1}{u'(c(y))} - \frac{G'(U(y))}{\lambda}\right] \frac{dF(y)}{1-F(\theta)}
$$

As per Diamond (1998), we take preferences to be GHH, which has the interpretation of no income effects, so that the formula does not depend on both consumption and income simultaneously. This also has the benefit of being able to interpret utility in dollars.

$$U(c, l) = c - \psi (\varepsilon l^{\frac{1}{\varepsilon}})$$

where $\varepsilon$ is the elasticity of taxable income. Thus, $u'(c) = 1$, and $x \frac{v''(x)}{v'(x)} = \frac{1}{\epsilon}-1$. Also note that there is a 1 to 1 mapping between $\theta$ and $z$ under incentive compatibility. Thus, we can replace $\theta$ with $z$ in the formula. However, one should be careful that we now are using $f$ as the virtual density, which makes the assumption that taxes are linearized around $T(z)$. This is fine as long as individuals are not optimizing with respect to a highly nonlinear tax code. Therefore, the formula becomes:


$$
\frac{T'(z)}{1-T'(z)} =  \frac{1-F(z)}{\varepsilon z f(z)}  \int_z^{\bar{z}} \left[1 - \frac{G'(U(y))}{\lambda}\right] \frac{dF(y)}{1-F(z)}
$$


Let $g(z) = G'(U(z))/\lambda$ be the marginal social welfare weight. The interpretation is that $g(z)$ is the social welfare or value to the policy maker from giving an additional dollar of income or consumption to an agent earning $z$. Therefore, we get the formula used in Lockwood and Weinzierl (2015) and Jacobs, Jongen and Zoutman (2017):

$$
\frac{T'(z)}{1-T'(z)} =  \frac{1-F(z)}{\varepsilon z f(z)}  \int_z^{\bar{z}} \left[1 - g(y)\right] \frac{dF(y)}{1-F(z)}
$$


We assume there is an unbounded distribution of income, so $\bar{z} = \infty$. The key components are:

1. A hazard ratio $\frac{1-F(z)}{zf(z)}$. For a thin-tailed distribution such as the lognormal distribution, this converges to 0, giving us the "no distortion at the top" result of a 0 marginal tax rate at the top of the income distribution. For a Pareto distribution, this term is $\alpha$, which represents the thinness of the tail.
2. The elasticity of taxable income $\varepsilon$. This has estimates ranging from .12 to .4, with .25 as a middle of the road estimate. See Saez, Slemrod, Giertz (2012).
3. The planner's redistributionary motives, captured by $g(z)$. Note in general that these are endogenous; they depend on the equilibrium allocation. 


## Inverting the optimal tax formula

As per Lockwood and Weinzierl (2016), we can invert this formula:

$$\bar{g}(z) = 1-F(z) - \frac{\varepsilon z f(z) T'(z)}{1-T'(z)}$$

Where $\bar{g}(z) \equiv \int_z^\infty g(y)dF(y)$. By the fundamental theorem of calculus, $\frac{d}{dz}\bar{g}(z) = - g(z)f(z)$. Thus,

$$
\begin{align*}
g(z) &= -\frac{1}{f(z)}\frac{d}{dz}\left[ 1-F(z) - \varepsilon z f(z)\frac{ T'(z)}{1-T'(z)}\right]\\
&=1 + \frac{1}{f(z)}\frac{d}{dz}\left[\varepsilon z f(z)\frac{ T'(z)}{1-T'(z)}\right] \\
& = 1 + \theta_z \varepsilon \frac{T'(z)}{1-T'(z)} + \varepsilon \frac{zT''(z)}{(1-T'(z))^2}
\end{align*}
$$

Where $\theta_z \equiv 1 + \frac{zf'(z)}{f(z)}$ is the elasticity of the local tax base with respect to income. In the case of constant ETI, we get the formula used by Lockwood & Weinzierl (2016) on the first line, and Jacobs, Jongen and Zoutman (2017) on the 3rd line.

In the case of variable ETI, we get an extra term for the JJZ formula:

$$g(z) = 1 + \theta_z \varepsilon(z) \frac{T'(z)}{1-T'(z)} + \varepsilon(z) \frac{zT''(z)}{(1-T'(z))^2}+ \varepsilon'(z)\frac{zT'(z)}{1-T'(z)}$$

However, note that this is subject to the constraint that 

$$\int_0^\infty g(z) dF(z) = 1$$

To see why this is true, consider a reform in which the government collects an additional dollar from everyone. Since we have GHH preferences and utility is in terms of dollars, the welfare loss from this reform is 

$$ \int_0^\infty G(U(z)) - G(U(z)-1) dF(z) \approx \int_0^\infty G'(U(z))dF(z).$$

The gain to the social planner of collecting this dollar is $\lambda$, the value (or shadow price) of relaxing the government budget constraint. At optimum, the marginal benefit equals the marginal cost, so

$$\int_0^\infty G'(U(z))dF(z) = \lambda$$

$$\implies \int_0^\infty g(z) dF(z) = 1$$

by definition of $g$. 
