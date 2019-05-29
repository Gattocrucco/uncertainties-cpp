# print first 6 standardized moments for scipy.stat distributions

from scipy.stats import *

distributions = [
    anglit,
    arcsine,
    (argus, 1),
    (beta, 1, 2),
    (bradford, 1),
    (chi, 4),
    (chi2, 10),
    cosine,
    (dgamma, 2),
    (dweibull, 2),
    expon,
    (exponnorm, 1.5),
    (exponweib, 2, 3),
    (exponpow, 2),
    (fatiguelife, 1),
    (foldnorm, 3),
    (genlogistic, 2),
    (gennorm, 3.4),
    (genexpon, 1, 2, 3),
    (gamma, 3),
    (gengamma, 2, 3),
    gilbrat,
    (gompertz, 2),
    gumbel_r,
    gumbel_l,
    halfnorm,
    (halfgennorm, 3),
    hypsecant,
    (invgauss, 1),
    laplace,
    logistic,
    (loggamma, 1.2),
    maxwell,
    moyal,
    (nakagami, 0.9),
    norm,
    (norminvgauss, 2, 1),
    (pearson3, 1),
    (reciprocal, 1, 2),
    rayleigh,
    (rice, 1.2),
    semicircular,
    (skewnorm, -0.5),
    (t, 10),
    (trapz, 0.2, 0.7),
    (triang, 0.3),
    (truncexpon, 2.9),
    (truncnorm, -2, 1.5),
    uniform,
    wald,
    (weibull_min, 2.2),
    (weibull_max, 1.3),
    (wrapcauchy, 0.4)
]

for distr in distributions:
    if isinstance(distr, tuple):
        d = distr[0](*distr[1:])
        name = distr[0].name
    else:
        d = distr
        name = distr.name
    mu = d.mean()
    s = d.std()
    sm = []
    for k in range(3, 8 + 1):
        central_moment = d.expect(lambda x: (x - mu) ** k)
        std_moment = central_moment / s ** k
        sm.append(std_moment)
    print('{{' + ', '.join(map(lambda x: f'{x:22.15e}', sm)) + f'}}, "{name}"}},')
