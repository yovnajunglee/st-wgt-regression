Code for Models under Application of `Bayesian spatio-temporal weighted regression for integrating missing and misaligned environmental data`

The helper functions are described below:
1. STExp, STExpPred: Baseline model (Exponential kernel functions)
2. FuncSpTReg, FuncSpTRegPred: Functional Mean Model (Exponential kernel functions)
3. hFuncIndSpTReg, FuncIndSpTRegPred: Full Model  (Exponential kernel functions)
4. FuncIndBetaSpTReg, FuncIndBetaSpTRegPred: Varying Coefficient Model  (Exponential kernel functions)

The arguments in the functions (where applicable) are as follows:
1. Xmat: List. Each item i in list corresponds to buffer (adjusted by the Voronoi quadrature) for location i. X[[i]] = N \times T_X matrix where N is an upper bound on the number of observations in a spatial buffer of size r and. Each element of X[[i]] = area of Voronoi cell*X.
2. dist: List. d[[i]] consists of spatial distances of observations in X[[i]] from location i.
3. Ind, Lag: List. Lag[[i]] is T_i \times T_X a lag matrix and Ind[[i]] is the corresponding indicator matrix for lag values <= q.
4. maxL, Pmax: Temporal lag (q), spatial radius (r)
5. Ny: Number of spatial locations.
6. NyNt: Total number of space-time observations.
7. Yvec: Vector of responses ordered by location, time. 
8. param*_prior: Relates to choice of priors (scale/shape parameters) depending on specification in code. 
9. param*_fixed: Parameters that are fixed to this value.
10. Bmat: Matrix of splines or covariates. 
11. FireInd: Indicator vector for smoke plume covariate. 
12. init_param*: Initializations for parameters. 
