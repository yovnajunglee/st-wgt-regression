# Bayesian Spatio-Temporal Weighted Regression

Code for models described in:

> **Application of Bayesian Spatio-Temporal Weighted Regression for Integrating Missing and Misaligned Environmental Data**

---

## Overview

This repository provides C++ implementations (via **RcppArmadillo**) of Bayesian spatio-temporal regression models that integrate environmental exposure data across space and time using kernel-weighted aggregation. The models are designed to handle missing and misaligned data — for example, satellite-derived wildfire smoke observations that do not align exactly with ground-level air quality monitoring locations.

All models share a common structure: a spatially and temporally weighted predictor is constructed using exponential kernel functions over user-specified spatial buffers and temporal lag windows, then embedded in a Bayesian linear regression fitted via MCMC.

---

## Models

| Function pair | Model | Spatial kernel |
|---|---|---|
| `STExp` / `STExpPred` | Baseline | Exponential |
| `FuncSpTReg` / `FuncSpTRegPred` | Functional Mean | Exponential |
| `hFuncIndSpTReg` / `FuncIndSpTRegPred` | Full (heterogeneous scale) | Exponential |
| `FuncIndBetaSpTReg` / `FuncIndBetaSpTRegPred` | Varying Coefficient | Exponential |
| `bayes_collocated` / `bayes_collocated_pred` | Collocated (no kernel) | — |


---

## Arguments

The following arguments appear across functions where applicable.

| Argument | Type | Description |
|---|---|---|
| `Xmat` | `List` (length `Ny`) | Each element `Xmat[[i]]` is an `N × T_X` matrix of predictor values in the spatial buffer around location `i`. `N` is an upper bound on the number of observations within radius `r`. Each element is area-weighted: `Xmat[[i]][n,t] = Voronoi cell area × X`. |
| `dist` | `List` (length `Ny`) | Each element `dist[[i]]` contains spatial distances from observations in `Xmat[[i]]` to location `i`. |
| `Ind` | `List` (length `Ny`) | Each element `Ind[[i]]` is a `T_i × T_X` indicator matrix with 1 where the lag value is ≤ `maxL` (i.e., `q`), 0 otherwise. |
| `Lag` | `List` (length `Ny`) | Each element `Lag[[i]]` is a `T_i × T_X` matrix of temporal lag values (in days) between response time points and predictor time points. |
| `maxL` | `int` | Maximum temporal lag `q` (days). |
| `Pmax` | `int` | Maximum spatial radius `r` (units matching `dist`). |
| `Ny` | `int` | Number of spatial monitoring locations. |
| `NyNt` | `int` | Total number of space–time observations (sum of `T_i` across all locations). |
| `Yvec` | `mat` | Response vector ordered by location then time. |
| `Bmat` | `mat` | Matrix of spline basis functions or additional covariates appended to the design matrix. |
| `FireInd` | `mat` | Binary indicator vector (`NyNt × 1`): 1 on fire/smoke-plume days, 0 otherwise. |
| `sigma_prior` | `vec` | Shape and rate parameters `(a, b)` for the Inverse-Gamma prior on `sigma²` (error variance). |
| `sigmab_prior` | `vec` | Shape and rate parameters `(a, b)` for the Inverse-Gamma prior on `sigma_b²` (coefficient variance). |
| `phi*_prior` | `vec` | Shape and rate parameters for the Gamma prior on the spatial scale `phi*`. |
| `ell_prior` | `vec` | Shape and rate parameters for the Gamma prior on `1/ell` (temporal scale). |
| `*_fixed` | `double` | Parameters held fixed at this value throughout sampling (e.g., `phi0_fixed`, `ell_fixed`). |
| `init_beta` | `vec` | Initial values for the regression coefficient vector `beta`. |
| `init_scale` | `vec` | Initial values for the kernel scale parameters `(phi, ell)`. |
| `niter` | `int` | Total number of MCMC iterations. The first 50% are discarded as burn-in in prediction functions. |
| `rphi*`, `rell` | `double` | Step-size variances for log-normal Metropolis-Hastings proposals for `phi*` and `ell`. |
| `gamma` | `double` | Unused legacy argument (retained for API compatibility). |

---

## Return Values

All MCMC functions return an R `List` containing posterior samples for each parameter. Prediction functions return an `NyNt × (niter/2)` matrix of posterior predictive draws.

| Name | Description |
|---|---|
| `beta` | `dim × niter` matrix of regression coefficient samples |
| `sigmae` | `niter`-vector of error variance samples (`sigma²`) |
| `sigmab` | `niter`-vector of coefficient variance samples (`sigma_b²`) |
| `phi` / `phi0` / `phi1` | `niter`-vector(s) of spatial scale samples |
| `ell` | `niter`-vector of temporal scale samples |
| `alpha` | `2 × niter` matrix of hierarchical log-scale regression coefficients (full model only) |
| `nu` | `niter`-vector of log-scale variance samples (full model only) |
| `accept1`, `accept2` | MH acceptance counts for scale parameters |
| `prop` | `2 × niter` matrix of proposed scale values (for diagnostics) |

---
