
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::depends(RcppNumerical)]]
// [[Rcpp::depends(RcppEigen)]]



// we only include RcppArmadillo.h which pulls Rcpp.h in for us
#include "RcppArmadillo.h"
#include <RcppNumerical.h>
#include "RcppEigen.h"
#include <cmath>
#include <iostream>
using namespace arma;
using namespace Numer;
using namespace Eigen;


// Sample data using rnorm - vector of means and double sd
// [[Rcpp::export]]
arma::mat rcpp_dnorm_1(arma::mat X, arma::mat mean, double sd, bool log){
  arma::mat ff(X.n_rows, 1);
  
  for(int i = 0 ; i < X.n_rows; ++i){
    ff.col(0).row(i) = R::dnorm(as_scalar(X.col(0).row(i)),
           as_scalar(mean.col(0).row(i)), sd, log);
  }
  
  return ff;
  
}


// Sample from a MVN posterior distribution
// for large p by Bhattacharya et al. [assuming D is diag of 1]
// [[Rcpp::export]]
arma::mat sampleFastMVN(arma::mat Phi, arma::mat Dmat, arma::mat alpha) {
  // Obtain dimensions
  int p = Phi.n_cols;
  int n = Phi.n_rows;
  
  // Sample using algorithm
  
  // Step 1
  arma::mat u = arma::randn(p)/sqrt(Dmat.diag());
  arma::mat delta = arma::randn(n);
  
  // Step 2: Set v = phi*u + delta
  arma::mat v = Phi*u + delta;
  
  // Step 3: Solve system of linear equations
  
  arma::mat w = solve(Phi*Dmat*Phi.t() + arma::eye(n, n), alpha - v );
  
  //   solve(crossprod(sqrt(Ddiag)*t(Phi)) + diag(n), #Phi%*%diag(Ddiag)%*%t(Phi) + diag(n)
  //          alpha - v)
  
  
  // Step 4: Set theta = u + DPhiTw
  arma::mat theta = u + Dmat*Phi.t()*w;
  
  return (theta) ;
  
}

// [[Rcpp::export]]
arma::mat construct_Qmat_cov_List_fast(Rcpp::List Xmat,
                                       Rcpp::List dist,
                                       double phi,
                                       double ell,
                                       Rcpp::List Ind,
                                       Rcpp::List Lag,
                                       int maxL,
                                       int Pmax,
                                       int Ny,
                                       int NyNt){
  //******
  //Function to create weighted predictor
  //******
  
  arma::mat Fmat(NyNt, 1);
  arma::mat Wi;
  arma::mat X_i;
  arma::mat X;
  arma::mat D;
  arma::mat Lmat;
  arma::mat I;
  arma::mat F_i;
  int t = 0;
  double k1 = phi*(1-exp(-Pmax/phi));
  double k2 = ell*(1-exp(-maxL/ell));
  
  for(int i=0; i<Ny; ++i){
    
    X =  Rcpp::as<arma::mat>(Xmat[i]);
    D = Rcpp::as<arma::mat>(dist[i]);
    Lmat = Rcpp::as<arma::mat>(Lag[i]);
    I = Rcpp::as<arma::mat>(Ind[i]);
    //cout << i << std::endl;
    arma::mat F_i(1, X.n_cols);
    for(int j = 0; j<X.n_cols;++j){
      X_i = X.col(j);
      //cout << X_i.n_cols << std::endl;
      Wi = exp(-D.col(j)/phi);
      //cout << Wi.n_cols << std::endl;
      //cout << "Error here 1" << std::endl;
      
      F_i.row(0).col(j) = Wi.t()*X_i;
    }
    
    for(int ty=0; ty < Lmat.n_rows; ++ty){
      Fmat.row(t) =  (F_i%I.row(ty))*exp(-(pow(Lmat.row(ty).t(),1)/ell))/(k1*k2);
      t = t + 1;
    }
    
    
  }
  Fmat = join_rows(ones<vec>(NyNt), Fmat);
  
  return Fmat;
}

// [[Rcpp::export]]
Rcpp::List ST_Exp(Rcpp::List Xmat,
                                 Rcpp::List dist,
                                 Rcpp::List Ind,
                                 Rcpp::List Lag,
                                 int maxL,
                                 int Pmax,
                                 int Ny,
                                 int NyNt,
                                 arma::mat Yvec,
                                 arma::vec sigma_prior, arma::vec sigmab_prior,
                                 arma::vec phi_prior, arma::vec ell_prior,
                                 int niter, double rphi, double rell, arma::vec init_scale,
                                 arma::vec init_beta, double gamma){
  
  //******
  // MCMC for baseline method
  //******
  
  
  // Define variables
  
  double nobs = Yvec.n_rows;
  arma::mat beta(2, niter);
  beta.col(0) = init_beta; 
  double a1 = sigma_prior(0);
  double b1 = sigma_prior(1);
  double phiprop;
  double ellprop;
  double a2 = sigmab_prior(0);
  double b2 = sigmab_prior(1);
  
  cout << "Initialising ... " << std::endl;
  arma::vec sigma2(niter);
  sigma2(0) =  1/randg( distr_param(a1,1/b1) );
  
  arma::vec sigma2b(niter);
  sigma2b(0) = 1/randg( distr_param(a2,1/b2) );
  
  
  arma::vec phisample(niter);
  phisample(0) = init_scale(0);
  
  arma::mat postmean_beta;
  arma::mat postprec_beta;
  arma::vec ellsample(niter);
  ellsample(0) = init_scale(1); 
  
  
  // Init. weighted predictor
  arma::mat Qmat_k = construct_Qmat_cov_List_fast(Xmat, dist, phisample(0),
                                                  ellsample(0),Ind, Lag, maxL, Pmax, Ny,NyNt);
  
  arma::mat Qmat_prop;
  
  double loglikeprop;
  double loglikek;
  double ratio;
  double u;
  int accept1 = 0;
  int accept2 = 0;

  arma::mat prop(2, niter);
  
  
  cout << "Running MCMC ... " << std::endl;
  for(int n=1;n<niter;++n){
    if(n % 100 == 0){cout<< n << std::endl;} 

    //cout << "Sampling beta ... " << std::endl;
    
    postprec_beta = Qmat_k.t()*Qmat_k/sigma2(n-1) + eye(1 + 1,1 + 1)*(1/sigma2b(n-1));
    
    postmean_beta = ((postprec_beta).i())*Qmat_k.t()*Yvec/sigma2(n-1);
    
    beta.col(n) = (arma::mvnrnd(postmean_beta,postprec_beta.i()));

    
    sigma2(n) = 1/arma::randg(distr_param(nobs/2.0 + a1,
                              as_scalar(1/(0.5*(Yvec - Qmat_k*beta.col(n)).t()*(Yvec - Qmat_k*beta.col(n)) + b1) )));
    //sigma2b(n) = 1/arma::randg(distr_param(0.5 + a2, 1/(0.5*as_scalar(beta.col(n).t()*beta.col(n)) + b2) ));
    sigma2b(n) = 1/Rcpp::rgamma(1, 0.5 + a2, 1/(0.5*as_scalar(beta.col(n).t()*beta.col(n)) + b2) )[0];
    
    // Scale parameter phi for spatial covariance function
    
    phiprop = exp(log(phisample(n-1)) + sqrt(rphi)*randn(distr_param(0,1)));
    Qmat_prop = construct_Qmat_cov_List_fast(Xmat, dist, (phiprop),
                                             ellsample(n-1), Ind, Lag, maxL, Pmax, Ny, NyNt);
    
    loglikeprop = accu(rcpp_dnorm_1(Yvec, Qmat_prop*beta.col(n), sqrt(as_scalar(sigma2(n))), true)) +
      // R::dexp(phiprop, phi_prior(1), true)+
      R::dgamma(as_scalar(phiprop),phi_prior(0), phi_prior(1), true) +//- 2*log(as_scalar(phiprop))+
      log(phiprop);
    
    loglikek =  accu(rcpp_dnorm_1(Yvec, Qmat_k*beta.col(n), sqrt(as_scalar(sigma2(n))), true))  +
      // R::dexp(as_scalar(phisample(n-1)), phi_prior(1), true)+
      R::dgamma(as_scalar(phisample(n-1)), phi_prior(0), phi_prior(1), true)+//- 2*log(as_scalar(phisample(n-1))) +
      log(phisample(n-1)) ;
    ratio = std::min(exp(loglikeprop - loglikek),1.00);
    u = randu();
    
    if(ratio>=u){
      phisample(n) = phiprop;
      Qmat_k = Qmat_prop;
      accept1 = accept1 + 1;
    }else{
      phisample(n) = phisample(n-1);
    }
    
    // Scale parameter ell for time kernel function
    
    ellprop = exp(log(ellsample(n-1)) +  sqrt(rell)*randn(distr_param(0,1)));
    prop(1,n) = ellprop;
    
    Qmat_prop = construct_Qmat_cov_List_fast(Xmat, dist, phisample(n),
                                             ellprop, Ind, Lag, maxL, Pmax, Ny,NyNt);
    
    loglikeprop = accu(rcpp_dnorm_1(Yvec, Qmat_prop*beta.col(n), sqrt(as_scalar(sigma2(n))), true)) +
      R::dgamma(1/as_scalar(ellprop),ell_prior(0), 1/ell_prior(1),  true) - 2*log(ellprop) +
      log(ellprop);
    
    loglikek =  accu(rcpp_dnorm_1(Yvec, Qmat_k*beta.col(n), sqrt(as_scalar(sigma2(n))), true))  +
      R::dgamma(1/as_scalar(ellsample(n-1)),ell_prior(0), 1/ell_prior(1), true) - 2*log(ellsample(n-1)) +
      log(ellsample(n-1));
    
    ratio = std::min(exp(loglikeprop - loglikek),1.00);
    u = randu();
    if(ratio>u){
      ellsample(n) = ellprop;
      Qmat_k = Qmat_prop;
      accept2 = accept2 + 1;
    }else{
      ellsample(n) = ellsample(n-1);

    }
    
  }
  
  return  Rcpp::List::create(Rcpp::Named("beta")=beta,
                             Rcpp::Named("sigmae") = sigma2,
                             Rcpp::Named("sigmab") = sigma2b,
                             Rcpp::Named("phi")= phisample,
                             Rcpp::Named("ell")= ellsample,
                             Rcpp::Named("accept1")=accept1,
                             Rcpp::Named("accept2")=accept2,
                             Rcpp::Named("prop")=prop);
}




//[[Rcpp::export]]
arma::mat predict_cov_list(Rcpp::List Xmat_oos, Rcpp::List dist_oos,
                           Rcpp::List Ind,
                           Rcpp::List L,
                           arma::mat Bmat,
                           int maxL,
                           int Pmax,
                           int Ny_oos,
                           int NyNt,
                           Rcpp::List mcmc, int niter){
   // *** 
   // Function to sample from posterior predictive distributions
   //***
  
  arma::mat ypred(NyNt, 0.5*niter);
  
  arma::mat Qmat_k;//(Ny_oos*Jy, niter);
  arma::vec phi = mcmc["phi"];
  arma::vec ell = mcmc["ell"];
  arma::vec sigma = mcmc["sigmae"];
  arma::mat beta = mcmc["beta"];
  int k = 0 ;
  arma::vec mu;
  
  for(int n =(0.5*niter); n < niter; ++n){
    if(n % 1000 == 0){cout<< n << std::endl;}
    
    Qmat_k =  construct_Qmat_cov_List_fast(Xmat_oos, dist_oos, phi(n),
                                           ell(n),Ind, L, maxL, Pmax, Ny_oos, NyNt);
    mu = Qmat_k*beta.col(n);
    
    for(int i = 0; i < NyNt; ++i){
      ypred.col(k).row(i) = randn( distr_param(mu(i), sqrt(sigma(n))) );
    }
    
    k = k + 1;
  }
  
  return ypred;
  
  
}

//[[Rcpp::export]]
arma::mat RcppMatern(arma::mat dist, double kappa, double phi){
  Rcpp::Environment geoR("package:geoR");
  Rcpp::Function matern("matern");
  arma::mat BA = Rcpp::as<arma::mat>(matern(dist,phi,kappa));
  return BA;
}


// [[Rcpp::export]]
arma::mat QFuncSpTReg(Rcpp::List Xmat,
                      Rcpp::List dist,
                      double phi,
                      double ell,
                      arma::mat Bmat,
                      Rcpp::List Ind,
                      Rcpp::List Lag,
                      int maxL,
                      int Pmax,
                      int Ny,
                      int NyNt){
  
  // ***
  // Construct weighted predictor for functional mean model
  // ***
  
  arma::mat Fmat(NyNt, 1);
  arma::mat Wi(1, Pmax);
  arma::mat X_i(Pmax, 1);
  arma::mat X;
  arma::mat D;
  arma::mat Lmat;
  arma::mat I;
  arma::mat F_i;
  int t = 0;
  
  double rho = sqrt(3)/phi;
  double k1 =  phi*(1-exp(-Pmax/phi));
  double k2 = ell*(1-exp(-maxL/ell));
  //std::cout << rho << std::endl;
  for(int i=0; i<Ny; ++i){
    
    X =  Rcpp::as<arma::mat>(Xmat[i]);
    D = Rcpp::as<arma::mat>(dist[i]);
    Lmat = Rcpp::as<arma::mat>(Lag[i]);
    I = Rcpp::as<arma::mat>(Ind[i]);
    arma::mat F_i(1, X.n_cols);
    for(int j = 0; j<X.n_cols;++j){
      X_i = X.col(j);
      Wi =  exp(-D.col(j)/phi) ; 
      
      F_i.row(0).col(j) = Wi.t()*X_i;
    }
    
    for(int ty=0; ty < Lmat.n_rows; ++ty){
      Fmat.row(t) =  (F_i%I.row(ty))*exp(-(pow(Lmat.row(ty).t(),1)/ell))/(k1*k2);
      t = t + 1;
    }
}
  Fmat = join_rows(Bmat, Fmat);
  
  return Fmat;
}


// [[Rcpp::export]]
Rcpp::List FuncSpTReg(Rcpp::List Xmat,
                      Rcpp::List dist,
                      Rcpp::List Ind,
                      Rcpp::List Lag,
                      int maxL,
                      int Pmax,
                      int Ny,
                      int NyNt,
                      arma::mat Yvec,
                      arma::mat Bmat,
                      arma::vec sigma_prior, arma::vec sigmab_prior,
                      arma::vec phi_prior, arma::vec ell_prior,
                      int niter, double rphi, double rell, arma::vec init_scale,
                      arma::vec init_beta, double gamma){
  
  // ***
  // MCMC for functional mean model
  // ***
  
  double nobs = Yvec.n_rows;
  
  arma::mat beta(1+Bmat.n_cols, niter);
  beta.col(0) = init_beta; 
  double a1 = sigma_prior(0);
  double b1 = sigma_prior(1);
  
  double a2 = sigmab_prior(0);
  double b2 = sigmab_prior(1);
  
  cout << "Initialising ... " << std::endl;
  arma::vec sigma2(niter);
  sigma2(0) = 1/randg( distr_param(a1,1/b1) );
  
  arma::vec sigma2b(niter);
  sigma2b(0) = 1/randg( distr_param(a2,1/b2) );
  
  
  arma::vec phisample(niter);
  phisample(0) = init_scale(0); 
  
  arma::mat postmean_beta;
  arma::mat postprec_beta;
  
  double phiprop;
  double ellprop;
  
  arma::vec ellsample(niter);
  ellsample(0) = init_scale(1); //randu();// 0.25;
  
  arma::mat Qmat_k =  QFuncSpTReg(Xmat, dist, phisample(0),
                                  ellsample(0), Bmat, Ind, Lag, maxL, Pmax, Ny, NyNt);
  
  arma::mat Qmat_prop;
  
  cout << "Running MCMC ... " << std::endl;
  
  double loglikeprop;
  double loglikek;
  double ratio;
  double u;
  int accept1 = 0;
  int accept2 = 0;
  
  arma::mat prop(2, niter);
  
  for(int n=1;n<niter;++n){
    if(n % 100 == 0){cout<< n << std::endl;} 
    
    postprec_beta = Qmat_k.t()*Qmat_k/sigma2(n-1) + eye(1 + Bmat.n_cols,1 + Bmat.n_cols)*(1/sigma2b(n-1));
    
    postmean_beta = (inv(postprec_beta))*Qmat_k.t()*Yvec/sigma2(n-1);
    //cout << "Sampling beta ... " << std::endl;
    beta.col(n) = (arma::mvnrnd(postmean_beta,inv(postprec_beta)));
    //cout << "Sampling sigmas ... " << std::endl;
    sigma2(n) = 1/arma::randg(distr_param(nobs/2.0 + a1,
                              as_scalar(1/(0.5*(Yvec - Qmat_k*beta.col(n)).t()*(Yvec - Qmat_k*beta.col(n)) + b1) )));
    sigma2b(n) = 1/arma::randg(distr_param(0.5 + a2, 1/(0.5*as_scalar(beta.col(n).t()*beta.col(n)) + b2) ));
    
    // Metropolis steps for scale parameters
    
    // Scale parameter phi for spatial covariance function
    
    
    phiprop = exp(log(phisample(n-1)) + sqrt(rphi)*randn(distr_param(0,1)));
    prop(0,n) = phiprop;
    Qmat_prop =  QFuncSpTReg(Xmat, dist, (phiprop),
                             ellsample(n-1),Bmat, Ind, Lag, maxL, Pmax, Ny, NyNt);
    
    loglikeprop = accu(rcpp_dnorm_1(Yvec, Qmat_prop*beta.col(n), sqrt(as_scalar(sigma2(n))), true)) +
      R::dgamma(as_scalar(phiprop),phi_prior(0), phi_prior(1), true) +//- 2*log(as_scalar(phiprop))+
      log(phiprop);
    
    loglikek =  accu(rcpp_dnorm_1(Yvec, Qmat_k*beta.col(n), sqrt(as_scalar(sigma2(n))), true))  +
      R::dgamma(as_scalar(phisample(n-1)), phi_prior(0), 1/phi_prior(1), true) + //- 2*log(as_scalar(phisample(n-1))) +
      log(phisample(n-1)) ;
    
    ratio = std::min(exp(loglikeprop - loglikek),1.00);
    u = randu();
    
    if(ratio>=u){
      phisample(n) = phiprop;
      Qmat_k = Qmat_prop;
      accept1 = accept1 + 1;
    }else{
      phisample(n) = phisample(n-1);
    }
    
    // Scale parameter ell for time covariance function
    ellprop = exp(log(ellsample(n-1)) +  sqrt(rell)*randn(distr_param(0,1)));
    //pow(exp(adapt2),2)*as_scalar(Sigma.col(1).row(1))*randn(distr_param(0,1)));
    prop(1,n) = ellprop;
    
    Qmat_prop =  QFuncSpTReg(Xmat, dist, phisample(n),
                             ellprop, Bmat,Ind, Lag, maxL, Pmax, Ny, NyNt);
    
    loglikeprop = accu(rcpp_dnorm_1(Yvec, Qmat_prop*beta.col(n), sqrt(as_scalar(sigma2(n))), true)) +
      R::dgamma(1/as_scalar(ellprop),ell_prior(0), 1/ell_prior(1),  true) - 2*log(ellprop) +
      log(ellprop);
    
    loglikek =  accu(rcpp_dnorm_1(Yvec, Qmat_k*beta.col(n), sqrt(as_scalar(sigma2(n))), true))  +
      R::dgamma(1/as_scalar(ellsample(n-1)),ell_prior(0), 1/ell_prior(1), true) - 2*log(ellsample(n-1)) +
      log(ellsample(n-1));
    
    ratio = std::min(exp(loglikeprop - loglikek),1.00);
    u = randu();
    if(ratio>u){
      ellsample(n) = ellprop;
      Qmat_k = Qmat_prop;
      accept2 = accept2 + 1;
    }else{
      ellsample(n) = ellsample(n-1);
      
    }
  }
  
  return  Rcpp::List::create(Rcpp::Named("beta")=beta,
                             Rcpp::Named("sigmae") = sigma2,
                             Rcpp::Named("sigmab") = sigma2b,
                             Rcpp::Named("phi")= phisample,
                             Rcpp::Named("ell")= ellsample,
                             Rcpp::Named("accept1")=accept1,
                             Rcpp::Named("accept2")=accept2,
                             Rcpp::Named("prop")=prop);
}


//[[Rcpp::export]]
arma::mat FuncSpTRegPred(Rcpp::List Xmat_oos, Rcpp::List dist_oos,
                         Rcpp::List Ind,
                         Rcpp::List L,
                         arma::mat Bmat,
                         int maxL,
                         int Pmax,
                         int Ny_oos,
                         int NyNt,
                         Rcpp::List mcmc, int niter){
  
  // ***
  // Sample from posterior predictive dist. for functional mean model
  // ***
  arma::mat ypred(NyNt, 0.5*niter);
  
  arma::mat Qmat_k;
  arma::vec phi = mcmc["phi"];
  arma::vec ell = mcmc["ell"];
  arma::vec sigma = mcmc["sigmae"];
  arma::mat beta = mcmc["beta"];
  int k = 0 ;
  arma::vec mu;
  
  for(int n =(0.5*niter); n < niter; ++n){
    if(n % 1000 == 0){cout<< n << std::endl;}
    
    
    Qmat_k = QFuncSpTReg(Xmat_oos, dist_oos, phi(n),
                         ell(n),Bmat, Ind, L, maxL, Pmax, Ny_oos,NyNt);
    mu = Qmat_k*beta.col(n);
    
    for(int i = 0; i < NyNt; ++i){
      ypred.col(k).row(i) = randn( distr_param(mu(i), sqrt(sigma(n))) );
    }
    
    k = k + 1;
  }
  
  return ypred;
  
  
}





// [[Rcpp::export]]
arma::mat hQFuncIndSpTReg(Rcpp::List Xmat,
                          Rcpp::List dist,
                          double phi0, double phi1,
                          double ell,
                          arma::mat Bmat,
                          arma::mat FireInd,
                          Rcpp::List Ind,
                          Rcpp::List Lag,
                          int maxL,
                          int Pmax,
                          int Ny,
                          int NyNt, double nu){
  // ***
  // Construct weighted predictor for full model
  // ***  
  arma::mat Fmat(NyNt,1);
  arma::mat Fmat_1(NyNt, 1);
  arma::mat Fmat_0(NyNt, 1);
  arma::mat Wi;
  arma::mat X_i;
  arma::mat X;
  arma::mat D;
  arma::mat Lmat;
  arma::mat I;
  int t = 0;
  double phiaug = phi1;
  double phit =  phi0;
  double k1 = phit*(1-exp(-Pmax/phit));
  double k1aug = phiaug*(1-exp(-Pmax/phiaug));
  double k2 = ell*(1-exp(-maxL/ell));
  
  for(int i=0; i<Ny; ++i){
    
    X =  Rcpp::as<arma::mat>(Xmat[i]);
    D = Rcpp::as<arma::mat>(dist[i]);
    Lmat = Rcpp::as<arma::mat>(Lag[i]);
    I = Rcpp::as<arma::mat>(Ind[i]);
    //cout << i << std::endl;
    arma::mat F_i0(1, X.n_cols);
    arma::mat F_i1(1, X.n_cols);
    for(int j = 0; j<X.n_cols;++j){
      X_i = X.col(j);

      // When FireInd = 0
      Wi = exp(-D.col(j)/phit);
      F_i0.row(0).col(j) = Wi.t()*X_i;
      // When FireInd = 1
      Wi = exp(-D.col(j)/(phiaug));
      F_i1.row(0).col(j) = Wi.t()*X_i;
    }
    
    
    for(int ty=0; ty < Lmat.n_rows; ++ty){
      Fmat_0.row(t) =  (F_i0%I.row(ty))*exp(-(pow(Lmat.row(ty).t(),1)/ell));
      Fmat_1.row(t) =  (F_i1%I.row(ty))*exp(-(pow(Lmat.row(ty).t(),1)/ell));
      t = t + 1;
    }
    
    
  }
  
  Fmat_0 = Fmat_0/(k1*k2);
  Fmat_1 = Fmat_1/(k1aug*k2);
  Fmat = join_rows(Bmat, Fmat_0%(1-FireInd) + Fmat_1%(FireInd), Fmat_1%FireInd);
  return Fmat;
}


// [[Rcpp::export]]
Rcpp::List hFuncIndSpTReg(Rcpp::List Xmat,
                          Rcpp::List dist,
                          Rcpp::List Ind,
                          Rcpp::List Lag,
                          int maxL,
                          int Pmax,
                          int Ny,
                          int NyNt,
                          arma::mat Yvec,
                          arma::mat FireInd,
                          arma::mat Bmat,
                          arma::vec sigma_prior, arma::vec sigmab_prior,
                          arma::vec phi_prior, arma::vec ell_prior,
                          int niter, double phi0_fixed, double rphi1, double ell_fixed, arma::vec init_scale,
                          arma::vec init_beta, double gamma){
  
  // ***
  // MCMC for full model
  // ***
  
  double nobs = Yvec.n_rows;
  
  arma::mat beta((Bmat.n_cols+2), niter);
  beta.col(0) = init_beta; 
  double a1 = sigma_prior(0);
  double b1 = sigma_prior(1);
  
  double a2 = sigmab_prior(0);
  double b2 = sigmab_prior(1);
  
  cout << "Initialising ... " << std::endl;
  arma::vec sigma2(niter);
  sigma2(0) =  1/randg( distr_param(a1,1/b1) );
  
  arma::vec sigma2b(niter);
  sigma2b(0) =  1/randg( distr_param(a2,1/b2) );
  arma::mat alpha(2, niter);
  alpha.col(0) = randn(2);
  
  arma::vec nusample(niter);
  nusample(0) = randu() ;
  
  
  arma::vec phi0sample(niter);
  phi0sample(0) = phi0_fixed;
  
  
  arma::vec phi1sample(niter);
  phi1sample(0) = exp(randu());
  
  
  
  arma::mat postmean_beta;
  arma::mat postprec_beta;
  
  
  
  arma::mat postmean_alpha;
  arma::mat postsigma_alpha;
  
  
  double phi0prop;
  double phi1prop;
  
  double phi0aux;
  double phi1aux;
  
  double phiprop;
  double phi;
  double ellprop;
  
  arma::vec ellsample(niter);
  ellsample(0) =  ell_fixed;
  
  arma::mat Qmat_k =  hQFuncIndSpTReg(Xmat, dist, phi0sample(0), phi1sample(0),
                                      ellsample(0), Bmat, FireInd, Ind, Lag, maxL, Pmax, Ny, NyNt, nusample(0));
  
  arma::mat Qmat_prop;
  
  cout << "Running MCMC ... " << std::endl;
  
  double loglikeprop;
  double loglikek;
  double ratio;
  double u;
  int accept1 = 0;
  int accept2 = 0;
  int accept3 = 0;
  arma::vec mu = zeros(2);
  arma::mat Sigma(2, 2, fill::zeros);
  Sigma.row(0).col(0) =1;
  Sigma.row(1).col(1) =1;
  arma::mat prop(2, niter);
  arma::mat Falpha = join_rows(ones<vec>(NyNt), FireInd);
  arma::mat phivec =  (1-FireInd)*phi0sample(0)+ FireInd*phi1sample(0);
  arma::mat phivecprop;
  
  for(int n=1;n<niter;++n){
    if(n % 1000== 0){cout<< n << std::endl;} 
    postprec_beta = Qmat_k.t()*Qmat_k/sigma2(n-1) + eye((2 + Bmat.n_cols),(2 + Bmat.n_cols))*(1/sigma2b(n-1));
    
    postmean_beta = (inv(postprec_beta))*Qmat_k.t()*Yvec/sigma2(n-1);
    //cout << "Sampling beta ... " << std::endl;
    beta.col(n) = (arma::mvnrnd(postmean_beta,inv(postprec_beta)));
    //cout << "Sampling sigmas ... " << std::endl;
    sigma2(n) = 1/arma::randg(distr_param(nobs/2.0 + a1,
                              as_scalar(1/(0.5*(Yvec - Qmat_k*beta.col(n)).t()*(Yvec - Qmat_k*beta.col(n)) + b1) )));
    sigma2b(n) = 1/arma::randg(distr_param(0.5 + a2, 1/(0.5*as_scalar(beta.col(n).t()*beta.col(n)) + b2) ));
    
    //cout << "Sampling scales ... " << std::endl;
    
    // Metropolis steps for scale parameters
    
    // Scale parameter phi for spatial covariance function
    phi1aux = log(phi1sample(n-1)) +  sqrt(rphi1)*randn(distr_param(0,1));
    phi1prop = exp(phi1aux);
    prop(1,n) = phi1prop;
    prop(0,n) = phi0_fixed;
    phi0sample(n) = phi0_fixed;//1.0;
    
    // Sample phi1
    
    Qmat_prop =  hQFuncIndSpTReg(Xmat, dist, (phi0sample(n)), phi1prop,
                                 ellsample(n-1),Bmat, FireInd,Ind, Lag, maxL, Pmax, Ny, NyNt, nusample(n-1));
    phivecprop =  (1-FireInd)*(phi0sample(n)) + FireInd*phi1prop;
    
    loglikeprop = accu(rcpp_dnorm_1(Yvec, Qmat_prop*beta.col(n), sqrt(as_scalar(sigma2(n))), true)) +
      accu(rcpp_dnorm_1(log(phivecprop), as_scalar(alpha(0,n-1)) + as_scalar(alpha(1,n-1))*FireInd,
                        sqrt(as_scalar(nusample(n-1))),true)) +log(phi1prop)+
                          R::dgamma(as_scalar(phi1prop),phi_prior(0), phi_prior(1), true) ;

    loglikek =  accu(rcpp_dnorm_1(Yvec, Qmat_k*beta.col(n), sqrt(as_scalar(sigma2(n))), true))  +
      accu(rcpp_dnorm_1(log(phivec),as_scalar(alpha(0,n-1)) + as_scalar(alpha(1,n-1))*FireInd,
                        sqrt(as_scalar(nusample(n-1))),true)) +log(phi1sample(n-1))+
                          R::dgamma(as_scalar(phi1sample(n-1)),phi_prior(0), phi_prior(1), true);

    ratio = std::min(exp(loglikeprop - loglikek),1.00);
    u = randu();
    
    if(ratio>=u){
      phi1sample(n) = phi1prop;
      phivec = phivecprop;
      Qmat_k = Qmat_prop;
      accept1 = accept1 + 1;
    }else{
      phi1sample(n) = phi1sample(n-1);
    }
    
    ellsample(n) =  ell_fixed;
    
    // // Sample alphas
    postsigma_alpha = inv((Falpha.t()*Falpha)/nusample(n-1) + eye(2,2)*(1/2.0));
    postmean_alpha = postsigma_alpha*Falpha.t()*log(phivec)/nusample(n-1);
    alpha.col(n) = (arma::mvnrnd(postmean_alpha, (postsigma_alpha)));
    // Sample nu
    nusample(n) = 1/arma::randg(distr_param(0.5 + 1, 1/(0.5*as_scalar(alpha.col(n).t()*alpha.col(n)) + 1) ));
  }
  
  return  Rcpp::List::create(Rcpp::Named("beta")=beta,
                             Rcpp::Named("sigmae") = sigma2,
                             Rcpp::Named("sigmab") = sigma2b,
                             Rcpp::Named("phi0")= phi0sample,
                             Rcpp::Named("phi1")= phi1sample,
                             
                             Rcpp::Named("ell")= ellsample,
                             Rcpp::Named("accept1")=accept1,
                             Rcpp::Named("accept2")=accept2,
                             Rcpp::Named("prop")=prop,
                             Rcpp::Named("alpha")=alpha,
                             Rcpp::Named("nu")=nusample);
}


//[[Rcpp::export]]
arma::mat FuncIndSpTRegPred(Rcpp::List Xmat_oos, Rcpp::List dist_oos,
                            Rcpp::List Ind,
                            Rcpp::List L,
                            arma::mat Bmat,
                            arma::mat FireInd,
                            int maxL,
                            int Pmax,
                            int Ny_oos,
                            int NyNt,
                            Rcpp::List mcmc, int niter){
  
  // ***
  // Samples from posterior predictive models
  // ***
  arma::mat ypred(NyNt, 0.5*niter);
  
  arma::mat Qmat_k;
  arma::vec phi0 = mcmc["phi0"];
  arma::vec phi1 = mcmc["phi1"];
  arma::vec ell = mcmc["ell"];
  arma::vec sigma = mcmc["sigmae"];
  arma::mat beta = mcmc["beta"];
  arma::mat nu = mcmc["nusample"];
  
  int k = 0 ;
  arma::vec mu;
  
  for(int n =(0.5*niter); n < niter; ++n){
    if(n % 1000 == 0){cout<< n << std::endl;}
    
    
    Qmat_k = hQFuncIndSpTReg( Xmat_oos, dist_oos, phi0(n),phi1(n),
                              ell(n), Bmat,FireInd,Ind, L, maxL, Pmax, Ny_oos, NyNt, nu(n));
    mu = Qmat_k*beta.col(n);
    
    for(int i = 0; i < NyNt; ++i){
      ypred.col(k).row(i) = randn( distr_param(mu(i), sqrt(sigma(n))) );
    }
    
    k = k + 1;
  }
  
  return ypred;
  
  
}



// [[Rcpp::export]]
arma::mat QFuncIndBetaSpTReg(Rcpp::List Xmat,
                             Rcpp::List dist,
                             double phi,
                             double ell,
                             arma::mat Bmat,
                             arma::mat FireInd,
                             Rcpp::List Ind,
                             Rcpp::List Lag,
                             int maxL,
                             int Pmax,
                             int Ny,
                             int NyNt){
  // ***
  // Construct weighted predictor for varying coefficient model
  // ***  
  arma::mat Fmat(NyNt,1);
  arma::mat Fmat_0(NyNt, 1);
  arma::mat Wi(1, Pmax);
  arma::mat X_i(Pmax, 1);
  arma::mat X;
  arma::mat D;
  arma::mat Lmat;
  arma::mat I;
  arma::mat F_i0;
  arma::mat F_i1;
  
  int t = 0;
  
  
  double k1 = phi*(1-exp(-Pmax/phi));
  double k2 = ell*(1-exp(-maxL/ell));
  
  for(int i=0; i<Ny; ++i){
    X =  Rcpp::as<arma::mat>(Xmat[i]);
    D = Rcpp::as<arma::mat>(dist[i]);
    Lmat = Rcpp::as<arma::mat>(Lag[i]);
    I = Rcpp::as<arma::mat>(Ind[i]);
    arma::mat F_i0(1, X.n_cols);
    for(int j = 0; j<X.n_cols;++j){
      X_i = X.col(j);

      // When FireInd = 0
      Wi = exp(-D.col(j)/phi);
      
      F_i0.row(0).col(j) = Wi.t()*X_i;
      
    }
    
    for(int ty=0; ty < Lmat.n_rows; ++ty){
      Fmat_0.row(t) =  (F_i0%I.row(ty))*exp(-(pow(Lmat.row(ty).t(),1)/ell));
      t = t + 1;
      
    }
    
  }
  
  Fmat_0 = Fmat_0/(k1*k2);
  Fmat = join_rows(Bmat, FireInd, Fmat_0, Fmat_0%FireInd);
  
  return Fmat;
}



// [[Rcpp::export]]
Rcpp::List FuncIndBetaSpTReg(Rcpp::List Xmat,
                             Rcpp::List dist,
                             Rcpp::List Ind,
                             Rcpp::List Lag,
                             int maxL,
                             int Pmax,
                             int Ny,
                             int NyNt,
                             arma::mat Yvec,
                             arma::mat FireInd,
                             arma::mat Bmat,
                             arma::vec sigma_prior, arma::vec sigmab_prior,
                             arma::vec phi_prior, arma::vec ell_prior,
                             int niter, double rphi, double fixed_ell, arma::vec init_scale,
                             arma::vec init_beta, double gamma){
  
  // ***
  // MCMC for varying coefficient model
  // ***
  double nobs = Yvec.n_rows;
  
  arma::mat beta((Bmat.n_cols+3), niter);
  beta.col(0) = init_beta; 
  double a1 = sigma_prior(0);
  double b1 = sigma_prior(1);
  
  double a2 = sigmab_prior(0);
  double b2 = sigmab_prior(1);
  
  cout << "Initialising ... " << std::endl;
  arma::vec sigma2(niter);
  sigma2(0) = 1/randg( distr_param(a1,1/b1) );
  
  arma::vec sigma2b(niter);
  sigma2b(0) =  1/randg( distr_param(a2,1/b2) );
  
  
  arma::vec phisample(niter);
  phisample(0) = randu(distr_param(10,20));
  
  arma::mat postmean_beta;
  arma::mat postprec_beta;
  
  double phiprop;
  double ellprop;
  
  arma::vec ellsample(niter);
  ellsample(0) =  fixed_ell;
  
  arma::mat Qmat_k =  QFuncIndBetaSpTReg(Xmat, dist, phisample(0),
                                         ellsample(0), Bmat, FireInd, Ind, Lag, maxL, Pmax, Ny,NyNt);
  
  arma::mat Qmat_prop;
  
  cout << "Running MCMC ... " << std::endl;
  
  double loglikeprop;
  double loglikek;
  double ratio;
  double u;
  int accept1 = 0;
  int accept2 = 0;
  
  double adapt1 = rphi;
  
  arma::vec mu = zeros(2);
  arma::mat Sigma(2, 2, fill::zeros);
  Sigma.row(0).col(0) =1;
  Sigma.row(1).col(1) =1;
  arma::mat prop(2, niter);
  
  for(int n=1;n<niter;++n){
    if(n % 1000 == 0){cout<< n << std::endl;} 
    postprec_beta = Qmat_k.t()*Qmat_k/sigma2(n-1) + eye((3 + Bmat.n_cols),(3 + Bmat.n_cols))*(1/sigma2b(n-1));
    
    postmean_beta = (inv(postprec_beta))*Qmat_k.t()*Yvec/sigma2(n-1);
    
    beta.col(n) = (arma::mvnrnd(postmean_beta,inv(postprec_beta)));

    sigma2(n) = 1/arma::randg(distr_param(nobs/2.0 + a1,
                              as_scalar(1/(0.5*(Yvec - Qmat_k*beta.col(n)).t()*(Yvec - Qmat_k*beta.col(n)) + b1) )));
    sigma2b(n) = 1/arma::randg(distr_param(0.5 + a2, 1/(0.5*as_scalar(beta.col(n).t()*beta.col(n)) + b2) ));
    
    // Metropolis steps for scale parameters
    
    // Scale parameter phi for spatial covariance function
    
    
    phiprop = exp(log(phisample(n-1)) + sqrt(rphi)*randn(distr_param(0,1)));
    prop(0,n) = phiprop;
    Qmat_prop =  QFuncIndBetaSpTReg(Xmat, dist, (phiprop),
                                    ellsample(n-1),Bmat, FireInd,Ind, Lag, maxL, Pmax, Ny, NyNt);
    
    loglikeprop = accu(rcpp_dnorm_1(Yvec, Qmat_prop*beta.col(n), sqrt(as_scalar(sigma2(n))), true)) +
      R::dgamma(as_scalar(phiprop),phi_prior(0), phi_prior(1), true) +// - 2*log(as_scalar(phiprop))+
      log(phiprop);
    
    loglikek =  accu(rcpp_dnorm_1(Yvec, Qmat_k*beta.col(n), sqrt(as_scalar(sigma2(n))), true))  +
      R::dgamma(as_scalar(phisample(n-1)), phi_prior(0), phi_prior(1), true) +//- 2*log(as_scalar(phisample(n-1))) +
      log(phisample(n-1));
    ratio = std::min(exp(loglikeprop - loglikek),1.00);
    u = randu();
    
    if(ratio>=u){
      phisample(n) = phiprop;
      Qmat_k = Qmat_prop;
      accept1 = accept1 + 1;
    }else{
      phisample(n) = phisample(n-1);
    }
    ellsample(n) =  fixed_ell;
  }
  
  return  Rcpp::List::create(Rcpp::Named("beta")=beta,
                             Rcpp::Named("sigmae") = sigma2,
                             Rcpp::Named("sigmab") = sigma2b,
                             Rcpp::Named("phi")= phisample,
                             
                             Rcpp::Named("ell")= ellsample,
                             Rcpp::Named("accept1")=accept1,
                             Rcpp::Named("accept2")=accept2,
                             Rcpp::Named("prop")=prop);
}


//[[Rcpp::export]]
arma::mat FuncIndBetaSpTRegPred(Rcpp::List Xmat_oos, Rcpp::List dist_oos,
                                Rcpp::List Ind,
                                Rcpp::List L,
                                arma::mat Bmat,
                                arma::mat FireInd,
                                int maxL,
                                int Pmax,
                                int Ny_oos,
                                int NyNt,
                                Rcpp::List mcmc, int niter){
  // ***
  // Sampling from posterior predictive distribution varying coefficient model
  // ***'
  
  arma::mat ypred(NyNt, 0.5*niter);
  arma::mat Qmat_k;
  arma::vec phi = mcmc["phi"];
  arma::vec ell = mcmc["ell"];
  arma::vec sigma = mcmc["sigmae"];
  arma::mat beta = mcmc["beta"];
  int k = 0 ;
  arma::vec mu;
  
  for(int n =(0.5*niter); n < niter; ++n){
    if(n % 1000 == 0){cout<< n << std::endl;}
    Qmat_k = QFuncIndBetaSpTReg(Xmat_oos, dist_oos, phi(n),
                                ell(n), Bmat,FireInd,Ind, L, maxL, Pmax, Ny_oos,NyNt);
    mu = Qmat_k*beta.col(n);
    
    for(int i = 0; i < NyNt; ++i){
      ypred.col(k).row(i) = randn( distr_param(mu(i), sqrt(sigma(n))) );
    }
    
    k = k + 1;
  }
  
  return ypred;
  
  
}



// [[Rcpp::export]]
Rcpp::List bayes_collocated(arma::mat& Xmat,
                            arma::mat Yvec,
                            arma::vec sigma_prior, arma::vec sigmab_prior,
                            int niter,
                            arma::vec init_beta){
  
  // ***
  // MCMC for collocated model
  // ***'
  
  double nobs = Yvec.n_rows;
  
  double np = Xmat.n_cols;
  arma::mat beta(np, niter);
  beta.col(0) = init_beta; 
  double a1 = sigma_prior(0);
  double b1 = sigma_prior(1);
  arma::mat XtX = Xmat.t()*Xmat;
  arma::mat Xty = Xmat.t()*Yvec;
  
  cout << "Initialising ... " << std::endl;
  arma::vec sigma2(niter);

  double a2 = sigmab_prior(0);
  double b2 = sigmab_prior(1);
  
  sigma2(0) =  1/randg( distr_param(a1,1/b1) );
  
  arma::vec sigma2b(niter);
  sigma2b(0) = 1/randg( distr_param(a2,1/b2) );
  
  arma::mat postmean_beta;
  arma::mat postprec_beta;
  
  
  for(int n=1;n<niter;++n){
    if(n % 1000 == 0){cout<< n << std::endl;} 
    
    postprec_beta = XtX/sigma2(n-1) + eye(np,np)*(1/sigma2b(n-1));
    
    postmean_beta = ((postprec_beta).i())*Xty/sigma2(n-1);
    
    beta.col(n) = (arma::mvnrnd(postmean_beta,postprec_beta.i()));

    sigma2(n) = 1/arma::randg(distr_param(nobs/2.0 + a1,
                              as_scalar(1/(0.5*(Yvec - Xmat*beta.col(n)).t()*(Yvec - Xmat*beta.col(n)) + b1) )));
    sigma2b(n) = 1/arma::randg(distr_param(0.5 + a2, 1/(0.5*as_scalar(beta.col(n).t()*beta.col(n)) + b2) ));
    
  }
  
  
  return  Rcpp::List::create(Rcpp::Named("beta")=beta,
                             Rcpp::Named("sigmae") = sigma2,
                             Rcpp::Named("sigmab") = sigma2b);
  
}

//[[Rcpp::export]]
arma::mat bayes_collocated_pred(arma::mat Xmat_oos,
                                int NyNt,
                                Rcpp::List mcmc, int niter){
  // ***
  // Predictive dist. for collocated
  // ***'
  
  arma::mat ypred(NyNt, 0.5*niter);
  arma::vec sigma = mcmc["sigmae"];
  arma::mat beta = mcmc["beta"];
  int k = 0 ;
  arma::vec mu;
  
  for(int n =(0.5*niter); n < niter; ++n){
    
    mu = Xmat_oos*beta.col(n);
    
    for(int i = 0; i < NyNt; ++i){
      ypred.col(k).row(i) = randn( distr_param(mu(i), sqrt(sigma(n))) );
    }
    
    k = k + 1;
  }
  
  return ypred;
  
}


