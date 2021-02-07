data {
    int<lower=1> N;  // number of observations
    int<lower=2> V;  // number of variables
    vector[V] x[N];  // input data: rows are observations, columns are the V variables
}

parameters {
    vector[V] mu;                           // locations of the normal distributions
    vector<lower=0>[V] sigma;               // scales of the normal distributions
    cholesky_factor_corr[V] Lrho;           // correlation matrix
}

transformed parameters {
    matrix[V,V] Omega;
    matrix[V,V] cov;
    // degrees of freedom of the marginal t distributions
    real<lower=1> nu = (N / 10.0) + 1;

    // Correlation and Covariance matrices
    Omega = multiply_lower_tri_self_transpose(Lrho);
    cov   = quad_form_diag(Omega, sigma); 
}

model {
    // Noninformative priors on all parameters
    target += uniform_lpdf(sigma | 0, 10);
    target += normal_lpdf(mu | 0, 10);
    target += lkj_corr_cholesky_lpdf(Lrho| 3);
    
    // Likelihood
    // Multivariate Student's t distribution
    target += multi_student_t_lpdf(x | nu, mu, cov);
}

generated quantities {
  vector[V] x_rand;

  x_rand = multi_student_t_rng(nu, mu, cov);
}
