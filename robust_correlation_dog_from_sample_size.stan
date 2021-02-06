data {
    int<lower=1> N;  // number of observations
    vector[2] x[N];  // input data: rows are observations, columns are the two variables
}

parameters {
    vector[2] mu;                 // mean vector of the marginal t distributions
    real<lower=0> sigma[2];       // variance vector of the marginal t distributions
    real<lower=-1, upper=1> rho;  // correlation coefficient
}

transformed parameters {
    // degrees of freedom of the marginal t distributions
    real<lower=1> nu = (N / 10.0) + 1;

    // Covariance matrix
    cov_matrix[2] cov = [[      sigma[1] ^ 2       , sigma[1] * sigma[2] * rho],
                         [sigma[1] * sigma[2] * rho,       sigma[2] ^ 2       ]];
}

model {
    // Likelihood
    // Bivariate Student's t-distribution instead of normal for robustness
    x ~ multi_student_t(nu, mu, cov);

    // Noninformative priors on all parameters
    sigma ~ uniform(0, 1000);
    mu ~ normal(0, 1000);
    rho ~ uniform(-1,1);//non informative prior between -1 and 1
}

generated quantities {
    // Random samples from the estimated bivariate t-distribution (for assessment of fit)
    vector[2] x_rand;
    x_rand = multi_student_t_rng(nu, mu, cov);
}
