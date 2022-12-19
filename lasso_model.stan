data {
    int<lower=0> p;  // number of parameters (cols)
    int<lower=0> n;  // number of samples (rows)
    matrix[n, p] X;
    real y[n];

    real<lower=0> lambda_1;
    // real<lower=0> lambda_2;
}
parameters {
    vector[p] beta;
    real<lower=0> sigma;
}
model {
    // Prior on beta
    beta ~ double_exponential(0, lambda_1);

    // Likelihood of data
    y ~ normal(X * beta, sigma^2);
}
