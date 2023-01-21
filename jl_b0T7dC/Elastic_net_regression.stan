data {
        int<lower=0> p;  // number of parameters (cols)
        int<lower=0> n;  // number of samples (rows)
        matrix[n, p] X;
        real y[n];

        real<lower=0> lambda_1;
        real<lower=0> lambda_2;
    }
    parameters {
        vector[p] beta;
        real<lower=0> sigma;
    }
    model {
        // Prior on beta
        beta ~ exponential(- lambda_1 * sum(abs(beta)) - lambda_2 * sqrt(sum(square(beta))));
        // beta ~ normal(-lambda_1/lambda_2/2, sigma^2 / lambda_2^2 );
        // beta ~ normal((alpha-1)/(2 alpha), sigma^2 / (lambda * alpha) );

        // Likelihood of data
        y ~ normal(X * beta, sigma^2);
    }