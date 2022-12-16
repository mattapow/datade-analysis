using StanSample, MCMCChains
using StatsPlots

include("../datade/src/read.jl")
include("../datade/src/transform_library.jl")

# Don't forget to set the command stan home variable
# set_cmdstan_home!("/opt/miniconda3/envs/datade/bin/cmdstan")

function main(; root_experiment, ensembleAverage, angles)

    elasticnet = "
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
    "

    # load in the data
    n_samples = 1000
    X_in, X_labels, y = multi_loader(root_experiment, n_samples, angles, ensembleAverage)
    X = SampleStatistics(X_in, X_labels)

    # normalise the data
    X, σ_X, μ_X = normalise_sigma(X)
    # X = stat_combine(X)
    println("Data size is: $(size(X.data))")

    # put data into a dictionary
    observed_data = Dict(
        "p" => size(X.data, 2),
        "n" => length(y),
        "y" => y,
        "X" => X.data,
        "lambda_1" => 112.6
        # "lambda_2" => 0.01,
    )

    # run the model
    tmpdir = mktempdir()
    sm = SampleModel("Elastic_net_regression", elasticnet, tmpdir)
    rc = stan_sample(sm, data=observed_data, num_samples=10000, num_warmups=1000)

    if success(rc)
        chns = read_samples(sm, :mcmcchains)

        # Describe the results
        chns |> display

        # Optionally, read samples as a a DataFrame
        df = read_samples(sm, :dataframe)
        first(df, 5)

        df = read_summary(sm)
        df[df.parameters.==:theta, [:mean, :ess]]
    end
    return df, chns, X_labels
end

function extract_variables(chns; alpha=0.05)
    intervals = hpd(chns, alpha=alpha);
    select_variables = []
    # for each variable in the chains (except the error sigma)
    for i in 1:size(chns)[2]-1
        # get the highest posterior intervals
        (lower, upper) = intervals[i, :]
        # if the interval doesn't span zero
        if !(lower < 0.0 && 0.0 < upper)
            # add it to the selected varaibles
            push!(select_variables, labels[i])
        end
    end
    return select_variables
end

ensembleAverage = false
do_mill = true
if do_mill
    root_experiment = "../Mill"
    angles = ["4_46", "5_35", "8_92"]  #"2_23", 
else
    root_experiment = "../Inclined_deep"
    angles = [25, 25.5, 26, 26.5, 27, 27.5]
end
df, chns, labels = main(root_experiment=root_experiment, ensembleAverage=ensembleAverage, angles=angles)

plot(chns, inc_warmup = false)
fp = root_experiment * "/results/bayes_ensemble$ensembleAverage.pdf"
savefig(fp)

select_variables = extract_variables(chns, alpha=0.1)
println(select_variables)
