###
# Read in data from multiple chute simulations
# Filter to valid observations (buffer boundaries)
# Then take average over time and space
# Plot mean(μ) vs mean(I) for various simulations
###
using LaTeXStrings, Plots, Statistics
using GLM, DataFrames

# using datade
include("../datade/src/read.jl")
include("../datade/src/transform_library.jl")
include("../datade/src/lasso.jl")

include("../datade-analysis/post_plots.jl")

function main(root_experiment, n_samples, angles; ensembleAverage=false, use_convex=false, n_augment=0, importance=0.001, intercept=true)
    X_in, X_labels, y_in = multi_loader(root_experiment, n_samples, angles, ensembleAverage)

    # plot linear models
    file_extension = ".pdf"
    # glm_plot(I_k3, y_in, L"I^k_3", L"\tau / P", "mu_Ik3", file_extension = file_extension)
    # glm_plot(I_3, y_in, L"I_3", L"\tau / P", "mu_I3", file_extension = file_extension)
    # glm_plot(log.(I_k1), y, L"\log(I_{k1})", L"\tau / P", "mu_ln_Ik1", file_extension = file_extension)
    # glm_plot(ϕ.^2, I_k1.^.5, L"\phi^2", L"I_{k1}^{1/2}",  "phi2_Ik1half", file_extension = file_extension)
    # glm_plot(I_k1.^.5, ϕ.^2,  L"I_{k1}^{1/2}", L"\phi^2",  "Ik1half_phi2", file_extension = file_extension)
    # # glm_plot(I_k1_half, y, L"I_{k1}^{1/2}", L"\tau / P", "mu_Ik1_half_all", file_extension = file_extension)
    # glm_plot(ϕ, y_in, L"\phi", L"\tau / P", "mu_phi", file_extension = file_extension)
    # glm_plot(I_k3, ϕ, y, L"I_{k3}", L"\phi", L"y", "mu_phi_Ik3_all", file_extension = file_extension)

    # convert to SampleStatistics struct
    restrict = [1, 2, 3, 4, 5, 6, 7]
    X_all = SampleStatistics(X_in[:, restrict], X_labels[restrict])

    # # plot the covariance of parameters
    fp = root_experiment * "/results/cor_ensemble$ensembleAverage.pdf"
    plot_covariance(X_all, fp)

    # augment data
    if n_augment > 0
        println("Augmenting")
        for _ in 1:n_augment
            X_all = stat_combine(X_all)
        end
    end
    println(size(X_all.data))

    # reshape
    y_vec = vec(y_in)

    # normalise data
    X_all, σ_X, μ_X = normalise_sigma(X_all)

    # clean data
    good_data_idx = []
    for i in 1:size(X_all.data, 1)
        if !any_bad(X_all.data[i, :])
            push!(good_data_idx, i)
        end
    end
    X_clean = SampleStatistics(X_all.data[good_data_idx, :], X_all.labels)
    println(size(X_clean.data))

    # split test and train
    X, X_test, y, y_test = train_test_split(X_clean, y_vec, split=0.75, randomise=false)
    
    # train model
    constraint_groups=[]
    β_biased = train(
        X,
        y,
        constraint_groups,
        intercept=intercept,
        use_convex=use_convex,
        start_alpha=0.004,
        alpha_high=1e1,
        alpha_low=1e-10,
        n_terms_target=1,
        log_search=false,
        importance=importance,
        )
    
    # list of included coefficients
    select = findall(list_terms(β_biased, X.labels; omit_final=use_convex, silent=false, importance=importance))
    println(select)
    
    # get unbiased coefficients
    ols = get_beta_unbiased(y, X, σ_X, μ_X, select)
    β_unbiased = coef(ols)
    press = compute_press(β_unbiased, X_test.data[:, select], y_test)
    println("PRESS: $press")
    # p_squared = compute_p_squared(y_test, press)
    # println("p_squared: $p_squared")
    # println("rmse: $loss")

    plot([], [], label=nothing)
    X_unstandard = inv_normalise_sigma(X_clean.data, σ_X, μ_X)
    plot_y_yhat(X_unstandard, y_vec, ols, select, X_clean.labels[select])


    # ## Round 2
    # X_sub = SampleStatistics(X.data[:, select], X.labels[select])
    # X_sub_test = SampleStatistics(X_test.data[:, select], X.labels[select])

    # # augment data
    # if n_augment > 0
    #     println("Augmenting")
    #     for _ in 1:n_augment
    #         X_sub = stat_combine(X_sub)
    #         X_sub_test = stat_combine(X_sub_test)
    #     end
    # end
    # println(size(X_sub.data))

    # β_biased = train(
    #     X_sub,
    #     y,
    #     constraint_groups,
    #     intercept=intercept,
    #     use_convex=false,
    #     start_alpha=1.0,
    #     alpha_high=1e3,
    #     alpha_low=0.0,
    #     n_terms_target=1,
    #     log_search=false,
    #     importance=importance,
    #     )

    # # list of included coefficients
    # select = findall(list_terms(β_biased, X_sub.labels; omit_final=false, silent=false, importance=importance))
    # println(select)
    
    # # get unbiased coefficients
    # ols = get_beta_unbiased(y, X_sub, σ_X, μ_X, select)
    # β_unbiased = coef(ols)
    # # press = compute_press(β_unbiased, X_test.data[:, select], y_test)
    # # println("PRESS: $press")

    # plot([], [], label=nothing)
    # plot_y_yhat(X_sub_test, y_test, ols, select, X_labels[select])
end

# root_experiment = "../Inclined_deep"
# angles = [25, 25.5, 26, 26.5, 27, 27.5]
root_experiment = "../Mill"
angles = ["4_46", "5_35", "8_92"]  #"2_23", 

n_samples = 1000
# for a1 in angles
#     for a2 in angles
#         for a3 in angles
#             if a1 < a2 && a2 < a3
# for n_samples in exp10.([4.5, 5])
#     n_samples = Int(round(n_samples))
angles_sub = angles
main(root_experiment, n_samples, angles_sub, ensembleAverage=false, n_augment=2, use_convex=false, importance=0.001, intercept=true)
#             end
#         end
#     end
# end
 
