using StatsPlots, LaTeXStrings, Plots, DataFrames


function make_plot(y, labels, title)
    p = scatter(y, markersize=5, linewidth=0, legend=false, tickfontsize=5)
    title!(title)
    xticks!(1:length(labels), labels)
    return p
end

function make_plot_boot(y, labels, n_resamples=500)
    n_vars = length(labels)
    violin(repeat(1:n_vars, outer=n_resamples), y[:], legend=false, fill=true, trim=true, tickfontsize=8)
    title!(L"\mu = \sum ...")
    xticks!(1:n_vars, labels)
    xlims!(.5, n_vars+.5)
end

"After bootstrap, see what proportion are non-zero."
function selection_frequency(β; eps = 1e-8)
    n_stats, n_samples = size(β)
    freq = zeros(n_stats)
    for (i, stat) in enumerate(eachrow(β))
        freq[i] = sum(abs.(stat).>eps) / n_samples
    end
    return freq
end

function plot_covariance(X, fp)
    gr(size=(1200,900))
    println(X.labels[3])
    @show(cor(X.data)[3, :])
    cornerplot(X.data, label = X.labels, compact=true)
    # savefig(fp)
end

function plot_covariance(X, select, fp)
    X_subset = SampleStatistics(X.data[:, select], X.labels[select])
    return plot_covariance(X_subset, fp)
end

# plot y vs y_hat
function plot_y_yhat(X, y, ols, select, model_label; model_colour=nothing)
    df = DataFrame(X[:, select], :auto)
    y_pred = GLM.predict(ols, df)
    if model_colour === nothing
        scatter!(vec(y), y_pred, label=model_label, markersize=1, markerstrokewidth=0, legend=:right, alpha=0.5)
    else
        scatter!(vec(y), y_pred, label=model_label, markersize=1, markerstrokewidth=0, legend=:right, color=model_colour, alpha=0.5)
    end
    ylabel!(L"\hat{y}")
    xlabel!(L"y")
    # title!("Performance on test data")
end


# # plot
# y_test_select = DataFrame(X = X_test.data[:, select])
# y_pred = GLM.predict(ols, y_test_select)
# if n_terms_target > 1
#     scatter(y_test, y_pred, legend=false)  # markercolor=colors[n_augment+1]
# else
#     display(scatter!(y_test, y_pred, legend=false))  # markercolor=colors[n_augment+1]
# end
# # display(plot(y, y, legend=false))
# # display(scatter!(X_subset, y_pred))


function glm_plot(X, y, X_label, y_label, filename; file_extension = ".pdf")
    X_vec = vec(X)
    y_vec = vec(y)
    
    # linear fit y ~ X
    data = DataFrame(y=y_vec, X=X_vec)
    ols = lm(@formula(y ~ X), data)
    println(ols)
    bic = 1 * log(length(y_vec)) - 2 * loglikelihood(ols)
    println(bic)

    # plot
    y_pred = GLM.predict(ols, DataFrame(X=X_vec))
    plot(X_vec, y_pred, legend=false)
    scatter!(X_vec, y_vec)
    xlabel!(X_label)
    ylabel!(y_label)

    # save figure
    savefig("../Inclined_deep/results/" * filename * file_extension)
    open("../Inclined_deep/results/" * filename * ".txt", "w") do io
        println(io, ols)
     end
end

function glm_plot(X1, X2, y, X1_label, X2_label, y_label, filename; file_extension = ".pdf")
    # linear fit y ~ X1 + X2
    y_vec = vec(y)
    X1_vec = vec(X1)
    X2_vec = vec(X2)
    data = DataFrame(y=vec(y), X1=X1_vec, X2=X2_vec)
    ols = lm(@formula(y ~ X1 + X2), data)
    println(ols)

    # plot
    y_pred = GLM.predict(ols, DataFrame(X1=X1_vec, X2=X2_vec))
    scatter(y_pred, vec(y), legend=false)
    xlabel!(L"\hat{y}")
    ylabel!(y_label)
    title!(y_label * "~" * X1_label * "+" * X2_label)
    
    # save figure
    savefig("../Inclined_deep/results/" * filename * file_extension)
    open("../Inclined_deep/results/" * filename * ".txt", "w") do io
        println(io, ols)
    end
end
