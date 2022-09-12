using Plots
using HDF5
using NaNStatistics
using TensorDecompositions
using Random
using Combinatorics
using LinearAlgebra
using DelimitedFiles
using StatsBase
using Glob

Random.seed!(42)

# plot backend
gr()         # or gr() plotlyjs() pyplot()

#get PSTH function
include("trials_extract.jl")
include("similarity_tca.jl")

# choose to perform regular Canonical polyadic decomposition (CANDECOMP/PARAFAC) or 
# Non-negative CANDECOMP/PARAFAC
do_non_negative = false

# choose to perform optimization (time consuming step)
do_opt = true



# get NWB files
fileloc = "/Volumes/labs/dmclab/Pierre/NPX_Database/mPFC/Aversion/"
filelist = glob("*.nwb",fileloc)


#f = filelist[1]

for f in filelist

    filename = split(f, "/")[end]
    println("Processing "*filename)
    prefix = split(filename, ".")[1]
    nwb = h5open(f, "r")

    # Get units spike times, Load jagged arrays
    unit_times_data = read(nwb["units/spike_times"]);
    unit_times_idx = read(nwb["units/spike_times_index"]);
    pushfirst!(unit_times_idx,1);
    unit_ids = read(nwb["units/id"]);
    spk_times = [unit_times_data[unit_times_idx[i]:unit_times_idx[i+1]] for i in 1:length(unit_ids)];

    # Get trials timestamps
    trial_ts = read(nwb["intervals/trials/start_time"])

    # Output folder for optimization
    Destfolder = "/Users/pielem/Desktop/ANALYSIS/mPFC_Passive_Neuropixels/TCA/"
    csv_err = Destfolder*"model_optimization/"*prefix*"_err.csv"
    csv_sim = Destfolder*"model_optimization/"*prefix*"_sim.csv"

    # extract trials and build tensor
    rec_dur = maximum(maximum(spk_times))
    t = [trials_extract(spk_times[i],trial_ts,[-1 3],0.01) for i in 1:length(unit_ids)];
    tensor = cat(t...,dims=3);
    T = permutedims(tensor, (3, 1, 2));
    T = Float64.(T)

    ##############################################################################################################################################################################
    # Optimization
    if do_opt       
        max_number_of_components = 10 #number of components
        iterations = 10 #number of iterations per component

        # if file already exist do nothing
        if isfile(csv_err)

            err = readdlm(csv_err,'\t')
            sim = readdlm(csv_sim,'\t')
            println("Optimization already done")

        else  # if file doesn't exist run otpimization 

            all_F = Array{Any}(undef, iterations)
            err = Array{Any}(undef, max_number_of_components, iterations)
            sim = Array{Any}(undef, max_number_of_components, length(collect(combinations(1:iterations, 2))))
            combs = collect(combinations(1:iterations, 2))
            # parameter optimization
                Threads.@threads for r in 1:max_number_of_components
                    # randomized initialisation
                    #initial_guess = ntuple(k -> randn(size(T, k), r), ndims(T)) 
                    # run tensor decomposition, shut down verbose and progress bar
                    if do_non_negative
                        all_F = [nncp(T, number_of_components, compute_error=true, maxiter=200) for ii = 1:iterations]
                    else
                        all_F = [candecomp(T, r, ntuple(k -> randn(size(T, k), r), ndims(T)), compute_error=true, method=:ALS, maxiter=200, verbose=false) for ii = 1:iterations]
                    end
                    # collect errors
                    err[r,:] = [collect(values(all_F[ii].props))[1] for ii = 1:iterations]
                    sim[r,:] = [similarity_tca(all_F[combs[c][1]],all_F[combs[c][2]]) for c in 1:length(combs)]
                end

            # save optimization parameters
            writedlm(csv_err, err)
            writedlm(csv_sim, sim)
            println("Optimization done")    
        end


        # Plot optimization results
        p = plot(layout = (1, 2))
        plot!(mean(err,dims=2), color="black", xlabel="components", ylabel = "error", subplot=1,legend = false)
        [scatter!(1:max_number_of_components, err[:,ii], color=RGBA(0,0,0,1), markerstrokewidth=0,subplot=1) for ii in 1:iterations]
        plot!(mean(sim,dims=2), color="black", xlabel="components", ylabel = "similarity",  subplot=2,legend = false)
        [scatter!(1:max_number_of_components, sim[:,ii], color=RGBA(0,0,0,1), markerstrokewidth=0,subplot=2) for ii in 1:iterations]
        plot!(size=(750,400), plot_title = "Optimization results")
        display(p)
        savefig(p,Destfolder*prefix*"_optimization.svg")
        savefig(p,Destfolder*prefix*"_optimization.png")

    end

    ##############################################################################################################################################################################
    # Run TCA for optimized parameters
    number_of_components = 10 #number of components
    if do_non_negative
        F = nncp(T, number_of_components, compute_error=true, maxiter=200)
    else
        F = candecomp(T, number_of_components, ntuple(k -> randn(size(T, k), number_of_components), ndims(T)), compute_error=true, method=:ALS, maxiter=200) 
    end 
    
    
    # plot
    #lambdas_order = sortperm(F.lambdas)
    lambdas_order = 1:10
    #weights_order = sortperm(F.factors[1][:,findmin(lambdas_order)[2]])
    weights_order = 1:size(F.factors[1],1)
    p2 = plot(layout = (number_of_components, 3))
    for ii in 1:number_of_components
        if lambdas_order[ii]==number_of_components
            bar!(F.factors[1][weights_order,ii], color="black", xlabel="units", subplot=lambdas_order[ii]+2*(lambdas_order[ii]-1), legend = false)
            plot!(F.factors[2][:,ii], color="red", xlabel="time bins", subplot=lambdas_order[ii]+2*(lambdas_order[ii]-1)+1, legend = false)
            plot!(F.factors[3][:,ii], color="green", xlabel="trials", subplot=lambdas_order[ii]+2*(lambdas_order[ii]-1)+2, legend = false)    
        else    
            bar!(F.factors[1][weights_order,ii], color="black", subplot=lambdas_order[ii]+2*(lambdas_order[ii]-1), legend = false)
            plot!(F.factors[2][:,ii], color="red", subplot=lambdas_order[ii]+2*(lambdas_order[ii]-1)+1, legend = false)
            plot!(F.factors[3][:,ii], color="green", subplot=lambdas_order[ii]+2*(lambdas_order[ii]-1)+2, legend = false)
        end
    end
    plot!(size=(800,1000))
    display(p2)
    if do_non_negative
        savefig(p2,Destfolder*prefix*"_nnTCA_results.svg")
        savefig(p2,Destfolder*prefix*"_nnTCA_results.png")
    else
        savefig(p2,Destfolder*prefix*"_TCA_results.svg")
        savefig(p2,Destfolder*prefix*"_TCA_results.png")
    end

    # save
    if do_non_negative
        csv_neurons = Destfolder*prefix*"_units_nnTCA.csv"
        csv_temporal = Destfolder*prefix*"_timebins_nnTCA.csv"
        csv_trial = Destfolder*prefix*"_trials_nnTCA.csv"
    else
        csv_neurons = Destfolder*prefix*"_units_TCA.csv"
        csv_temporal = Destfolder*prefix*"_timebins_TCA.csv"
        csv_trial = Destfolder*prefix*"_trials_TCA.csv"
    end
    writedlm(csv_neurons, F.factors[1])
    writedlm(csv_temporal, F.factors[2])
    writedlm(csv_trial, F.factors[3])

end