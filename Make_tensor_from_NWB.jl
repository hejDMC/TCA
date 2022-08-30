using Plots
using HDF5
using NaNStatistics
using TensorDecompositions
using Random

Random.seed!(42)

# plot backend
gr()         # or gr() plotlyjs() pyplot()

#get PSTH function
include("trials_extract.jl")

# get NWB files
#fileloc = "/Volumes/labs/dmclab/Pierre/NPX_Database/mPFC/Passive/"
#filelist = glob("*.nwb",fileloc)

f = "/Volumes/labs/dmclab/Pierre/NPX_Database/mPFC/Passive/PL026_20190430-probe0.nwb"
nwb = h5open(f, "r")

# Get units spike times, Load jagged arrays
unit_times_data = read(nwb["units/spike_times"]);
unit_times_idx = read(nwb["units/spike_times_index"]);
pushfirst!(unit_times_idx,1);
unit_ids = read(nwb["units/id"]);
spk_times = [unit_times_data[unit_times_idx[i]:unit_times_idx[i+1]] for i in 1:length(unit_ids)];

# Get trials timestamps
trial_ts = read(nwb["intervals/trials/start_time"])

# list comprehension extract trials and build tensor
rec_dur = maximum(maximum(spk_times))
t = [trials_extract(spk_times[i],trial_ts,[-1 3],0.1) for i in 1:length(unit_ids)];
tensor = cat(t...,dims=3);
T = permutedims(tensor, (3, 1, 2));
T = Float64.(T)

# run tensor decomposition
r = 10 #number of components
initial_guess = ntuple(k -> randn(size(T, k), r), ndims(T))
F = candecomp(T, r, initial_guess, compute_error=true, method=:ALS);

F.props
F.factors[1]

# plot
plot(layout = (r, 3))
[bar!(F.factors[1][:,ii], color="black", subplot=ii+2*(ii-1), legend = false) for ii = 1:r]
[plot!(F.factors[2][:,ii], color="red", subplot=ii+2*(ii-1)+1, legend = false) for ii = 1:r]
[plot!(F.factors[3][:,ii], color="green", subplot=ii+2*(ii-1)+2, legend = false) for ii = 1:r]

plot!(size=(750,750))
