# Tensor Component Analysis

Inspired by [Williams et al., Neuron 2018](https://www.sciencedirect.com/science/article/pii/S0896627318303878?via%3Dihub).

Builds on the [TensorDecompositions.jl](https://github.com/yunjhongwu/TensorDecompositions.jl) package by yunjhongwu.

Takes as an input an NWB file with single unit spiking data and trial information. Performs the tensor decomposition of a N units x T time bins x K trials tensor using a canonical polyadic decomposition (CANDECOMP/PARAFAC). Optimize the number of components according to [Williams et al., Neuron 2018](https://www.sciencedirect.com/science/article/pii/S0896627318303878?via%3Dihub) using the reconstruction error and the model similarity. Plots the data for the optimal number of components.
