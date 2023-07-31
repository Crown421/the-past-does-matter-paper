# The Past Does Matter: Correlation of Subsequent States in Trajectory Predictions of Gaussian Process Models
### Steffen Ridderbusch, Sina Ober-Blöbaum, Paul James Goulart

This repo contains the code for the paper [The Past Does Matter: Correlation of Subsequent States in Trajectory Predictions of Gaussian Process Models](https://proceedings.mlr.press/v216/ridderbusch23a.html), published at UAI 2023. 

The method presented in this paper has been incorporated into the Julia package [GPDiffEq.jl](https://proceedings.mlr.press/v216/ridderbusch23a.html), which has also been used in this repo. 

## Requirements
To use the code in this repo, you need at least Julia 1.9, which can be downloaded [here](https://julialang.org/download/) or via [juliaup](https://github.com/JuliaLang/juliaup). 
Then, clone the repo and open Julia with the local project. 

```
git clone git@github.com:Crown421/the-past-does-matter-paper.git
cd the-past-does-matter-paper
julia --project
```
From here, install all required packages via the package manager (accessed by pressing `]`):
```julia
julia>]
(the-past-does-matter-paper) pkg> instantiate
```

## Contents
This repo contains two notebooks and two scripts. 
- `Linear-Uncertainty.ipynb`: This notebooks contains the code and plots for the linear prototype shown in section 3 of this paper, including both the plots for the paper and the poster. 
- `Local_Lin.ipynb`:
