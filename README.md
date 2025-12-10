# Scaled Cut-based Nested Benders Decomposition (ScaledCutNBD)

This repository contains the implemetation of the experiments from:

> W. Romeijnders, N. Van der Laan, and S. Kang, **Benders decomposition with scaled cuts for multistage stochastic mixed-integer programs**, preprint, 2025. [[link]](https://optimization-online.org/?p=26876)

The code is released under the MIT License. If you use this code in your work, please cite the paper above.

## Overview

The code implements a Nested Benders Decomposition (NBD) algorithm with **scaled cuts** and parametric outer approximations for solving multistage stochastic mixed-integer programs. For full problem formulations and theoretical details, please refer to the paper.

An example script for the lot-sizing problem used in the paper is provided in `examples/run_ex_03.jl`. If you only need instance data for testing, refer to `examples/ex_03_data`. The data includes
- `num_branch`: Vector giving the number of branches at each stage.
- `num_item`: Vector giving the number of item types at each stage.
- `demand`: Vector containing realizations of the random demand.
- `holding_cost`: Vector containing holding costs for all stages and item types.
- `backlogging_cost`: Vector containing backlog costs for all stages and item types.
- `fixed_cost`: Vector containing fixed production costs for all item types.
- `production_capacity`: The production capacity, i.e., $C$.
- `inventory_capacity`: Vector giving inventory capacity at each stage; this serves as the upper bound on $s_{ni}$ and $b_{ni}$.

## Quick Setup Guide for Beginners

1. Clone the repository and instantiates the Julia environment.
```bash
git clone https://github.com/suminK/ScaledCutNBD.git
cd ScaledCutNBD
julia --project=. -e 'using Pkg; Pkg.instantiate()
```

2. Run the 1D example.
```bash
julia --project=. ScaledCutNBD/examples/ex_01_1D.jl 1
```
Try different values of `r` from 1 to 5. If you encounter numerical issues, we recommend increasing CGSP_TOL, e.g. (CGSP_TOL=1e-1).
