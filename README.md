# Scaled Cut-based Nested Benders Decomposition (ScaledCutNBD)

This repository contains the implemetation of the experiments from:

> W. Romeijnders, N. Van der Laan, and S. Kang, **Benders decomposition with scaled cuts for multistage stochastic mixed-integer programs**, preprint, 2025.

The code is released under the MIT License. If you use this code in your work, please cite the paper above.

## Overview

The code implements a Nested Benders Decomposition (NBD) algorithm with **scaled cuts** and parametric outer approximations for solving multistage stochastic mixed-integer programs.

For full problem formulations and theoretical details, please refer to the paper.

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
