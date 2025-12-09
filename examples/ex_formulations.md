## Formulation of problem in ex_01_1D.jl
"Stochastic Lipschitz dynamic programming," Ahmed et al. (2022)
\[
	\min_{x_n = (y_n,z_n)}\left\{\sum_{n \in \mathcal{N}} p_n \beta^{t_n-1}|y_n|: x_n \in X_n,\,\, y_n - y_{a(n)}-z_n = h_n,\quad\forall n \in \mathcal{N}\right\},
\]
with $X_n = \{(y_n,z_n) \in \R{} \times \{-1,1\}: |y_n| \leq M\}$, $\beta \in \R$ representing a discount factor, and with stagewise independent random parameters $\xi_t = h_t$, $t = 1,\ldots,T$. We let $N_t$ denote the realizations of $\xi_t$ in time stage $t = 1,\ldots,T$.