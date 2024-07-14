% uncomment for Maxwell matrices
% g = zeros(size_m, 1);
[x, relres_approx, relres_true, itn, dot_prods] = ...
    minres_mixed(A, B, R1', R1, R2', R2, f, g, 0, 0);     