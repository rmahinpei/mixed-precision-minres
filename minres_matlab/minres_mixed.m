function [x, relres_approx, relres_true, itn, dot_prods] = ...
    minres_mixed(A, B, L1, U1, L2, U2, f, g, fp32_prod, fp32_solve)
%------------------------------------------------------------------
% Solves the saddle point system [A B'; B 0] [x1; x2] = [f; g] with MINRES.
% Code is based off of Michael Saunder's original MINRES implementation
%   but has been adapted to support saddle point systems and mixed precision
%   matrix-vector products and preconditioner solves
%
% Inputs:   
% A  = n x n symmetric. 
% B  = m x n matrix with rank(B) = m
% L1 = n x n lower triangular factor of M1 preconditioner
% U1 = n x n upper triangular factor of M1 preconditioner
% L2 = m x m lower triangular factor of M2 preconditioner
% U2 = m x m upper triangular factor of M2 preconditioner
% f  = n x 1 vector
% g  = m x 1 vector 
% fp32_prod = whether matvec prod should be carried out in single precision
% fp32_solve = whether precond solves should be carried out in single precision
%
% Outputs:
% x             = final estimate of the solution
% relres_approx = approximated relative residual of the preconditioned system for each iteration
% relres_true   = true relative residual for each iteration
% itn           = number of iterations until termination
% dot_prods     = vector of dot products between the two most recent
%                 Lanczos vectors at each iteration
%------------------------------------------------------------------
  % Initialize
  [m, n] = size(B); 
  b      = [f; g];
  bnorm  = norm(b);
  itnlim = 10;
  itn    = 0;
  rnorm  = bnorm;  
  rtol   = 1e-6;
  relres_true = zeros(itnlim, 1);
  relres_approx = zeros(itnlim, 1);
  dot_prods = zeros(itnlim, 1);
  if (fp32_solve == 1)
      L1 = single(full(L1));
      U1 = single(full(U1));
      L2 = single(full(L2));
      U2 = single(full(U2));
  end
  % need to keep both the fp64 and fp32 versions of A and B
  A_fp32 = single(full(A));
  B_fp32 = single(full(B));
  %------------------------------------------------------------------
  % Set up y and v for the first Lanczos vector v1.
  % y  =  beta1 P' v1,  where  P = C**(-1).
  % v is really P' v1.
  %------------------------------------------------------------------
  % We solve A*x = r0 and later set x = x0 + x.
  x   = zeros(n+m,1);   
  r1  = b;
  r2  = b;
  y   = opM(L1, U1, L2, U2, b);
  beta1  = b'*y;               beta1     = full(beta1);
  beta1  = sqrt(beta1);       % Normalize y to get v1 later.
  %------------------------------------------------------------------
  % Initialize other quantities.
  % ------------------------------------------------------------------
  oldb   = 0;       beta = beta1;   dbar = 0;       epsln = 0;
  phibar = beta1;   cs   = -1;      sn   = 0;
  rtol0  = rtol;  % Save original rtol. 
  w  = zeros(n+m,1);
  w2 = zeros(n+m,1);
  tic % start timing MINRES iterations
  %---------------------------------------------------------------------
  % Main iteration loop.
  % --------------------------------------------------------------------
    while itn < itnlim                  % k = itn = 1 first time through
      itn    = itn+1;
      %-----------------------------------------------------------------
      % Obtain quantities for the next Lanczos vector vk+1, k = 1, 2,...
      % The general iteration is similar to the case k = 1 with v0 = 0:
      %
      %   p1      = Operator * v1  -  beta1 * v0,
      %   alpha1  = v1'p1,
      %   q2      = p2  -  alpha1 * v1,
      %   beta2^2 = q2'q2,
      %   v2      = (1/beta2) q2.
      %
      % Again, y = betak P vk,  where  P = C**(-1).
      % .... more description needed.
      %-----------------------------------------------------------------
      s = 1/beta;                 % Normalize previous vector (in y).
      v = s*y;                    % v = vk if P = I
      old_v = v; % !!!
        
      if (fp32_prod == 0)
          y = opSaddle(A, B, v);
      else
          y = opSaddle(A_fp32, B_fp32, v);
      end
      if itn >= 2, y = -(beta/oldb)*r1 + y; end

      alfa   = full(v'*y);        % alphak
      y      = -(alfa/beta)*r2 + y;
      r1     = r2;
      r2     = y;
      y = opM(L1, U1, L2, U2, r2);
      oldb   = beta;              % oldb = betak
      beta   = full(r2'*y);       % beta = betak+1^2
      beta   = sqrt(beta);
      
      new_v = (1/beta)*y; % !!!
      dot_prods(itn) = dotprod(L1, L2, old_v, new_v); % !!!

      % Apply previous rotation Qk-1 to get
      %   [deltak epslnk+1] = [cs  sn][dbark    0   ]
      %   [gbar k dbar k+1]   [sn -cs][alfak betak+1].
      oldeps = epsln;
      delta  = cs*dbar + sn*alfa; % delta1 = 0         deltak
      gbar   = sn*dbar - cs*alfa; % gbar 1 = alfa1     gbar k
      epsln  =           sn*beta; % epsln2 = 0         epslnk+1
      dbar   =         - cs*beta; % dbar 2 = beta2     dbar k+1

      % Compute the next plane rotation Qk
      gamma  = norm([gbar beta]); % gammak
      gamma  = max([gamma eps]);
      cs     = gbar/gamma;        % ck
      sn     = beta/gamma;        % sk
      phi    = cs*phibar ;        % phik
      phibar = sn*phibar ;        % phibark+1

      % Update x and rnorm.
      w1     = w2;
      w2     = w;
      w      = (v - oldeps*w1 - delta*w2)*(1/gamma);
      x      = x + phi*w;
      rnorm  = phibar;

      % Save approx and true relres for current iteration
      relres_approx(itn) = rnorm / bnorm;
      relres_true(itn) = norm(opSaddle(A, B, x) - b) / bnorm;
      %================================================================
      % Check stopping condition.
      %================================================================
      if rnorm <= bnorm*rtol
         % Preconditioned system thinks it satisfied rtol.
         % See if Ax = b is satisfied to rtol0.
         rnormk = norm(b - opSaddle(A, B, x));
         if rnormk <= bnorm*rtol0
            break;
         else   % Reduce rtol
            rtol    = rtol/10;
         end
      end

    end % while itn < itnlim (main loop)
  timerVal = toc;
  fprintf('\nNumber of iterations: %d\n', itn)
  fprintf('\nTime of iterations %8.6f seconds\n', timerVal)

  relres_true = relres_true(1:itn);
  relres_approx = relres_approx(1:itn);
  dot_prods = dot_prods(1:itn);
end
%-----------------------------------------------------------------------
% End function minres
%-----------------------------------------------------------------------


function [y] = opM(L1, U1, L2, U2, x)
    x1 = x(1:size(L1,1)); 
    x2 = x(size(L1,1) + 1:end);
    y1 = U1\(L1\x1);
    y2 = U2\(L2\x2);
    y = [double(y1); double(y2)];
end
%-----------------------------------------------------------------------
% End private function opM
%-----------------------------------------------------------------------


function [w, r, s] = opSaddle(A, B, x)
% returns r = A*x1 + B'*x2; s = B*x1
    n = size(A,1);
    x1 = x(1:n); x2 = x(n+1:end);
    r = A*x1 + B'*x2;
    s = B*x1;
    w = [double(r); double(s)];
end
%-----------------------------------------------------------------------
% End private function opSaddle
%-----------------------------------------------------------------------

function [prod] = dotprod(R1, R2, old, new)
% returns dot prod between two consectuive Lanczos vectors
    n = size(R1,1);
    v00 = old(1:n); v01 = old(n+1:end);
    v_old = [R1'*v00 ; R2'*v01];

    v10 = new(1:n); v11 = new(n+1:end);
    v_new = [R1'*v10 ; R2'*v11];

    prod = abs(v_new'*v_old);
%     prod = abs(old'*new);
end
%-----------------------------------------------------------------------
% End private function dotprod
%-----------------------------------------------------------------------
