function [ B,elapsed_time ] = netConf( adj_mat,H,priors,varargin )
% A - N*N adjacency matrix (typically sparse)
% H - k*k modulation matrix
% E - N*k priors (set to 1/k..1/k for unseeded nodes)
%% OPTIONAL ARGUMENTS
% ep - ep/rho(A) is the modulation multiplier, default 0.5
% stop - 0 -> till convergence; +ve no -> fixed #iterations
% verbose - 0 (just #iterations, defa1 (pult) or er iteration stats)
% max_iter - maximum iterations, default 100
%% process arguments
numvarargs = length(varargin);
if numvarargs > 4,
    error('netConf:TooManyInputs', ...
        'requires at most 4 optional inputs');
end
optargs = {0.5, 0, 0, 100}; optargs(1:numvarargs) = varargin;
[ep,stop,verbose,max_iter] = optargs{:};
%% algo
v = abs(eigs(adj_mat,1));
[N,k] = size(priors);
B = priors;
l = zeros(N,1);
M = ep/v*H;
M1 = M/(eye(k)-M*M);
M2 = M*M1;
D = diag(sum(adj_mat)); diff1 = 1;
if verbose == 1,
    fprintf('It\tmax{del(B)}\tdel(label)\n');
end
if stop==0,
    n_iter = max_iter;
else
    n_iter = stop;
end
tic;
for i=1:n_iter
    if stop == 0 && diff1<1e-4,
        if verbose == 0,
            fprintf('%d iterations\n',i);
        end
        break
    end
    Bold = B; lold = l;
    B = priors + adj_mat*B*M1 - D*B*M2;
    [~,l] = max(B,[],2);
    diff2 = sum(lold~=l);
    diff1 = max(max(abs(B-Bold)));
    if verbose == 1,
        fprintf('%d\t%f\t%d\n',i,diff1,diff2);
    end
end
elapsed_time = toc;
end