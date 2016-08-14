%% Run this script to test the L2,1 MADMM function with the simple setting where:
% X = argmin_X f(X) + lambda*|v(X)|_2,1, 
% where: f(X) = 0.5*|AX-B|_F^2
% v(X) = FX
% In case you change the functions f(x) or v(X) I strongly recommend checking the
% gradients by uncommenting the "check grad" code snippets.

clear all;close all;
%% Dependencies
addpath(genpath('./../../manopt/'))

%% params:
N = 10; % num rows
M = 10; % num cols
lambda = 5;
rho = 20;
rng(42);
%% set a simple data term: f(x) = 0.5*|AX-B|_F^2
A = rand(N);
B = rand(N,M);

functions.fun_f = @(X)0.5*sum( sum( (A*X - B).^2 ) );
functions.dfun_f = @(X)A'*(A*X - B)

%% check grad
% problem.M = euclideanfactory(N, M);
% problem.cost = functions.fun_f;
% problem.egrad = functions.dfun_f;
% x0 = rand(N,M);
% checkgradient(problem);
%%
F = rand(N,N);
functions.fun_v = @(X)F*X;
functions.dfun_v = @(X)F';

%% set the l2 term for the Z parameter replacement
functions.fun_h = @(X,Z,U)0.5*sum( sum( ( Z - functions.fun_v(X) - U).^2 ) );  
functions.dhdx = @(X,Z,U)functions.dfun_v(X) * (functions.fun_v(X)+U-Z)
% functions.dhdz = @(X,Z,U)Z - functions.fun_v(X) - U

%% check grad
% Z = rand(N,M);
% U = rand(N,M);
% problem.M = euclideanfactory(N, M);
% problem.cost = @(X)fun_h(Z,X,U);
% problem.egrad = @(X)dfun_h(Z,X,U);
% x0 = rand(N,M);
% checkgradient(problem);

%% run the madmm_l21 function
x0 = eye(N);%rand(N,M);
params.lambda = lambda;
params.rho = rho;
params.manifold = stiefelfactory(N, M);
params.is_plot = 1;
params.max_iter = 100;
X_out = madmm_l21(x0,functions,params);

%% show result
figure, subplot(121); imagesc(F*x0); subplot(122); imagesc(F*X_out)


