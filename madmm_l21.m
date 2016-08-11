function madmm_l21(x0,functions,params)

% set reg term: g(x) = |X|_2,1 (sum of norm over columns)
functions.fun_g = @(Z)sum(sqrt(sum(Z.^2,1)));


T = 10; %number of iterations

% set the manifold type
problem.M = params.manifold;
options.verbosity = 0;

% set the "shrinkage" parameter
c = params.lambda / params.rho;


X = x0;
Z = X;
U = zeros(size(Z));

original_cost = @(x)functions.fun_f(x) + params.lambda*functions.fun_g(x)
keep_cost = zeros(T,1);

if params.is_plot, fig = animatedline; title('cost'); end

for step = 1:T   
    
    keep_cost(step) = original_cost(X);
    
    if params.is_plot, addpoints(fig,step,keep_cost(step)); pause(1); end

    
    disp(['cost: ' num2str(keep_cost(step))]);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % X-step - solve a smooth minimization problem on the manifold    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    disp('Strating step X');
%     % set auxilary function h(x) = rho*|Z - V|_F^2
%     
%     
%     fun_h = @(Z,X,U)0.5*rho*sum( sum( (Z - fun_v(X,U)).^2 ) );
%     dfun_h = @(Z,X,U)rho*( fun_v(X,U) - Z  );

    problem.cost = @(x)functions.fun_f(x) + params.rho*functions.fun_h(x,Z,U)
    problem.egrad = @(x)functions.dfun_f(x) + params.rho*functions.dhdx(x,Z,U)
%     checkgradient(problem);

    
    
    
    
    X = conjugategradient(problem,X,options);
    disp('Finished step X.');
    disp(['cost: ' num2str(original_cost(X))]);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Z-step     
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    disp(['Step Z. Cost before: ' num2str(params.lambda*functions.fun_g(Z) + params.rho*functions.fun_h(X,Z,U) )]);
    Z = prox_l21(X+U,c);    
    disp(['Cost After: ' num2str(params.lambda*functions.fun_g(Z) + params.rho*functions.fun_h(X,Z,U) )]);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % U-update 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    U = U + X - Z;
    
    
end


if params.is_plot, figure, subplot(121), imagesc(x0), subplot(122), imagesc(X); colormap; end


end

