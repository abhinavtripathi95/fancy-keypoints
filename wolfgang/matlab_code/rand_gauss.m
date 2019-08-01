%% random numbers following a multivariate Gaussian  with
% mean vector mu and covariance matrix C, considering correlations and
% potential singularities in C
%
% ret = rand_gauss(mu, C, n)
% 
% mu  = double dx1, mean vector of dimension d
% C   = double dxd, covariance matrix
% n   = int, number of samples to generate
% ret = double dxn, n samples of Gaussian(Mu,C)
%
% author: Richard Steffen

function ret = rand_gauss(mu, C, n)
    
% Zerlegung in eigenvectoren und Eigenwerte
[R,S] = eig(full(C));   
% Eigenvektoren skalieren mit Eigenwerten
A = R .* repmat(sqrt(abs(diag(S)))',length(S),1);
ret = repmat(mu,1,n) + A*randn(length(mu),n);
  
