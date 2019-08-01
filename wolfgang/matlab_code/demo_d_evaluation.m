%% demo evaluation of distances
%
% distances from points with the same CovM
%                            different CovM
%
% measure N(d < T_d)
%
% wf 2019/06/04

display('-----------------------------------')
display('----- demo evaluate distances -----')
display('-----------------------------------')

clearvars
close all

type_distance = 0;  % Euclidean distance
%type_distance = 1;  % Euclidean distance/sigma_D
type_distance = 2;  % Euclidean distance/sigma_D/sigma_d_i
%type_distance = 3;  % Mahalanobis

Ps = [0.90,0.05,0.99,0.999];
Ts = [0.5,1,1.5,2];
%% generate data
%
% varying CovM =sigma_D^2 * f * CovM0(alpha,dV), f in [0.3..1]
%
N_samples = 1000     % number of keypoints
%
sigma_D = 0.5        % mean std of keypoint coordinates
                     % = 1 if precision = 1 pixel
                     
sigma_dlambda = 1  % std of varaince of eigenvalues of CoVM (anisotropy)
                     % = 0 then isotropy
%
f_min = 0.5          % minimum factor for stdev of keypoints
                     % if 1.0 then no variation
f_max = 1/f_min      % maximum = 1/minimum

% simulated coordinate differences
D=zeros(N_samples,2);
sigma=zeros(N_samples,1);
for i=1:N_samples
    sigma(i) = rand(1)*(f_max-f_min)+f_min; % random factor for CovM
    alpha    = rand(1)*pi; % random rotation
    R = [cos(alpha) -sin(alpha);sin(alpha) cos(alpha)];
    d_lambda = exp(randn(1)*sigma_dlambda); % random variance ratio
    V = diag([d_lambda,1/d_lambda]);    
    CovM0 = R*V*R';
    D(i,:)   = rand_gauss([0;0],sigma_D^2*sigma(i)^2*CovM0,1)'; 
    C(i,:) = reshape(sigma_D^2*sigma(i)^2*CovM0,1,4);
end

%% evaluate differences
for type_distance=0:3

% derive distances
d=zeros(N_samples,1);
switch type_distance
    case 0
        d = sqrt(diag(D*D'));
        d_type = 'Euclidean';
    case 1
        d = sqrt(diag(D*D'))/sigma_D;
        d_type = 'Euclidean/sigma_D';
    case 2      
        d = sqrt(diag(D*D'))/sigma_D./sigma;
        d_type = 'Euclidean/sigma_D/sigma_i';
    case 3
        for i=1:N_samples
            d(i) = sqrt(D(i,:)*inv(reshape(C(i,:),2,2))*D(i,:)');
        end
        d_type = 'Mahalanobis';
end

[NPs,PNs] = evaluate_distances(d,Ps,Ts,d_type);

Thresholds_Porbabilities=[Ps;NPs]
Probabilities_Thresholds=[Ts;PNs]
end
