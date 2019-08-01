%% evaluate_distances(d)
%
% evaluate_distances(d)
%
% 1. plot histogram of d's (with adapted Rayleigh)
% 2. plot Prob(d < d0)
% 3. determine thresholds d0(P) such that Prob(d<d0) = P

function[d_Ps,Ns_rel]=evaluate_distances(d,Ps,Ts,d_type)

mean_dq = mean(d.^2); % mean of chi^2-distribution
factor = sqrt( mean_dq/2 ); % parameter of best fitting Rayleigh

N = length(d);
dsort = sort(d);
figure
plot_histogram_density(d,@raylpdf,d_type,1,factor)


figure
plot(dsort,(1:N)/N,'-b')

n_Ps = length(Ps);
for n=1:n_Ps
    d_Ps(n) = kth_element(dsort,N,round(N*Ps(n)));
end

title([d_type,': P(d<T_d)'])
xlabel('T_d')

for n = 1:length(Ts)
    ind = find(d < Ts(n)*ones(N,1));
    Ns_rel(n) = length(ind)/N;
end
end