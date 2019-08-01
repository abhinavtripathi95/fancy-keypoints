%% plots histogrqam with theoretical density function
%
% x     = data vector
% pdf   = expected density
% npar  = number of parameters for density (0,1,2,3)
% A,B,C = parameters (possibly B or C is not set)
% 
% assumes figure with hold on is open.
%
% Wolfgang Förstner 3/2017
% wfoerstn@uni-bonn.de 

function sugr_plot_histogram_density(x,pdf,npar,A,B,C)

% number of samples
N = length(x);

%prepare plot

N_bin  = floor(sqrt(N));        % number of bins
[NN,r] = hist(x,N_bin);         % calculate histogram

%% plot histogram
bar(r,NN)                             
hold on
%% plot expected density 

% range of histogram
range = abs(r(N_bin)-r(1))*N_bin/(N_bin-1);                       
switch npar
    case 0
        plot(r,N_bin*range*pdf(r),'-r','LineWidth',4);
    case 1
        plot(r,N_bin*range*pdf(r,A),'-r','LineWidth',4);
    case 2
        plot(r,N_bin*range*pdf(r,A,B),'-r','LineWidth',4);
    case 3
        plot(r,N_bin*range*pdf(r,A,B,C),'-r','LineWidth',4);
end
xlim([r(1),r(N_bin)]);
ylim([0,max(NN)*1.1]);
xlabel('x');ylabel('\propto p_x(x)')
title(['Histogram of the sample with ',num2str(N_bin),' bins, overlayed with its probability density.'])



