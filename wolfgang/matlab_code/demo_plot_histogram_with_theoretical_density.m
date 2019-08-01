%% plot histogram with theoretical density function

close all

addpath(genpath('../General-Functions'))

% default parameters
A=0;
B=0;
C=0;

% Number of generated samples
N=1000

%distribution --- uncomment unused alternatives
 
distribution = 'f'
density      = @fpdf
npar=2
A=53;
B=1000000;
% 
% distribution = 'normal';
% density      = @normpdf
% npar=2
% A=3;
% B=4;
% 
% distribution = 'chi2';
% density      = @chi2pdf
% npar=1
% A=5;

% take sample
switch npar
    case 1
        x=random(distribution,A,N,1);
    case 2
        x=random(distribution,A,B,N,1);
end

figure
hold on

sugr_plot_histogram_density(x,density,npar,A,B,C)

