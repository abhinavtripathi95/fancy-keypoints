# wolfgang explanation of matlab code

enclosed a simulation study for the different measures.

N_samples of coordinate differences are generated according to some sepcifyable CovM.
- average sigma_D, indicating the standard deviation is not 1 (pixel).
- stdev for a factor for sigma_D (indicating that the keypoints may have different precision)
- stdev for the anisotropy of the CovM, indicating that some points may have an anisotropic CovM

Then the four different distances d 

- type_distance = 0;  % Euclidean distance
- type_distance = 1;  % Euclidean distance/sigma_D
- type_distance = 2;  % Euclidean distance/sigma_D/sigma_d_i
- type_distance = 3;  % Mahalanobis distance

are analysed, wrt to their histogram and probability being below T_d and, 
for given probabilities and threholds the threshold and probabilities are determined: 

- P(T_d) = Prob(d < T_d)   for a set of T_d's

and

- T_d(P)  such that Prob(d < T_d) = P  for a set of P's

which may be used in the tables.

The histograms show, that if the CovM vary a lot, they are not smooth and do not follow a Rayleigh-distribution.