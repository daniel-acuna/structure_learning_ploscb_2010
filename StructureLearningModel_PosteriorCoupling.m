function [pdfc_phi] = StructureLearningModel_PosteriorCoupling(prior_pdfc_phi, alpha1, beta1,...
  alpha2, beta2, alpha3, beta3, s1,f1,s2,f2)
% Computes the posterior p(c=1) given the priors over theta_i, with
% parameters alpha_j, and beta_j, for 1<=j<=3, and the rewards observed
% (s_j successes and f_j failures on arm j, with 1<=j<=2).


pdfc_phi = [
      ((1-prior_pdfc_phi)*beta(alpha1+s1,beta1+f1)*beta(alpha2+s2,beta2+f2))/...
      (beta(alpha1,beta1)*beta(alpha2,beta2))
    
    (prior_pdfc_phi*beta(alpha3+s1+f2,beta3+f1+s2))/...
      (beta(alpha3,beta3))];
pdfc_phi = pdfc_phi(2)/sum(pdfc_phi);
% pdfc_phi = ...
%     1/(1 - ((-1 + prior_pdfc_phi)*beta(alpha1 + s1,beta1 + f1))/(prior_pdfc_phi*beta(alpha1 + beta2 + f2 + s1,alpha2 + beta1 + f1 + s2)));

% pdfc_phi = ...
%     1/(1 - ((-1 + prior_pdfc_phi)*beta(alpha1 + s1,beta1 + f1))/(prior_pdfc_phi*beta(alpha1 + beta2 + s2 + s1,alpha2 + beta1 + f1 + f2)));
% pdfc_phi = pdfc_phi/sum(pdfc_phi);