function [r] = StructureLearningModel_R(alpha1, beta1, alpha2, beta2, ...
  ~, ~, pdfc_phi, s1, f1, s2, f2, z)
% Compute the expected reward for belief state pdftheta1, pdftheta2,
% pdfc_phi for pulling z

if (z==1)
    r= -(((-1 + pdfc_phi)*(alpha1 + s1))/(alpha1 + beta1 + f1 + s1)) + (pdfc_phi*(alpha1 + beta2 + s2 + s1))/(alpha1 + alpha2 + beta1 + beta2 + f1 + f2 + s1 + s2);
else
    r = -(((-1 + pdfc_phi)*(alpha2 + s2))/(alpha2 + beta2 + f2 + s2)) + (pdfc_phi*(alpha2 + beta1 + s1 + s2))/(alpha1 + alpha2 + beta1 + beta2 + f1 + f2 + s1 + s2);
end