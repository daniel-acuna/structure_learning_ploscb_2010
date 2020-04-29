function [pdftheta1 pdftheta2 pdftheta3 pdfc_phi] = ...
  StructureLearningModel_FullPosterior(s1, f1, s2, f2, alpha1, beta1, alpha2, beta2, ...
  alpha3, beta3, pdfc_phi, gd)
% This model computes the numerical marginal distributions over p(theta1),
% p(theta2),p(theta3), p(c) after observing s1 successes and f1 failures on
% arm 1, and s2 successes and f2 failures on arm 2.
% gd is the size of grid

pdf_theta1_theta2_theta3_c = zeros(gd+1,gd+1,gd+1,2);

%K = crmodel3_evaluate_normalization(alpha1,beta1,...
%  alpha2,beta2,alpha3,beta3);


for ic=1:2
  c=ic-1;
  for itheta1=1:(gd+1)
    theta1=(itheta1-1)/(gd+1);
    for itheta2=1:(gd+1)
      theta2=(itheta2-1)/(gd+1);
      for itheta3=1:(gd+1)
        theta3=(itheta3-1)/(gd+1);
        pdf_theta1_theta2_theta3_c(itheta1,itheta2,itheta3,ic) = ...
          crmodel3_evaluate_posterior(...
          s1,f1,s2,f2,alpha1,beta1,alpha2,beta2,alpha3,beta3,pdfc_phi,...
          theta1,theta2,theta3,c);
      end
    end
  end
end


dom = 0:(1/gd):1;

% Integrate out each distribution
pdfc_phi = trapz(dom,trapz(dom,trapz(dom, ...
  pdf_theta1_theta2_theta3_c,1), 2),3);
pdfc_phi = reshape(pdfc_phi, [2 1]);
pdfc_phi = pdfc_phi(2)/sum(pdfc_phi);

% Computing pdftheta1
pdftheta1 = trapz(dom, ...
  trapz(dom,sum(pdf_theta1_theta2_theta3_c,4),2),3);
pdftheta1 = reshape(pdftheta1, [1 gd+1]);
% Normalazing pdftheta1
K = trapz(dom, pdftheta1);
pdftheta1 = (pdftheta1/K)';

% Computing pdftheta2
pdftheta2 = trapz(dom, ...
  trapz(dom,sum(pdf_theta1_theta2_theta3_c,4),1),3);
pdftheta2 = reshape(pdftheta2, [1 gd+1]);
% Normalazing pdftheta2
K = trapz(dom, pdftheta2);
pdftheta2 = (pdftheta2/K)';

% Computing pdftheta3
pdftheta3 = trapz(dom, ...
  trapz(dom,sum(pdf_theta1_theta2_theta3_c,4),1),2);
pdftheta3 = reshape(pdftheta3, [1 gd+1]);
% Normalazing pdftheta2
K = trapz(dom, pdftheta3);
pdftheta3 = (pdftheta3/K)';

end

function [prob] = crmodel3_evaluate_posterior(s1, f1, s2, f2, ...,
  alpha1, beta1, alpha2,  beta2, alpha3, beta3, pdfc_phi, ...
  theta1, theta2, theta3, c)

if c==0
  prob = (1-pdfc_phi)*...
    (theta1)^(alpha1-1+s1)*(1-theta1)^(beta1-1+f1)*...
    (theta2)^(alpha2-1+s2)*(1-theta2)^(beta2-1+f2)*...
    (theta3)^(alpha3-1)*(1-theta3)^(beta3-1);
else
  prob = pdfc_phi*...
    (theta1)^(alpha1-1)*(1-theta1)^(beta1-1)*...
    (theta2)^(alpha2-1)*(1-theta2)^(beta2-1)*...
    (theta3)^(alpha3-1+s1+f2)*(1-theta3)^(beta3-1+f1+s2);
end
end

% function [K] = ...
%   crmodel3_evaluate_normalization(alpha1, beta1, alpha2, ...
%   beta2, alpha3, beta3)
% 
% % Normalization constant
% K = 1/(beta(alpha1,beta1)*beta(alpha2,beta2)*beta(alpha3,beta3));
% end