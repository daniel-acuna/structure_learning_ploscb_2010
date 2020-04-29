function [Q] = StructureLearningModel_MeanModel(alpha1,beta1,alpha2,beta2,...
  alpha3,beta3,pdfc_phi,s1,f1,s2,f2, H, df)
% StructureLearningModel(alpha1,beta1,alpha2,beta2,...
%  alpha3,beta3,pdfc_phi,s1,f1,s2,f2, H, df)
%
% StructureLearningModel computes the value of arms according to the
% structure learning model of Acuna & Schrater (NIPS 2008).
% The order of the algorithm is proportional to
% (1/24)(1+H)(2+H)(3+H)(4+H)~= H^4
%
% alpha1: prior on success arm 1 if independent
% beta1: prior on failure arm 1 if independent
% alpha2: prior on success arm 2 if independent
% beta2: prior on failure arm 2 if independent
% alpha3: prior on positive obs. that arm 1 is better than 2 if coupled
% beta3: prior on negative obs. that arm 1 is better thatn 2 if coupled
% pdfc_phi: prior on probability of coupling
% s1: number of success observed on arm 1
% f1: number of failures observed on arm 1
% s2: number of success observed on arm 2
% f2: number of failures observed on arm 2
% H: horizon used to compute values. 
% df: discount factor
% 
% pdfc_phi = double(pdfc_phi >= 0.5);
r1 = StructureLearningModel_R(alpha1, beta1, alpha2, beta2, alpha3, beta3, pdfc_phi, s1, f1, s2, f2, 1);
r2 = StructureLearningModel_R(alpha1, beta1, alpha2, beta2, alpha3, beta3, pdfc_phi, s1, f1, s2, f2, 2);
% 
% V = zeros(H,H,H,H);
% 
% for hs2=H:-1:0
%   for  hf2=(H-hs2):-1:0
%     for hs1=(H-hf2-hs2):-1:0
%       for hf1=(H-hs1-hf2-hs2):-1:0
%         r1=StructureLearningModel_R(alpha1, beta1, alpha2, beta2, alpha3, beta3, pdfc_phi, s1, f1, s2, f2, 1);
%         r2=StructureLearningModel_R(alpha1, beta1, alpha2, beta2, alpha3, beta3, pdfc_phi, s1, f1, s2, f2, 2);
%         if (hs2+hf2+hs1+hf1>=H)
%           V(hs2+1,hf2+1,hs1+1,hf1+1) = max(r1,r2);
%         else
%           V(hs2+1,hf2+1,hs1+1,hf1+1) = max(r1 + df*r1*V(hs2+1,hf2+1,hs1+2,hf1+1) + df*(1-r1)*V(hs2+1,hf2+1,hs1+1,hf1+2),...
%             r2 + df*r2*V(hs2+2,hf2+1,hs1+1,hf1+1) + df*(1-r2)*V(hs2+1,hf2+2,hs1+1,hf1+1));
%         end
%       end
%     end
%   end
% end

Q = [r1 + max(r1,r2)*(H-1);
     r2 + max(r1,r2)*(H-1)];
 
 
% r1=StructureLearningModel_R(alpha1, beta1, alpha2, beta2, alpha3, beta3, ...
%   pdfc_phi, s1, f1, s2, f2, 1);
% r2=StructureLearningModel_R(alpha1, beta1, alpha2, beta2, alpha3, beta3, ...
%   pdfc_phi, s1, f1, s2, f2, 2);
% Q = [r1 + df*r1*V(1,1,2,1) + df*(1-r1)*V(1,1,1,2);
%      r2 + df*r2*V(2,1,1,1) + df*(1-r2)*V(1,2,1,1)];