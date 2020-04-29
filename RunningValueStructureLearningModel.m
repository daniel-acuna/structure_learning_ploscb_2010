function [V] = RunningValueStructureLearningModel(m, A, R, pdfc_phi, gamma, H)
% RunningValueCRunningValueStructureLearningModeloupledArms(m, A, R,
% pdfc_phi. gamma, H)
%
% Returns the value assigned to each arm at time t after the
% actions A(1:t-1), and rewards R(1:t-1) have been seen according to the
% Structure learnig model.
%
% m: The number of arms
% A: The action vector
% R: The reward vector
% pdfc_phi: Prior over coupling
% gamma: Discount factor
% H: horizon
%
% Daniel Acuna (2008), Dept. of CS and Eng.,U.of Minnesota
% -modified by Daniel Acuna 2009: adding discount factor, name changed

% Number of actions
n = size(A,1);
% Action value vector
V = zeros(size(A,1)+1,m);

for t=0:n
	s1 = sum(R(A(1:t)==1)>0);
	f1 = sum(R(A(1:t)==1)==0);
	s2 = sum(R(A(1:t)==2)>0);
	f2 = sum(R(A(1:t)==2)==0);
	
	%tmpV = StructureLearningModel(1,1,1,1,...
	% 1,1,pdfc_phi,s1,f1,s2,f2, H, gamma);
	
	tmpV = StructureLearningModel(1,1,1,1,...
		1,1,StructureLearningModel_PosteriorCoupling(pdfc_phi, 1, 1,1, 1, 1, 1, s1,f1,s2,f2),s1,f1,s2,f2, H, gamma);
	
	% Making values that are close the same. There are some numerical
	% problems that need to be addressed
	tmpV = round(tmpV*10000)/10000;
	
	V(t+1, 1) = tmpV(1)*(1-gamma);
	V(t+1, 2) = tmpV(2)*(1-gamma);
end