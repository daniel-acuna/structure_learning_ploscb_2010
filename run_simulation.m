% Runnion example

% Run simulation
thetas = [0.8 0.9];
r = [];
a = [];
all_values = {};
all_max_values = {};
H = 20;
parfor i = 1:(8*10)
    disp(i);
    r = [];
    a = [];
    values = [];
    max_values = [];
    for t = 1:H
        s1 = nnz(r==1 & a == 1);
        f1 = nnz(r==0 & a == 1);
        s2 = nnz(r==1 & a == 2);
        f2 = nnz(r==0 & a == 2);
%         s1 = 1;
%         f1 = 2;
%         s2 = 0;
%         f2 = 1;
%         s1 = 0;
%         f1 = 0;
%         s2 = 0;
%         f2 = 5;
% %         [s1, f1, s2, f2] = [1   2     0     1];
%        
% %         s1 = 4;
% %         f1 = 2;
% %         s2 = 3;
% %         f2 = 2;
% %         prior_c = 0.5;
        
        prior_c = StructureLearningModel_PosteriorCoupling(0.5, 1, 1, 1, 1, 1, 1, s1, f1, s2, f2);
%         prior_c=0.9;
        % Compute mean model

%         s1=0;
%         f1=0;
%         s2=0;
%         f2=0;
%         prior_c=0.5;
        % Optimal value
%         H=8;
        opt_V = StructureLearningModel(1,1,1,1,1,1,prior_c,s1, f1, s2, f2,H-t+1,1);
%         disp(opt_V)
        % Value of mean model
        mean_V = StructureLearningModel_MeanModel(1,1,1,1,1,1,prior_c,s1, f1, s2, f2,H-t+1,1);
%         disp(mean_V);
        % Value of reward learning
        rl_V = StructureLearningModel_RewardProbability(1,1,1,1,1,1,prior_c,s1, f1, s2, f2,H-t+1,1);
%         disp(rl_V);
        % Compute value of lookahead only on structure learning
        sl_V = StructureLearningModel_StructureLearning(1,1,1,1,1,1,prior_c,s1, f1, s2, f2,H-t+1,1);
%         disp(sl_V);
        
        % Choose action according to optimal value
        %[~, a(t)] = max(opt_V);
%         if t == 1
%             a(t) = 1;
%         else
%             a(t) = 3 - a(t-1);
%         end
        a(t) = (rand<0.5) + 1;
        r(t) = rand < thetas(a(t));
        %values = [values; opt_V(a(t)) mean_V(a(t)) sl_V(a(t)) rl_V(a(t))];
        values = [values; mean_V(a(t)) rl_V(a(t)) prior_c];
        max_values = [max_values; max(opt_V) max(mean_V) max(sl_V) max(rl_V)];
        %max_values = [max_values; max(mean_V) max(rl_V)];
    end
    all_values{i} = values;
    all_max_values{i} = max_values;
end

%% Analysis
all_max_values2 = cell2mat(all_max_values);
% Optimal exploratory bonus across runs
eb = (all_max_values2(:, 1:4:end)-all_max_values2(:, 2:4:end))';
% Optimal exploratory bonus of structure
ebs = (all_max_values2(:, 3:4:end)-all_max_values2(:, 2:4:end))';
% Optimal exploratory bonus of parameter learning
ebp = (all_max_values2(:, 4:4:end)-all_max_values2(:, 2:4:end))';
% Probability of coupling
vals2 = reshape(cell2mat(all_values), [20, 3, length(all_values)]);
vals2_mean = mean(vals2, 3);
vals2_std = std(vals2, [], 3);

figure(1);
% plot(mean(eb), '+-');
errorbar(mean(eb), sqrt(var(eb)/sqrt(size(eb,1))));
figure(2);
% plot(mean(ebs), '+-');
errorbar(mean(ebs), sqrt(var(ebs)/sqrt(size(ebs,1))));

% figure(3);
% plot(mean(ebs), '+-');
% errorbar(mean(eb), sqrt(var(eb)/sqrt(size(eb,1))), 'b');
% hold on;
% errorbar(mean(ebp), sqrt(var(ebp)/sqrt(size(ebp,1))), 'r');
% figure(4);
% errorbar(vals2_mean(:, 3), vals2_std(:, 3)/length(all_values));
% title('Probability of coupling');
%
figure(4);
plotyy(1:length(mean(ebs)), mean(ebs), ...
    1:length(mean(ebs)), vals2_mean(:, 3));

