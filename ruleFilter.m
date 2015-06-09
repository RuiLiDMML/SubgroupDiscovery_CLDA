function newRule = ruleFilter(rules)
%%% remove irrelevant rules, i.e. a rule has low quality and a superset of
%%% other rules

nRule = size(rules.rule, 1);
ct = 0;
index = [];
for i = nRule:-1:2
   candiF = rules.feature{i, 1};
   candiV = rules.rule{i, 1};
   len = length(candiF);
   for j = i-1:-1:1
      compF = rules.feature{j, 1};
      compV = rules.rule{j, 1};
      len2 = length(compF);
      if len < len2 %% might be a potential irrelevant rule
          Fea = intersect(candiF, compF);
          pdCandiV = zeros(1, len2);
          pdCandiV(1:len) = candiV;
          if size(pdCandiV, 1) < size(pdCandiV, 2) 
            diff = pdCandiV' - compV; % check difference
          else
            diff = pdCandiV - compV; % check difference  
          end

          diffL = length(find(diff(1:len)==0));
          if length(Fea) == len && diffL == len
             ct = ct + 1;
             index(ct, 1) = i; % rule index supposed to be removed
          end
          clear pdCandiV diff
      end
   end    
end
if ~isempty(index)
    uni = unique(index);
    rules.pg(uni) = [];
    rules.rule(uni) = [];
    rules.feature(uni) = [];
    rules.coverage(uni) = [];
    rules.support(uni) = [];
    rules.ruleSignificance(uni) = [];
%     rules.sampleCover(uni) = [];
end
rules.ruleSizeClass = size(rules.pg, 1); % rule complexity, average number of rules per class
len = zeros(size(rules.pg, 1), 1);
sig = zeros(size(rules.pg, 1), 1);
cov = zeros(size(rules.pg, 1), 1);
supp = zeros(size(rules.pg, 1), 1);
pg = zeros(size(rules.pg, 1), 1);


for i = 1:size(rules.pg, 1)
    candi = rules.feature{i, 1};
    len(i, 1) = length(candi);
    sig(i, 1) = rules.ruleSignificance(i, 1);
    cov(i, 1) = rules.coverage(i, 1);
    supp(i, 1) = rules.support(i, 1);
    pg(i, 1) = rules.pg(i, 1);
end
rules.ruleSizeFeature = mean(len);
rules.ruleSizeFeatureStd = std(len);
rules.ruleSignificanceMean = mean(sig);
rules.ruleSignificanceStd = std(sig);
rules.coverageMean = mean(cov);
rules.coverageMeanStd = std(cov);
rules.supportMean = mean(supp);
rules.supportStd = std(supp);
rules.pgMean = mean(pg);
rules.pgStd = std(pg);


newRule = rules;