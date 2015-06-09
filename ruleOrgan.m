function rules = ruleOrgan(sd, target)
%% organize the rules

n = size(sd, 1);

pgAll = [];
rule = [];
feature = [];
coverage = [];
support = [];
ruleSignificance = [];
ruleTopic = [];

%% put all things together

if target == 1
    for i = 1:n
            tar = sd;
            pgAll = [pgAll; tar.pg];
            rule = [rule; tar.rule];
            feature = [feature; tar.feature];
            coverage = [coverage; tar.coverage];
            support = [support; tar.support];
            ruleSignificance = [ruleSignificance; tar.ruleSignificance];
            ruleTopic = [ruleTopic; tar.ruleTopic];
    end
else
    for i = 1:n
            tar = sd;
            pgAll = [pgAll; tar.pg];
            rule = [rule; tar.rule];
            feature = [feature; tar.feature];
            coverage = [coverage; tar.coverage];
            support = [support; tar.support];
            ruleSignificance = [ruleSignificance; tar.ruleSignificance];
            ruleTopic = [ruleTopic; tar.ruleTopic];
    end    
end
[feature index] = repFeaCheck(feature, rule); % remove repeated feature pairs
pgAll = pgAll(index);
rule = rule(index);
coverage = coverage(index);
support = support(index);
ruleSignificance = ruleSignificance(index);
ruleTopic = ruleTopic(index);

clear index

%%% sorting according to pg score    
[pgAll index] = sort(pgAll, 'descend');
rules.pg = pgAll;
rules.rule = rule(index);
rules.feature = feature(index);
rules.coverage = coverage(index);
rules.support = support(index);
rules.ruleSignificance = ruleSignificance(index);
rules.ruleTopic = ruleTopic(index);

%%% compute average
rules.ruleSizeClass = size(rules.pg, 1);
sz = size(rules.pg, 1);
sumSF = 0;
for i = 1:sz
    sumSF = sumSF + size(rules.feature{i, 1}, 1);
end
rules.ruleSizeFeature = sumSF/sz;

rules.ruleSignificanceMean = mean(ruleSignificance);
rules.ruleSignificanceStd = std(ruleSignificance);
rules.coverageMean = mean(coverage);
rules.coverageMeanStd = std(coverage);
rules.supportMean = mean(support);
rules.supportStd = std(support);
rules.pgMean = mean(pgAll);
rules.pgStd = std(pgAll);
% rules.sampCovCount = sd{1,1}{1,1}.sampCovCount;
%% remove irrelevant rules    
rules = ruleFilter(rules); 








