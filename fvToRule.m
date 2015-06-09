function SDrules = fvToRule(idxSum, training, trainLabel, featureIdx, featureValue,kk)
% feature value to SD rule

%%% idxSum: the index of feature values that belong to same topic
nT = length(idxSum);
maxDepth = 4;
uniLabel = unique(trainLabel);
nPos = length(find(trainLabel == uniLabel(1)));
nNeg = length(find(trainLabel == uniLabel(2)));
a = 1;
thres = 0.01;
ct = 0;

[n p] = size(training);
p0F1 = nPos/n; % fraction of positive targets
p0F2 = nNeg/n;
p0FPN = [p0F1; p0F2];
p0F = p0FPN(kk);

pgscore = [];
SDrules = [];

for i = 1:nT
    idx = idxSum{i}; 
    selFea = featureIdx(idx);
    selFeaValue = featureValue(idx);
    len = length(selFea);
    for j = 1:maxDepth
        if len >= j
            combs = nchoosek(1:len, j); 
            for k = 1:size(combs, 1)
               comb = combs(k, :);
               realFea = selFea(comb);
               uni = unique(realFea);
               if length(uni) == length(realFea) % if not from the same features
                   ct = ct + 1;
                   fea = training(:, realFea);
                   candiFea = selFeaValue(comb);
                   [gF fIdx] = featureIntersect(fea, candiFea'); % coverage
                   [pF count] = featureLabIntersect([fea trainLabel], [candiFea' uniLabel(kk)]); % support
                   pgTemp = gF^a*(pF-p0F);
                   %%% compute rule significance, it considers both classes
                   if kk == 1
                       [pFS countS] = featureLabIntersect([fea trainLabel], [candiFea' uniLabel(2)]); % support of class2
                   else
                       [pFS countS] = featureLabIntersect([fea trainLabel], [candiFea' uniLabel(1)]); % support of class1
                   end
                   sig1 = count*log2((count+1)/((nPos*gF)+2)); % class positive part
                   sig2 = countS*log2((countS+1)/((nNeg*gF)+2)); % class negative part
                   sig = 2*(sig1+sig2);
                   if pgTemp > thres
                       pgscore(ct, 1) = gF^a*(pF-p0F); % quality score
                       rule{ct, 1} = candiFea; % feature values
                       ruleFea{ct, 1} = realFea; % features used
                       ruleCov(ct, 1) = gF; % coverage
                       ruleSupp(ct, 1) = pF; % support
                       ruleSig(ct, 1) = sig;
                       ruleTopic(ct, 1) = i; % from which topic
                   else
                       ct = ct - 1;
                   end
               end % match if length(uni) == length(realFea)
            end % match for k = 1:size(combs, 1)
        end % match if len > j
    end  
end


% pgscore(ct+1:end) = [];
% rule(ct+1:end) = [];
% ruleFea(ct+1:end) = [];
% ruleCov(ct+1:end) = [];
% ruleSupp(ct+1:end) = [];
% ruleSig(ct+1:end) = [];
% ruleTopic(ct+1:end) = [];


%% sort subgroup description by quality
if ~isempty(pgscore)
    remove = find(pgscore < thres); % quality smaller than 0.1, then remove
    pgscore(remove) = [];

    [pgscore index] = sort(pgscore, 'descend');

    rule(remove) = [];
    ruleFea(remove) = [];
    ruleCov(remove) = [];
    ruleSupp(remove) = [];
    ruleSig(remove) = [];
    ruleTopic(remove) = [];
    rules = cell(size(pgscore, 1), 1); % ordered new rules
    rulesFea = cell(size(pgscore, 1), 1); 
    for i = 1:size(pgscore, 1)
        rules{i, 1} = rule{index(i)};
        rulesFea{i, 1} = ruleFea{index(i)};
        ruleCov(i, 1) = ruleCov(index(i));
        ruleSupp(i, 1) = ruleSupp(index(i));
        ruleSig(i, 1) = ruleSig(index(i));
        ruleTopic(i, 1) = ruleTopic(index(i));
    end
    SDrules.pg = pgscore;
    SDrules.rule = rules;
    SDrules.feature = rulesFea;  
    SDrules.coverage = ruleCov;
    SDrules.support = ruleSupp;
    SDrules.ruleSignificance = ruleSig;
    SDrules.ruleTopic = ruleTopic;
end

%% sub-functions

    %%% the probability of feature pairs appear, i.e. how often [1 3] appear in
    %%% whole feature set, i.e. coverage cov(R) = n(cond)/n_s
    function [probF index] = featureIntersect(feature, candidate)
        n = size(feature, 1); % all examples
        temp = repmat(candidate, n, 1); % repeat matrix
        res = abs(feature-temp); % feature difference, same feature will be zero after minus
        index = find(sum(res, 2) == 0); % matched feature pairs
        inter = length(find(sum(res, 2) == 0)); % how many common paris found
        class = 2; % two class problem
        probF = (inter+1)/(n+class); % probability of this feature pair using Laplace estimate
    end

    %%% the prob. of feature, label pairs appear, i.e. support
    function [probF inter] = featureLabIntersect(feaLabel, candidate)
        n = size(feaLabel, 1); % all examples
        temp = repmat(candidate, n, 1); % repeat matrix
        res = abs(feaLabel-temp); % feature difference, same feature will be zero after minus
        inter = length(find(sum(res, 2) == 0)); % no. of times feature and this lable appear, n(cond,class)
        feature = feaLabel(:, 1:end-1); % only features
        cond = repmat(candidate(1, 1:end-1), n, 1); % feature condition
        res2 = feature-cond;
        inter2 = length(find(sum(res2, 2) == 0)); % no. of times feature appears, n(cond)
        class = 2; 
        probF = (inter+1)/(inter2+class); % probability of this feature pair using Laplace estimate
    end
    %% end sub-functions



end