function topicFea = feaRemove(topicFea, thres, nTraining, trainLabel, target)
%%%% remove the feature condition which is less than the minimum required
%%%% number of samples using quality function

uniLabel = unique(trainLabel);
nPos = length(find(trainLabel == uniLabel(1)));
nNeg = length(find(trainLabel == uniLabel(2)));
p0F1 = nPos/nTraining; % fraction of positive targets
p0F2 = nNeg/nTraining;
p0FPN = [p0F1; p0F2];

minN = floor(nTraining*thres/(1-p0FPN(target)));

topicFea(find(topicFea<= minN)) = 0;
