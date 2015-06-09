%% subgroup discovery via Quadratic Programming
clc
clear
close all

% reference:
% Li, R, Ahmadi, Z, and Kramer, S (2014).
% Constrained Latent Dirichlet Allocation for Subgroup Discovery with Topic Rules
% In: ECAI 2014: 21st European Conference on Artificial Intelligence, IOS Press.

%% doc word construction example, each feature condition is a word,
%% constrained LDA, select feature conditions from WP for SD, using only the
%% smallest perplexity for SD
%     F1 F2 F3
% d1  1  2  5...
% ... 1
%     2
% 
% F represents the feature (feature value as the word in LDA), d reprents the instance (as documents in LDA)
% We need to re-code the data so that the feature values are distinguable
% in the proposed CLDA algorithm.
% DS: 1 1 1 1 1 1 1 1 1 1 1... (document indicator)
% The core constrained LDA code is written in C++ and compiled into Matlab runnable
% mex function
% results is saved in "rulesCLDASum"
%%

Betas = [0.01 0.05 0.1];

N = 500; 
SEED = 3;
OUTPUT = 1;
maxDepth = 4;
fold = 5;
thres = 0.01; % threshold for beta
MiniTopic = 5;
MaxiTopic = 10;

load 'Pima'
[data cuts] = entroDis(data, label);
[num dim] = size(data);
oriData = data;
dataIndex = data;
for k = 1:dim
    dataIndex(:, k) = k;
end

%%% make a pesudo matrix of data in order to distinguish features and the
%%% respective values
dataCopy = data;
code = 0;
for k = 1:dim
    fea = data(:, k);
    feaCopy = fea;
    uniFea = unique(fea);
    for mm = 1:length(uniFea)
        code = code + 1;
        idx = find(fea == uniFea(mm));
        feaCopy(idx) = code;
    end
    dataCopy(:, k) = feaCopy;
end
data = dataCopy;
  
indices = crossvalind('Kfold', label, fold);

%% run a cross-validation to divide the data into disjoint sets
%% training data is used to select the optimal number of topics
%% test data is used to get the rules
%%

for i = 1:fold
    % global index
    testIndex = (indices == i);
    trainIndex = ~testIndex;

    training = data(trainIndex, :);
    test = data(testIndex, :);    
    trainLabel = label(trainIndex);
    testLabel = label(testIndex);
    ind = 1:size(data, 2);

    trainingIndex = dataIndex(trainIndex, :);

    for kk = 1:2
        uniLabel = unique(trainLabel);
        nPos = length(find(trainLabel == uniLabel(1)));
        nNeg = length(find(trainLabel == uniLabel(2)));
        thres = 0.01; % threshold for subgroup rule
        [n p] = size(trainingIndex);
        p0F1 = nPos/n; % fraction of positive targets
        p0F2 = nNeg/n;
        p0FPN = [p0F1; p0F2];
        p0F = p0FPN(kk);           

        tempInd = find(trainLabel == kk);
        tempIndTest = find(testLabel == kk);
        trainingClass = training(tempInd, :); % training data for a certain class
        trainingIndexClass = trainingIndex(tempInd, :); % training data for a certain class, the index
        testClass = test(tempIndTest, :);

        [train_row, train_col] = size(trainingClass);

        %% construct the documents (samples) matrix, token(feature) matrix
        Docs = [];
        for k = 1:train_row
            doc = trainingClass(k, :);
            docLen = length(doc); % indicator len of a document
            Docs = [Docs k*ones(1, docLen)];
        end

        [trainingClass uniFeaSum fIdx uniFeas] = LDAdata(trainingClass); % re-code the data

        Tokens = reshape(trainingClass.',1,[]);
        TokensIndex = reshape(trainingIndexClass.',1,[]);
        nFC = zeros(size(trainingClass, 2), 1);

        for k = 1:size(trainingClass, 2)
            nFC(k) = length(unique(trainingClass(:, k))); 
        end

        fCell = cell(size(trainingClass, 2), 1);
        for k = 1:size(trainingClass, 2)
            fCell{k} = unique(trainingClass(:, k)); 
        end        

        featuresExhaust = cell(50, 1);
        feaIdx = zeros(max(uniFeaSum), 1);
        %%% perplexiy
        perp = zeros(length(Betas), length(MiniTopic:MaxiTopic));
        for B = 1:length(Betas)
            BETA = Betas(B);
            count = 0;
            for T = MiniTopic:MaxiTopic
                ALPHA = 50/T;
                count = count + 1;
                [WP, DP, Z] = gibbsampleCLDAmex(Tokens, Docs, T, N, ALPHA, BETA, SEED, 1, TokensIndex, nFC, fCell);
                wordTProb = wtp(WP, BETA, size(WP, 1));
                [perp(B, count) WP] = LDAppx(WP, BETA, test, uniFeaSum, DP, ALPHA);
                WPfull = full(WP); % WP(i,j) contains the number of times word i has been assigned to topic j
                DPfull = full(DP); % number of times a word token in document d has been assigned to topic j. 
                WPmatrix{B, count} = WPfull;
            end % match for T = MiniTopic:50
        end
        WPmatrixClass{kk, 1} = perp;
        WPmatrixClass{kk, 2} = WPmatrix;
        WPmatrixClass{kk, 3} = fIdx;
        WPmatrixClass{kk, 4} = uniFeas;
        WPmatrixClass{kk, 5} = p0F;
        WPmatrixClass{kk, 6} = thres;
        WPmatrixClass{kk, 7} = n;

    end % match for kk = 1:2
    
    nTraining = size(training, 1);
%% discover the SD rules using the lowest perplexity
           
    clear idxSum
    nT = MaxiTopic;
    for kk = 1:2
        cla = WPmatrixClass(kk,:);
        perplex = cla{1};
        [xind yind] = find(perplex == min(min(perplex))); %index minimum of the matrix A 
        bestTopicIdx = size(cla{1, 2}{xind, yind}, 2);
        featureIdx = cla{3};
        featureValue = cla{4};
        dc = cla{5}; % default accuracy
        thresSD = cla{6};
        nSamp = cla{7};
        idxSum = [];
        for j = 1:bestTopicIdx
            topicFea = cla{1, 2}{xind, yind}(:, j);
            topicFea = feaRemove(topicFea, thres, nTraining, trainLabel, kk);
            idx = find(topicFea);
            nwt = topicFea(idx);
            idx = sort(idx, 'descend');
            idxSum{j, 1} = idx;
        end
        classRule = fvToRule(idxSum, training, trainLabel, featureIdx, featureValue, kk);
        classRule = ruleOrgan(classRule, kk);
        rulesCLDA{kk} = classRule;
        nBestTopic(1, kk) = bestTopicIdx;
    end % match for kk = 1:2
    rulesCLDASum{i} = rulesCLDA;    
end            































