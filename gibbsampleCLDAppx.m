function [DPtest] = gibbsampleCLDAppx(test, T, maxIter, Alpha, Beta, seed)
%%% compute the DP using test data, for perplexity

% TokenIdx: tokens from the feature index, used for constraints

Docs = [];
docLen = size(test, 2);
for k = 1:size(test, 1)
    Docs = [Docs k*ones(1, docLen)];
end

Tokens = reshape(test.',1,[]);

nTokens = length(Tokens);
nWords = max(Tokens); % assume Tokens starts from 1!
nDocs = max(Docs);

%% random initialization
wp = zeros(nWords, T);
dp = zeros(nDocs, T);
ztot = zeros(T, 1); % number of tokens assigned to topic
z = zeros(nTokens, 1);

% RandStream.setGlobalStream(RandStream('mt19937ar','seed', 1+seed*2))

for i = 1:nTokens
    topic = randi([1 T], 1); % pick random topic
    word = Tokens(i);
    doc = Docs(i);
    wp(word, topic) = wp(word, topic) + 1; % nr. times word assigned to topic
    dp(doc, topic) = dp(doc, topic) + 1; % nr. times doc assigned to topic
    ztot(topic) = ztot(topic) + 1;
%     dtot(topic) = dtot(topic) + 1;
    z(i) = topic;
end

order = 1:nTokens;
order = order(randperm(length(order))); % random shuffle the numbers
WBeta = nWords*Beta;
KAlpha = T*Alpha;

%%% gibbs sampling
probs = zeros(T, 1);

for iter = 1:maxIter
    for i = 1:nTokens
        token = order(i); % pick a token
        word = Tokens(token);
        doc = Docs(token);
        topic = z(token); % current topic assignment
        ztot(topic) = ztot(topic) - 1;
        wp(word, topic) = wp(word, topic) - 1;
        dp(doc, topic) = dp(doc, topic) - 1;           
                
        totProb = 0;
        for j = 1:T
            phi_hat = (wp(word, j)+Beta)/(ztot(j)+WBeta);
            theta_hat = (dp(doc, j)+Alpha)/(sum(dp(doc, :))+KAlpha);
%             same as without normalizing term (sum(dp(doc, :))+KAlpha)
            probs(j) = phi_hat*theta_hat; % modified constrained LDA
            totProb = totProb + probs(j);
        end
        %%% sample a topic from the distribution
        rp = totProb*rand(1, 1);
        maxi = probs(1);
        topic = 1;
        while (rp > maxi)
          topic = topic + 1;
          maxi = maxi + probs(topic);
        end
        z(token) = topic;
        wp(word, topic) = wp(word, topic) + 1;
        dp(doc, topic) = dp(doc, topic) + 1;   
        ztot(topic) = ztot(topic) + 1;    
    end % match for i = 1:nTokens
end % match for iter = 0:maxIter

DPtest = dp;










