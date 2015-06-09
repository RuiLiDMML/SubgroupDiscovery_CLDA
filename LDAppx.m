function [perplexity WPout] = LDAppx(WP, BETA, test, uniFeaSum, DP, ALPHA)

nWords = size(WP, 1);
nTopics = size(WP, 2);
WPfull = full(WP);
total_words = size(test, 1)*size(test, 2);
[n_doc n_dim] = size(test);         
ztot = sum(WPfull);    
DPfull = full(DP);
docTotal = 0;
maxi1 = max(uniFeaSum);
maxi2 = max(max(test));
maxi = max([maxi1 maxi2]);
WPnew = zeros(maxi, nTopics);
WPnew(uniFeaSum, :) = WPfull; % pad zeros if the test data are not included in the training
P_w_z = zeros(size(WPnew, 1), nTopics);
sumWP = sum(WP, 1) + BETA*nWords;

for t = 1:nTopics
    P_w_z(:,t) = (WPnew(:, t)+BETA)./(repmat(sumWP(t),size(WPnew, 1),1));
end

seed = 3;
maxIter = 500;
DPfull = gibbsampleCLDAppx(test, nTopics, maxIter, ALPHA, BETA, seed);

for i = 1:n_doc
    for j = 1:n_dim % the words
        sumProb = 0;
        tar = test(i, j);
        for k = 1:nTopics % the topics
            phi_hat = P_w_z(tar, k);
            theta_hat = (DPfull(i, k)+ALPHA)/(sum(DPfull(i, :))+nTopics*ALPHA);
            prob = phi_hat*theta_hat;
            sumProb = sumProb + prob;
        end
        if sumProb ~= 0
            docTotal = docTotal + log2(sumProb);        
        end
    end
end

perplexity = exp(-docTotal/total_words);
WPout = WPfull;