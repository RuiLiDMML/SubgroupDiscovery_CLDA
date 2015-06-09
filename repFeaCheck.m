function [fea index] = repFeaCheck(feature, rule)

n = size(feature, 1);
ct = 0;
index1 = zeros(2*n, 1);    
if nargin < 2

    for m = 1:n-1
        candi = feature{m, 1}; % candidate
        len = length(candi);
        for p = m+1:n
            compF = feature{p, 1}; % to be compared
            len2 = length(compF);
            if len == len2 % if potentially to repeat, check further
                diff = candi - compF;
                if sum(abs(diff)) == 0 % repeated found
                    ct = ct + 1;
                    index1(ct, 1) = p;
                end

            end
        end 
    end
    index1(ct+1:end) = [];
    idx = unique(index1);
    index = (1:n)';
    index(idx) = [];
    fea = feature(index);

else%% consider fature and their values
    for m = 1:n-1
        candi = feature{m, 1}; % candidate
        candiR = rule{m, 1}; % feature value
        len = length(candi);
        for p = m+1:n
            compF = feature{p, 1}; % to be compared
            compFR = rule{p, 1}; %feature value
            len2 = length(compF);
            if len == len2 % if potentially to repeat, check further
                diff = candi - compF;
                diffR = candiR - compFR;
                if sum(abs(diff)) == 0 && sum(abs(diffR)) == 0% repeated found
                    ct = ct + 1;
                    index1(ct, 1) = p;
                end

            end
        end 
    end
    index1(ct+1:end) = [];
    idx = unique(index1);
    index = (1:n)';
    index(idx) = [];
    fea = feature(index);    
    
end