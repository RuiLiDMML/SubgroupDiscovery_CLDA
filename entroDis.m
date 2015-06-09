function [feature cuts] = entroDis(data, label, type)
%%% entropy discretization
%%% reference: Multi-interval discretization of continuousvalued attributes
%%% for classification learning, Usama M. Fayyad and Keki B. Irani, 1993

data = roundn(data, -3); % round off data, this may bring up error if the data is too small
[n p] = size(data);
uniLabel = unique(label);
feature = zeros(n, p);

if nargin < 3
    type = 'nonBinary';
end

if strcmp(type, 'binary')
    for i = 1:p
        % for all feature
        fea = data(:, i);
        uni = unique(fea);
        % test for each possible cut point
        n_uni = length(uni);
        E_ats = zeros(n_uni, 1);
        for j = 1:n_uni
            index1 = (fea > uni(j));
            index2 = (fea <= uni(j));
            subLabel1 = label(index1);
            subLabel2 = label(index2);
            % purity computation
            A = length(find(subLabel1 == uniLabel(1)));
            B = length(find(subLabel1 == uniLabel(2)));
            entro1 = entropy(A, B);
            A = length(find(subLabel2 == uniLabel(1)));
            B = length(find(subLabel2 == uniLabel(2)));
            entro2 = entropy(A, B);
            partA = length(subLabel1)/n*entro1;
            partB = length(subLabel2)/n*entro2;
            E_ats(j, 1) = partA + partB;
        end
        [val ind] = min(E_ats);
        cutPoint = uni(ind); % record the cut point
        cuts{i} = cutPoint;
        %%% discretize feature
        feaOri = fea;
        fea(feaOri <= cutPoint) = 1;
        fea(feaOri > cutPoint) = 2;
        feature(:, i) = fea;
    end
else% if multi-interval
    for i = 1:p
        % for all feature
        fea = data(:, i);
        uni = sort(unique(fea));
        n = length(uni);
        cutPoint = uni;
        cutPointCheck = zeros(n, 3);
        cutPointCheck(:, 1) = cutPoint;
        cutPointCheck(1, 2) = 1;
        cutPointCheck(end, 2) = 1;
        flag = 'true';
        iter = 0;
        sFea = fea;
        sLabel = label;
        points = [];
        useList = [];
        while strcmp(flag, 'true')
            iter = iter + 1;
            [point1 flag2] = split(sFea, sLabel);
            if strcmp(flag2, 'continue')
                cutPointCheck(cutPointCheck(:, 1) == point1, 2) = 1; % mark as 1 to continue
            end
            cutPointCheck(cutPointCheck(:, 1) == point1, 3) = 1; % mark as 1 to continue
                      
            points = condCheck(cutPointCheck(:, 1), cutPointCheck(:, 2), points);
            if iter > 1
                %%% remove checked interval
                pts = points(:, 1:2);
                [unused ind] = setdiff(pts, useList, 'rows');
                points = points(ind, :); % points never been tested
            end
            if ~isempty(find(points(:, 3))) % if not finished
                flag = 'true';
                index = find(points(:, 3));
                %%% used list
                useList = [useList; points(index(1), 1:2)]; % fill up used list for checking
                points(index(1), 3) = 0;
                small = (fea >= points(index(1), 1));
                big = (fea <= points(index(1), 2));
                valid = (small&big);
                sFea = fea(valid);
                sLabel = label(valid);
                clear index
            else
                flag = 'false';
            end
        end
        reCut = cutPointCheck(:, 3);
        validPoints = cutPointCheck(find(reCut == 1), 1);
        % discretize feature
        vliadPoints = sort(validPoints, 'ascend');
        cuts{i} = vliadPoints;
        feaOri = fea;
        for k = 1:length(validPoints)
            if k == 1
                idx = (feaOri <= validPoints(k)); 
            else
                idx1 = (feaOri >= validPoints(k-1));
                idx2 = (feaOri < validPoints(k));
                idx = (idx1&idx2);
            end
            fea(idx) = k;
        end
        if k == 1
            idx = (feaOri > validPoints(end)); 
        else
            idx = (feaOri >= validPoints(end));
        end
        fea(idx) = k+1;
        feature(:, i) = fea;
    end
    

end

%% compute entropy
function [entroInfo] = entropy(A, B)
    if A == 0 && B == 0
        entroInfo = 0;
    else
        M = A/(A+B) + eps;
        N = B/(A+B) + eps;
        entroInfo = -M*log2(M) - N*log2(N);
    end
end
%% recursive split
function [cutPoint flag] = split(feature, label)
    % test for each possible cut point
    uni = unique(feature);
    uniLabel = unique(label);
    n_uni = length(uni);
    E_ats = zeros(n_uni, 1);
    gain_ats = zeros(n_uni, 1);
    delta_ats = zeros(n_uni, 1);
    tol = zeros(n_uni, 1);
    nClass1 = length(find(label == uniLabel(1)));
    if length(uniLabel) == 2 % if there are two classes
        nClass2 = length(find(label == uniLabel(2)));
    else
        nClass2 = 0;
    end
    entroS = entropy(nClass1, nClass2);
    n = nClass1+nClass2;
    for k = 1:n_uni
        index1 = (feature > uni(k));
        index2 = (feature <= uni(k));
        subLabel1 = label(index1);
        subLabel2 = label(index2);
        % purity computation
        A = length(find(subLabel1 == uniLabel(1)));
        if length(uniLabel) == 2 % if there are two classes
            B = length(find(subLabel1 == uniLabel(2)));
        else
            B = 0;
        end
        entro1 = entropy(A, B);
        A = length(find(subLabel2 == uniLabel(1)));
        if length(uniLabel) == 2 % if there are two classes
            B = length(find(subLabel2 == uniLabel(2)));
        else
            B = 0; 
        end
        entro2 = entropy(A, B);
        partA = length(subLabel1)/n*entro1;
        partB = length(subLabel2)/n*entro2;
        E_ats(k, 1) = partA + partB;
        gain_ats(k, 1) = gainATS(entroS, E_ats(k, 1));
        k1 = length(unique(subLabel1));
        k2 = length(unique(subLabel2));
        total = 2;
        delta_ats(k, 1) = deltaATS(entro1, entro2, entroS, k1, k2, total);
        tol(k, 1) = stopCond(delta_ats(k, 1), n);
    end
    [val1 ind1] = min(E_ats);
    minGain = gain_ats(ind1);
    minTol = tol(ind1);
    cutPoint = uni(ind1); % record the cut point
    if minGain < minTol
        flag = 'stop';
    else
        flag = 'continue';
    end
    
end

function points = condCheck(interval, array, points)
    index = find(array);
    for k = 1:length(index)-1
        points = [points; interval(index(k))  interval(index(k+1)) 1];
    end
    % check repetation
    rep = points(:, 1:2);
    [rep ind] = unique(rep, 'rows');
    points = points(ind, :);
end

%% MDL criterion
function delta_ats = deltaATS(entro1, entro2, entroS, k1, k2, k)
    delta_ats = log2(3^k-2)-(k*entroS-k1*entro1-k2*entro2);
end

function gain_ats = gainATS(entroS, E_ats)
    gain_ats = entroS - E_ats;
end

function tol = stopCond(delta_ats, N)
    tol = log2(N-1)/N + delta_ats/N;
end




end
