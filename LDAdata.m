function [proData uniFeaSum uniFeaIdx uniFeas] = LDAdata(data)

%%% reorganize the data for LDA, such that feature condition is
%%% consectutive
[nRow nCol] = size(data);
proData = zeros(nRow, nCol);

uniFeaSum = [];
uniFeaIdx = []; % the feature condition belonging to n-th feature
uniFeas  = []; % the unique feature conditions
for i = 1:nCol
   fea = data(:, i);
   uniFea = unique(fea);
   len = length(uniFea);
   feaCopy = zeros(nRow, 1);
   if i == 1
       for j = 1:len
           idx = find(fea == uniFea(j));
           feaCopy(idx) = j;
           uniFeaSum = [uniFeaSum uniFea(j)];
       end
   else
       code = max(proData(:, i-1));
       for j = 1:len
           idx = find(fea == uniFea(j));
           feaCopy(idx) = j+code;
           uniFeaSum = [uniFeaSum uniFea(j)];
           
       end      
   end
   if i == 1
       uniFeaIdx = [uniFeaIdx; i*ones(max(uniFea), 1)];
       proData(:, i) = feaCopy;
   else
       maxi = max(uniFea);
       maxi2 = max(data(:, i-1));
       uniFeaIdx = [uniFeaIdx; i*ones(length(maxi2+1:maxi), 1)];
       proData(:, i) = feaCopy;       
   end

   uniFeas(uniFea, 1) = uniFea;
end
