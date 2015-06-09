function P_w_z = wtp(WP, BETA, K)

W = size(WP, 1);
T = size(WP, 2);
K = min([W K]);

sumWP = sum(WP, 1) + BETA*W;
WPfull = full(WP);
% WPfull(find(WPfull ~= 0)) = 1;

P_w_z = zeros(K, T);


for t = 1:T
    P_w_z(:,t) = (WPfull(:, t)+BETA)./(repmat(sumWP(t),K,1));
end
