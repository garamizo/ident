function [Xs, X0, P0] = linear_smooth(U, Y, X0, P0, A, B, C, D, Q, R, S)
% X_{k+1} = A*X_k + B*U_k + v
% Y_k = H*X_k + w
% v ~ N(0, Q)
% w ~ N(0, R)

N = size(Y, 1);  % number of samples
nx = size(A, 1);

Xm = zeros(N+1, nx);
Xp = zeros(N, nx);
Pm = zeros(nx, nx, N+1);
Pp = zeros(nx, nx, N);
Xs = zeros(N, nx);

Xm(1,:) = X0';
Pm(:,:,1) = P0;
for k = 1 : N
    % correct    
    K = P0 * C' / (C * P0 * C' + R);
    X0 = X0 + K * (Y(k,:)' - C * X0 - D * U(k,:)');
    P0 = P0 - K * C * P0;
    
    Xp(k,:) = X0';
    Pp(:,:,k) = P0;
    
    % predict    
    X0 = A * X0 + B * U(k,:)'; % k+1
    P0 = A * P0 * A' + Q;
    
    Xm(k+1,:) = X0';
    Pm(:,:,k+1) = P0;
end

Xs(end,:) = Xp(end,:);
X0 = Xp(end,:);
P0 = Pp(:,:,end);

for k = N-1 : -1 : 1
    % smooth    
    K = Pp(:,:,k) * A' / Pm(:,:,k+1);
%     MM = [
    P0 = Pp(:,:,k) + K * (P0 - Pm(:,:,k+1)) * K';
    X0 = Xp(k,:)' + K * (X0 - Xp(k,:)*A' - U(k,:)*B' - Y(k,:)*(S/R)');
    
    Xs(k,:) = X0;
end
