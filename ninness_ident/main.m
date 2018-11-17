%% Define SS model
clear

A = [0.9, 0.1
    -0.1, 0.9];

B = [0.1, -0.2
    1, -1.1];

H = [1, 0];

Q = [1e-1, 0; 0, 1e-1];  % state noise
R = diag(1);  % meas noise

%% Create data

N = 300;  % number of samples
nx = size(A, 1);
nu = size(B, 2);
ny = size(H, 1);

X = zeros(N, nx);
Y = zeros(N, ny);
Z = zeros(N, ny);

U = randn(N, nu);
X0 = [5; -5];

for k = 1 : N
    X0 = A * X0 + B * U(k,:)' + mvnrnd(zeros(nx,1), Q)';
    
    X(k,:) = X0';
    Y(k,:) = (H * X0)';
    Z(k,:) = (H * X0 + mvnrnd(zeros(ny,1), R)')';
end

% figure
% subplot(131), plot(X), title('X')
% subplot(132), plot(U), title('U')
% subplot(133), plot(Y, '--'), hold on, set(gca, 'ColorOrderIndex', 1), plot(Z), title('Y')

%% Smooth

Q = [1e-1, 0; 0, 1e-1];  % state noise
% Q = [1e-2, 0; 0, 1e-2];  % incorrect state noise

Xm = zeros(N, nx);
Xp = zeros(N, nx);
Pm = zeros(nx, nx, N);
Pp = zeros(nx, nx, N);
Xs = zeros(N, nx);

X0 = [0; 0];
P0 = [10, 0; 0, 10].^2;

for k = 1 : N
    % predict
    X0 = A * X0 + B * U(k,:)';
    P0 = A * P0 * A' + Q;
    
    Xm(k,:) = X0';
    Pm(:,:,k) = P0;
    
    % correct
    K = P0 * H' / (H * P0 * H' + R);
    X0 = X0 + K * (Z(k,:)' - H * X0);
    P0 = (eye(nx) - K * H) * P0;
    
    Xp(k,:) = X0';
    Pp(:,:,k) = P0;
end

Xs(end,:) = X0';

for k = N-1 : -1 : 1
    % smooth
    K = Pp(:,:,k) * A' / Pm(:,:,k+1);
    P0 = Pp(:,:,k) - K * (Pm(:,:,k+1) - P0) * K';
    X0 = Xp(k,:)' + K * (X0 - Xm(k+1,:)');
    
    Xs(k,:) = X0;
end

figure, 
h1 = subplot(221); plot(Xs, '.-'), grid on
hold on, set(gca, 'ColorOrderIndex', 1), plot(X, '.--'), title('smoothed')
h2 = subplot(223); plot(Xs - X, '.-'), grid on
h3 = subplot(222); plot(Xp, '.-'), grid on
hold on, set(gca, 'ColorOrderIndex', 1), plot(X, '.--'), title('filtered')
h4 = subplot(224); plot(Xp - X, '.-'), grid on
linkaxes([h1, h3], 'y'), linkaxes([h2, h4], 'y')

% figure,
% plot([Y, Xs * H', Z])
    
rows = 1 : 50;
fp = goodnessOfFit(X(rows,:), Xp(rows,:), 'NRMSE')';
fm = goodnessOfFit(X(rows,:), Xm(rows,:), 'NRMSE')';
fs = goodnessOfFit(X(rows,:), Xs(rows,:), 'NRMSE')';

rows = rows(end) : N;
fp(2,:) = goodnessOfFit(X(rows,:), Xp(rows,:), 'NRMSE')';
fm(2,:) = goodnessOfFit(X(rows,:), Xm(rows,:), 'NRMSE')';
fs(2,:) = goodnessOfFit(X(rows,:), Xs(rows,:), 'NRMSE')';

fit_table = table(fp, fm, fs, 'VariableNames', {'posteriori', 'priori', 'smooth'}, ...
    'RowNames', {'transient', 'permanent'})
