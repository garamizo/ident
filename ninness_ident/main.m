%% Generate SS model and data
clear
rng(1)

nx = 3;
ny = 1;
nu = 1;
Fs = 350;
N = 1000;

Q = 1e-1 * randn(nx,1); Q = Q * Q';
R = 1e-1 * randn(ny,1); R = R * R';
W = mvnrnd(zeros(nx,1), Q, N);
V = mvnrnd(zeros(ny,1), R, N);

sysn = drss(nx, ny, nu);
sysan = ss(sysn.A, [sysn.B, eye(nx)], sysn.C, [sysn.D, zeros(1, nx)], 1/Fs);

T = (0:N-1)' / Fs;
U = randn(N, nu);
X0 = randn(nx,1);

[Y, ~, X] = lsim(sysan, [U, W], T, X0);
Z = Y + V;

figure
subplot(131), plot(X), title('X')
hold on, set(gca, 'ColorOrderIndex', 1), plot(X-W, ':')
subplot(132), plot(U), title('U')
subplot(133), plot(Y, '--'), hold on, set(gca, 'ColorOrderIndex', 1), plot(Z), title('Y')

%% Estimate with subspace

data = iddata(Z, U, 1/Fs);

% Import   data               
 datae = data([1:750]);
 datav = iddata(Y, U, 1/Fs);
 datav = datav([750:1000]);
                               
% State space model estimation 
 Options = n4sidOptions;       
 Options.Display = 'off';
 Options.EnforceStability = true;
                               
 ss1 = n4sid(datae, 3, Options)
 figure, compare(datav, ss1, sysn)
                               

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
