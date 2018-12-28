%% Generate SS model and data
clear
rng(1)

nx = 3;
ny = 1;
nu = 1;
Fs = 350;
N = 1000;

FIT = zeros(10, nx);

for k = 1 : 10
    rng(k)

sysn = drss(nx, ny, nu);
sysn.B = sysn.B / max(abs(sysn.B(:)));  % enforce std(X) ~ 1
sysn.C = sysn.C / max(abs(sysn.C(:)));  % enforce std(Y) ~ 1
sysan = ss(sysn.A, [sysn.B, eye(nx)], sysn.C, [sysn.D, zeros(ny, nx)], 1/Fs);

Q = 1e-1 * randn(nx,1); Q = Q * Q';
R = 1e-1 * randn(ny,1); R = R * R';
W = mvnrnd(zeros(nx,1), Q, N);
V = mvnrnd(zeros(ny,1), R, N);

T = (0:N-1)' / Fs;
U = randn(N, nu);
X0 = randn(nx,1);

[Y, ~, X] = lsim(sysan, [U, W], T, X0);
Z = Y + V;

% figure
% subplot(131), plot(X), title('X')
% hold on, set(gca, 'ColorOrderIndex', 1), plot(X-W, ':')
% subplot(132), plot(U), title('U')
% subplot(133), plot(Y, '--'), hold on, set(gca, 'ColorOrderIndex', 1), plot(Z), title('Y')

%% smooth
P0 = eye(nx);
[Xh, X0h, P0h] = linear_smooth(U, Z, X0*0, P0, sysn.A, sysn.B, sysn.C, sysn.D, ...
    Q, R, zeros(nx,ny));

% %% filter
% A = sysn.A;
% B = sysn.B;
% C = sysn.C;
% D = sysn.D;
% 
% kf = extendedKalmanFilter(@(x,u) A*x + B*u, ...
%     @(x,u) C*x + D*u, X0);
% kf.ProcessNoise = Q;
% kf.MeasurementNoise = R;
% 
% Xh = zeros(size(Y,1), nx);
% 
% for i = 1:size(Y,1)
%     [CorrectedState,CorrectedStateCovariance] = correct(kf, Z(i,:)', U(i,:)');
%     Xh(i,:) = CorrectedState';
%     [PredictedState,PredictedStateCovariance] = predict(kf, U(i,:)');
% end

%%

% figure
% plot(X, ':')
% hold on, set(gca, 'ColorOrderIndex', 1), plot(Xh)

fit = goodnessOfFit(Xh, X, 'NRMSE');

FIT(k,:) = fit';
end
      
FIT
mean(FIT), std(FIT)

