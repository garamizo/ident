%% Define SS model
clear

A = 0.9;
B = [0.1, -0.2];
C = 0.5;
D = [0, 0];

Q = 0.001 * 1e1;  % state noise
R = 0.01 * 1e1;  % meas noise
S = 0;

X0 = 2.1;  % X_1

% Create data

N = 500;  % number of samples
nx = size(A, 1);
nu = size(B, 2);
ny = size(C, 1);

X = zeros(N, nx);
Y = zeros(N, ny);
Z = zeros(N, ny);

U = randn(N, nu);
Xk = X0;

rng(1)
for k = 1 : N
    noise = mvnrnd(zeros(ny+nx,1), [Q, S; S', R]);
    w = noise(1:nx);
    v = noise(nx+(1:ny));
    
    X(k,:) = Xk';
    Z(k,:) = (C * Xk + U(k,:) * D')';
    Y(k,:) = (C * Xk + U(k,:) * D' + v)';
    
    Xk = A * Xk + B * U(k,:)' + w;
end

%% Smooth

niter = 10;

X0h = 0;
P0h = 10;

Ah = 1.2;
Bh = [0.2, -0.5];
Ch = 0.9;
Dh = [0, 0];

% Qh = 0.015;
% Rh = 0.05;
% Sh = 1e-4;
Qh = Q * 1;
Rh = R * 1.0;
Sh = S;

log_data = [];

num = zeros(nu, nx+1);
den = zeros(nu, nx+1);

rows = randperm(N-1);
train_rows = rows(1:round(end*0.9));
val_rows = rows(round(end*0.9):end);

mse_val = zeros(niter, ny);  % val vs measured
mse_train = zeros(niter, ny);  % train vs measured
mse_non = zeros(niter, ny);  % all vs nominal
P_log = zeros(niter, nx*nx);

phi = zeros(nx+ny, nx+ny, N-1);
psi = zeros(nx+ny, nx+nu, N-1);
sii = zeros(nx+nu, nx+nu, N-1);

for kreps = 1 : niter
    [Xs, X0h, P0h] = linear_smooth(U, Y, X0h, P0h, Ah, Bh, Ch, Dh, Qh, Rh, Sh);
    
    %{
    zz = [Xs(1:end-1,:), U(1:end-1,:)];
    qsi = [Xs(2:end,:), Y(1:end-1,:)];
    for k = 1 : N-1
        phi(:,:,k) = qsi(k,:)' * qsi(k,:);
        psi(:,:,k) = qsi(k,:)' * zz(k,:);
        sii(:,:,k) = zz(k,:)' * zz(k,:);
    end
    PHI = mean(phi, 3);
    PSI = mean(psi, 3);
    SII = mean(sii, 3);
    
    Linv = PSI / SII;
    Ah = Linv(1:nx, 1:nx);
    Bh = Linv(1:nx, nx+(1:nu));
    Ch = Linv(nx+(1:ny), 1:nx);
    Dh = Linv(nx+(1:ny), nx+(1:nu));
    
    Pro = PHI - (PSI/SII)*PSI';
%     tmpChol = chol([SII, PSI'; PSI, PHI]);
%     Pro = tmpChol(4:5,4:5)' * tmpChol(4:5,4:5);
%     Qh = Pro(1:nx,1:nx);
%     Rh = Pro(nx+(1:ny), nx+(1:ny));
%     Sh = Pro(1:nx, nx+(1:ny));
    %}
    
    % LSQ silly 
    resCov = cov([Xs(1:end-1,:), U(1:end-1,:)] * [Ah', Ch'; Bh', Dh'] - ...
        [Xs(2:end,:), Y(1:end-1,:)])
%     Qh = resCov(1:nx,1:nx);
%     Rh = resCov(nx+(1:ny),nx+(1:ny));
%     Sh = resCov(1:nx,nx+(1:ny));

    % Solve for params
    AA = [Xs(1:end-1,:), U(1:end-1,:)];
	BB = [Xs(2:end,:), Y(1:end-1,:)];
    
    sol = (AA' * AA) \ AA' * BB;
    Ah = sol(1:nx,1:nx)';
    Bh = sol(nx+(1:nu),1:nx)';
    Hh = sol(1:nx,nx+(1:ny))';
    Dh = sol(nx+(1:nu),nx+(1:ny))';

    % transform to TF
    for k = 1 : nu
        [num(k,:), den(k,:)] = ss2tf(Ah, Bh, Ch, Dh, k);
    end
    log_data = [log_data; num(:)', den(:)', X0h(:)'];
    
    Ys = Xs * Ch' + U * Dh';
    mse_val(kreps,:) = goodnessOfFit(Ys(:,:), Y(:,:), 'NMSE');
    mse_train(kreps,:) = goodnessOfFit(Ys(train_rows,:), Y(train_rows,:), 'NMSE');
    mse_non(kreps,:) = goodnessOfFit(Ys, Z, 'NMSE');
    P_log(kreps,:) = P0h;
end

figure
plot(Y, ':k')
hold on, plot(Z, 'k'), plot(Ys)
legend('measured', 'nominal', 'smoothed')
title(sprintf('th = %.2f', mse_non(end,:)))

rows = 10:niter; % round(N/10):N;
% figure, plot(zscore(log_data(round(end/10):end,end-ny+1:end)), '--')
% hold on, plot(zscore(log_data(round(end/10):end,1:2*nu*(nx+1))))
% legend('fit', 'num', 'den1', 'den2', 'fit')
figure, plot(mse_train(rows,:), '--'), grid on
hold on, plot(mse_val(rows,:), '-')
hold on, plot(mse_non(rows,:), '-')
ylim([0, 1])
legend('train', 'validation', 'nominal', 'location', 'southwest')

sysEM = tf(ss(Ah, Bh, Ch, Dh, 1));

% figure, plot(P_log, '.-')
% 
% tbl = table(log_data(:,1:nx^2), log_data(:,nx^2+(1:nx*nu)), log_data(:,nx^2+nx*nu+(1:nx*ny)), ...
%     log_data(:,nx^2+nx*nu+nx*ny+nx+(1:ny)), 'VariableNames', {'A', 'B', 'H', 'NMSE'});
% tbl([1:50:end, end],:)

%% Sys ident
sysNON = tf(ss(A, B, C, D, 1))
mydata = iddata(Y, U, 1);
% mydata = iddata(Y(1:end-1), U(2:end), 1, 'InterSample', 'zoh');

 % Import   mydata               
 mydatae = mydata([1:151]);    
                                 
% State space model estimation   
%  Options = n4sidOptions;         
%  Options.Display = 'off';                   
%  ss1 = n4sid(mydatae, 1, Options)
 
% Transfer function estimation               
 Options = tfestOptions;                     
 Options.Display = 'off';                     
 Options.WeightingFilter = [];                       
 tf1 = tfest(mydatae, 1, 0, Options, 'Ts', 1)
 
 mydatav = mydata([151:299]);
figure, compare(mydatav, tf1)
