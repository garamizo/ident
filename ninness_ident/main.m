%% Generate SS model and data
clear
rng(1)

nx = 3;
ny = 1;
nu = 1;
Fs = 350;
N = 1000;

FIT = zeros(10, ny);


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

%% Separate training and validation data

data = iddata(Z, U, 1/Fs);

% Import   data               
datae = data([1:750]);
datav = iddata(Y, U, 1/Fs);
datav = datav([750:1000]);
                               
%% State space model estimation 
%  Options = n4sidOptions;       
%  Options.Display = 'off';
%  Options.EnforceStability = true;          
%  ss1 = n4sid(datae, 3, Options);
 
%% Ninness estimation
ss2 = ninnessid(datae, 3, []);

%%
 
 
 [~,fit,~] = compare(datav, ss1);
 
%  figure, compare(datav, ss1, sysn)

FIT(k,:) = fit';
end
      
FIT

%% Smooth
