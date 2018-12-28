function sys = ninnessid(data, nx, opt)
% Identifies state space system via Ninness method
%
% data: iddata
% nx: system order
% opt: additional options
%
% sys: identified system

%%
ny = size(data.y, 2);
nu = size(data.u, 2);
N = size(data.y, 1);
U = data.u;
Y = data.y;

niter = 10;

X0h = zeros(nx, 1);
P0h = eye(nx);

Ah = eye(nx);
Bh = ones(nx, nu);
Ch = ones(ny, nx);
Dh = ones(ny, nu);

Qh = eye(nx);
Rh = eye(ny);
Sh = zeros(nx, ny);

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
%     resCov = cov([Xs(1:end-1,:), U(1:end-1,:)] * [Ah', Ch'; Bh', Dh'] - ...
%         [Xs(2:end,:), Y(1:end-1,:)]);
%     Qh = resCov(1:nx,1:nx);
%     Rh = resCov(nx+(1:ny),nx+(1:ny));
%     Sh = resCov(1:nx,nx+(1:ny));

    % Solve for params
    AA = [Xs(1:end-1,:), U(1:end-1,:)];
	BB = [Xs(2:end,:), Y(1:end-1,:)];
    
    fitlm(AA, BB)
    
    sol = (AA' * AA) \ AA' * BB;
    Ah = sol(1:nx,1:nx)';
    Bh = sol(nx+(1:nu),1:nx)';
    Hh = sol(1:nx,nx+(1:ny))';
    Dh = sol(nx+(1:nu),nx+(1:ny))';
end

sys = ss(Ah, Bh, Hh, Dh, data.Ts);
