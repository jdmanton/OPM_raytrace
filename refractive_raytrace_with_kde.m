%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% James Manton, Nov 2019    %
% jmanton@mrc-lmb.cam.ac.uk %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Parameters
SAVE_PSF = 1;
n_1 = 1;
na = 0.95;
n_2 = 1.518;
field_size = 512;

surface_tilt = 27;

n = [sind(surface_tilt); 0; -cosd(surface_tilt)];

% Set up input rays
px = -2:0.1:2;
py = px;
[px, py] = meshgrid(px, py);
px = px(:)';
py = py(:)';
pr = sqrt(px.^2 + py.^2);
px(pr > 2 * na) = NaN;
py(pr > 2 * na) = NaN;
pz = -ones(1, numel(px));
i = vertcat(px, py, pz);

% Normalise all rays
i = i ./ vecnorm(i, 2, 1);
n = n ./ norm(n);
n = repmat(n, 1, size(i, 2));

% Calculate transmitted rays
mu = n_1 ./ n_2;
t = mu .* i + n .* (sqrt(1 - mu.^2 .* (1 - dot(n, i).^2)) - mu .* dot(n, i));
t = t ./ vecnorm(t, 2, 1);

% Calculate points for surface
w = null(n(:, 1)');
[p, q] = meshgrid(-1:0.1:1);
surf_x = w(1,1) * p + w(1,2) * q;
surf_y = w(2,1) * p + w(2,2) * q;
surf_z = w(3,1) * p + w(3,2) * q;
origin = zeros(size(i));

% Rotate vectors for output pupil display
rot = [cosd(surface_tilt), 0, sind(surface_tilt); 0, 1, 0; -sind(surface_tilt), 0, cosd(surface_tilt)];
t_rot = rot * t;

% Produce KDEs
kx = linspace(-2, 2, field_size);
ky = kx;
[kx, ky] = meshgrid(kx, ky);
kde_sigma = 0.1;

kde_in = zeros(field_size);
for j = 1:size(i, 2)
    if (any(isnan(i(:, j))))
        continue
    else
    kde_in = kde_in + exp(-(kx - i(1, j)).^2 / kde_sigma.^2) .* exp(-(ky - i(2, j)).^2 / kde_sigma.^2);
    end
end

kx = linspace(-2 * n_2, 2 * n_2, field_size);
ky = kx;
[kx, ky] = meshgrid(kx, ky);
kde_out = zeros(field_size);
for j = 1:size(i, 2)
    if (any(isnan(i(:, j))))
        continue
    else
    kde_out = kde_out + exp(-(kx - t_rot(1, j)).^2 / kde_sigma.^2) .* exp(-(ky - t_rot(2, j)).^2 / kde_sigma.^2);
    end
end


% Produce PSF
x = linspace(-2*n_2, 2*n_2, field_size);
[X, Y, Z] = ndgrid(x, x, x);
R = sqrt(X.^2 + Y.^2 + Z.^2);
phi = atan2(X, Y) * 180 / pi;
theta = atan2(X, Z) * 180 / pi;
R = 2 * R / max(x);
r = sqrt(X.^2 + Y.^2);
r = 2 * r / max(x);
pupil = repmat(kde_out, [1, 1, field_size]);
ctf = zeros(field_size, field_size, field_size);
ctf(R < 0.5 & R > 0.48 & Z > 0) = 1;
ctf = ctf .* pupil;
otf = fftshift(ifftn(fftn(ctf) .* conj(fftn(ctf))));
psf = abs(otf2psf(otf));
psf = psf ./ max(psf(:));

if SAVE_PSF
    psf_16 = uint16((2^16 - 1) * psf);
    imwrite(psf_16(:, :, 1), ['psf-', num2str(surface_tilt), '.tif'])
    for j = 2:field_size
        imwrite(psf_16(:, :, j), ['psf-', num2str(surface_tilt), '.tif'], 'WriteMode', 'append')
    end
end


% Draw results
figure(1)
surf(surf_x, surf_y, surf_z, zeros(size(surf_z)), 'EdgeColor', [0.5, 0.5, 0.5], 'FaceAlpha', 0.5)
colormap gray
hold on
iq = quiver3(-i(1, :), -i(2, :), -i(3, :), i(1, :), i(2, :), i(3, :), 0, 'b');
tq = quiver3(origin(1, :), origin(2, :), origin(3, :), t(1, :), t(2, :), t(3, :), 'r');
set(iq, 'MaxHeadSize', 0)
set(tq, 'MaxHeadSize', 0)
hold off
xlim([-1, 1])
ylim([-1, 1])
zlim([-1, 1])
axis vis3d
xlabel('x')
ylabel('y')
zlabel('z')
legend('Interface', 'Input', 'Output')
view(0, 0)

figure(2)
scatter(i(1, :), i(2, :), 1, 'b')
viscircles([0, 0], na, 'Color', 'b', 'LineWidth', 1);
xlim([-1, 1])
ylim([-1, 1])
axis square
title('Input pupil')

figure(3)
scatter(t_rot(1, :), t_rot(2, :), 1, 'r')
viscircles([0, 0], na, 'Color', 'b', 'LineWidth', 1);
viscircles([0, 0], 1, 'Color', 'r', 'LineWidth', 1);
xlim([-n_2, n_2])
ylim([-n_2, n_2])
axis square
title('Output pupil')

figure(4)
imshow(kde_in, [])
title('Input pupil')

figure(5)
imshow(kde_out, [])
title('Output pupil')
