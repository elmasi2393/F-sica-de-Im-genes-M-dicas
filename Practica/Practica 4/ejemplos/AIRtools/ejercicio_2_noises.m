% Importa la libreria AIRtools
addpath(genpath('/Users/maximilianogatto/OneDrive\ -\ Universidad\ Nacional\ de\ Cuyo/Balseiro/4\ semestre/Imagenes\ medicas/Practica/Practica\ 4/ejemplos/AIRtools'));

close all;
fprintf('Ejercicio 2 - Cambio en numero de detectores\n');

% Fijo el numero de angulos y de dectores, e itero para cada nivel de ruido

% Seteo los parametros de la cual quier calcular la reconstrucción de la imagen
N = 240;           % Imagen de salida
niter = 10;  % numero de iteraciones para obtener la imagen
N_noises = 10;
angle_fijo = 100;
detect_fijo = 128;

% Creo los valores sobre los cuales iterar

noises = linspace(0.01, 0.1, N_noises);


% itero sobre los valores de n_angles y n_detects
for i = 1:N_noises
    % ---------------- VARIANDO RUIDOS -----------------------
    fprintf('Iteracion %d de %d\n', i, N_noises)
    fprintf('Noise: %d\n', noises(i))
    theta = linspace(0, 180, angle_fijo);
    p = detect_fijo;
    [A,b_ex,x_ex] = paralleltomo(N,theta,p);
    % Guardo la imagen exacta si estamos en el primer nivel de ruido
    if i ==1
        img_ex_detects = reshape(x_ex, N, N);
        save(sprintf('/Users/maximilianogatto/OneDrive - Universidad Nacional de Cuyo/Balseiro/4 semestre/Imagenes medicas/Practica/Practica 4/src/resultados/noise/img_ex_detects_%d_angle_%d.mat', p, angle_fijo), "img_ex_detects")
    end
    % Noise level.
    eta = noises(i);
    delta = eta*norm(b_ex);

    % Add noise to the rhs.
    randn('state',0);
    e = randn(size(b_ex));
    e = delta*e/norm(e);
    b = b_ex + e;

    % reconstruyo la imagen mediante kaaczmarz, simekaczmarz y randkaczmarz
    Xkacz = kaczmarz(A,b,niter);
    Xsymk = symkaczmarz(A,b,niter);
    Xrand = randkaczmarz(A,b,niter);
    Xsart = sart(A,b,niter);

    Xkacz = reshape(Xkacz, N, N);
    Xsymk = reshape(Xsymk, N, N);
    Xrand = reshape(Xrand, N, N);
    Xsart = reshape(Xsart, N, N);

    % Guardo las imagenes
    save(sprintf('/Users/maximilianogatto/OneDrive - Universidad Nacional de Cuyo/Balseiro/4 semestre/Imagenes medicas/Practica/Practica 4/src/resultados/noise/img_kacz_detects_%d_angle_%d_noise_%d.mat', p, angle_fijo, eta), "Xkacz")
    save(sprintf('/Users/maximilianogatto/OneDrive - Universidad Nacional de Cuyo/Balseiro/4 semestre/Imagenes medicas/Practica/Practica 4/src/resultados/noise/img_symk_detects_%d_angle_%d_noise_%d.mat', p, angle_fijo, eta), "Xsymk")
    save(sprintf('/Users/maximilianogatto/OneDrive - Universidad Nacional de Cuyo/Balseiro/4 semestre/Imagenes medicas/Practica/Practica 4/src/resultados/noise/img_randk_detects_%d_angle_%d_noise_%d.mat', p, angle_fijo,eta), "Xrand")
    save(sprintf('/Users/maximilianogatto/OneDrive - Universidad Nacional de Cuyo/Balseiro/4 semestre/Imagenes medicas/Practica/Practica 4/src/resultados/noise/img_sart_detects_%d_angle_%d_noise_%d.mat', p, angle_fijo, eta), "Xsart")

end