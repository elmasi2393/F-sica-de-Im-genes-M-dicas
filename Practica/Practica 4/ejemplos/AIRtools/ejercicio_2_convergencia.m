% Importa la libreria AIRtools
addpath(genpath('/Users/maximilianogatto/OneDrive\ -\ Universidad\ Nacional\ de\ Cuyo/Balseiro/4\ semestre/Imagenes\ medicas/Practica/Practica\ 4/ejemplos/AIRtools'));

close all;
fprintf('Ejercicio 2 - Veo las imagenes en funcion del numero de iteraciones\n');

% Fijo el numero de angulos y de dectores, e itero para cada nivel de ruido

% Seteo los parametros de la cual quier calcular la reconstrucción de la imagen
N = 240;           % Imagen de salida
N_iter = 4;  % numero de iteraciones para obtener la imagen
% N_noises = 10;
angle_fijo = 100;
detect_fijo = 128;
niter = linspace(10, 100, N_iter); % Creo los valores sobre los cuales iterar
% Creo los valores sobre los cuales iterar



% itero sobre los valores de n_angles y n_detects
for i = 1:N_iter
    % ---------------- VARIANDO RUIDOS -----------------------
    fprintf('Iteracion %d de %d\n', i, N_iter)
    fprintf('Iteraciones: %d\n', niter(i))
    theta = linspace(0, 180, angle_fijo);
    p = detect_fijo;
    [A,b_ex,x_ex] = paralleltomo(N,theta,p);
    % Guardo la imagen exacta si estamos en el primer nivel de ruido
    if i ==1
        img_ex_detects = reshape(x_ex, N, N);
        save(sprintf('/Users/maximilianogatto/OneDrive - Universidad Nacional de Cuyo/Balseiro/4 semestre/Imagenes medicas/Practica/Practica 4/src/resultados/iter_1/img_ex_detects_%d_angle_%d.mat', p, angle_fijo), "img_ex_detects")
    end
    b = b_ex;

    % reconstruyo la imagen mediante kaaczmarz, simekaczmarz y randkaczmarz
    Xkacz = kaczmarz(A,b,niter(i));
    Xsymk = symkaczmarz(A,b,niter(i));
    Xrand = randkaczmarz(A,b,niter(i));
    Xsart = sart(A,b,niter(i));

    Xkacz = reshape(Xkacz, N, N);
    Xsymk = reshape(Xsymk, N, N);
    Xrand = reshape(Xrand, N, N);
    Xsart = reshape(Xsart, N, N);

    % Guardo las imagenes
    save(sprintf('/Users/maximilianogatto/OneDrive - Universidad Nacional de Cuyo/Balseiro/4 semestre/Imagenes medicas/Practica/Practica 4/src/resultados/iter_1/img_kacz_detects_%d_angle_%d_iter_%d.mat', p, angle_fijo, niter(i)), "Xkacz")
    save(sprintf('/Users/maximilianogatto/OneDrive - Universidad Nacional de Cuyo/Balseiro/4 semestre/Imagenes medicas/Practica/Practica 4/src/resultados/iter_1/img_symk_detects_%d_angle_%d_iter_%d.mat', p, angle_fijo, niter(i)), "Xsymk")
    save(sprintf('/Users/maximilianogatto/OneDrive - Universidad Nacional de Cuyo/Balseiro/4 semestre/Imagenes medicas/Practica/Practica 4/src/resultados/iter_1/img_randk_detects_%d_angle_%d_iter_%d.mat', p, angle_fijo,niter(i)), "Xrand")
    save(sprintf('/Users/maximilianogatto/OneDrive - Universidad Nacional de Cuyo/Balseiro/4 semestre/Imagenes medicas/Practica/Practica 4/src/resultados/iter_1/img_sart_detects_%d_angle_%d_iter_%d.mat', p, angle_fijo, niter(i)), "Xsart")
end