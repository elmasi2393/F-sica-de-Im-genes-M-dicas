% Importa la libreria AIRtools
addpath(genpath('/Users/maximilianogatto/OneDrive\ -\ Universidad\ Nacional\ de\ Cuyo/Balseiro/4\ semestre/Imagenes\ medicas/Practica/Practica\ 4/ejemplos/AIRtools'));

close all;
fprintf('Prueba de ejercicio 2\n');


% Seteo los parametros de la cual quier calcular la reconstrucción de la imagen
N = 240;           % Imagen de salida
niter = 10;  % numero de iteraciones para obtener la imagen
N_angles = 10;    % No de angulos diferentes dentro del intervalo
N_deetct = 10;
N_noises = 10;
angle_fijo = 100;
detect_fijo = 128;


% Creo los vectores a iterar
n_angles = round(linspace(30, 200, N_angles));
n_detects = round(linspace(60, 300, N_deetct));
noises = linspace(0.01, 0.2, N_noises);


% itero sobre los valores de n_angles y n_detects
for j = 1:N_noises
    fprintf('Iteracion %d de %d\n: noise: %d', j, N_noises, noises(j))
    eta = noises(j);
    for i = 1:N_angles
        % ---------------- VARIANDO DETECTORES -----------------------
        fprintf('Iteracion %d de %d\n', i, N_angles)
        fprintf('Detector: %d\n', n_detects(i))
        theta = linspace(0, 180, angle_fijo);
        p = n_detects(i);
        [A,b_ex,x_ex] = paralleltomo(N,theta,p);
        % Guardo la imagen exacta si estamos en el primer nivel de ruido
        if j ==1
            img_ex_detects = reshape(x_ex, N, N);
            save(sprintf('/Users/maximilianogatto/OneDrive - Universidad Nacional de Cuyo/Balseiro/4 semestre/Imagenes medicas/Practica/Practica 4/src/resultados/noise_240_10/img_ex_detects_%d_angle_%d.mat', p, angle_fijo), "img_ex_detects")
        end
        % Noise level.
        delta = eta*norm(b_ex);

        % Add noise to the rhs.
        randn('state',0);
        e = randn(size(b_ex));
        e = delta*e/norm(e);
        b = b_ex + e;

        % reconstruyo la imagen mediante kaaczmarz, simekaczmarz y randkaczmarz
        Xkacz = kaczmarz(A,b,niter);
        % Xsymk = symkaczmarz(A,b,niter);
        % Xrand = randkaczmarz(A,b,niter);
        % Xsart = sart(A,b,niter);

        Xkacz = reshape(Xkacz, N, N);
        % Xsymk = reshape(Xsymk, N, N);
        % Xrand = reshape(Xrand, N, N);
        % Xsart = reshape(Xsart, N, N);

        % Guardo las imagenes
        save(sprintf('/Users/maximilianogatto/OneDrive - Universidad Nacional de Cuyo/Balseiro/4 semestre/Imagenes medicas/Practica/Practica 4/src/resultados/noise_240_10/img_kacz_detects_%d_angle_%d_noise_%d.mat', p, angle_fijo, eta), "Xkacz")
        % save(sprintf('/Users/maximilianogatto/OneDrive - Universidad Nacional de Cuyo/Balseiro/4 semestre/Imagenes medicas/Practica/Practica 4/src/resulta2/img_symk_detects_%d_angle_%d_noise_%d.mat', p, angle_fijo, eta), "Xsymk")
        % save(sprintf('/Users/maximilianogatto/OneDrive - Universidad Nacional de Cuyo/Balseiro/4 semestre/Imagenes medicas/Practica/Practica 4/src/resulta2/img_randk_detects_%d_angle_%d_noise_%d.mat', p, angle_fijo,eta), "Xrand")
        % save(sprintf('/Users/maximilianogatto/OneDrive - Universidad Nacional de Cuyo/Balseiro/4 semestre/Imagenes medicas/Practica/Practica 4/src/resulta2/img_sart_detects_%d_angle_%d_noise_%d.mat', p, angle_fijo, eta), "Xsart")

        % ------------ VARIANDO ANGULOS ------------------
        fprintf('Angulos\n')
        theta = linspace(0, 180, n_angles(i));
        p = detect_fijo;

        [A,b_ex,x_ex] = paralleltomo(N,theta,p);
        % Guardo la imagen exacta si estamos en el primer nivel de ruido
        if j == 1
            img_ex_angles = reshape(x_ex, N, N);
            save(sprintf('/Users/maximilianogatto/OneDrive - Universidad Nacional de Cuyo/Balseiro/4 semestre/Imagenes medicas/Practica/Practica 4/src/resultados/noise_240_10/img_ex_angles_%d_detecs_%d.mat', n_angles(i), detect_fijo), "img_ex_angles")
        end

        % Noise level.
        delta = eta*norm(b_ex);

        % Add noise to the rhs.
        randn('state',0);
        e = randn(size(b_ex));
        e = delta*e/norm(e);
        b = b_ex + e;

        % reconstruyo la imagen mediante kaaczmarz, simekaczmarz y randkaczmarz
        Xkacz = kaczmarz(A,b,niter);
        % Xsymk = symkaczmarz(A,b,niter);
        % Xrand = randkaczmarz(A,b,niter);
        % Xsart = sart(A,b,niter);

        Xkacz = reshape(Xkacz, N, N);
        % Xsymk = reshape(Xsymk, N, N);
        % Xrand = reshape(Xrand, N, N);
        % Xsart = reshape(Xsart, N, N);

        % Guardo las imagenes
        save(sprintf('/Users/maximilianogatto/OneDrive - Universidad Nacional de Cuyo/Balseiro/4 semestre/Imagenes medicas/Practica/Practica 4/src/resulta2/img_kacz_angles_%d_detecs_%d_noise_%d.mat', n_angles(i), detect_fijo, eta), "Xkacz")
        % save(sprintf('/Users/maximilianogatto/OneDrive - Universidad Nacional de Cuyo/Balseiro/4 semestre/Imagenes medicas/Practica/Practica 4/src/resulta2/img_symk_angles_%d_detecs_%d_noise_%d.mat', n_angles(i), detect_fijo, eta), "Xsymk")
        % save(sprintf('/Users/maximilianogatto/OneDrive - Universidad Nacional de Cuyo/Balseiro/4 semestre/Imagenes medicas/Practica/Practica 4/src/resulta2/img_randk_angles_%d_detecs_%d_noise_%d.mat', n_angles(i), detect_fijo, eta), "Xrand")
        % save(sprintf('/Users/maximilianogatto/OneDrive - Universidad Nacional de Cuyo/Balseiro/4 semestre/Imagenes medicas/Practica/Practica 4/src/resulta2/img_sart_angles_%d_detecs_%d_noise_%d.mat', n_angles(i), detect_fijo, eta), "Xsart")
    end
end