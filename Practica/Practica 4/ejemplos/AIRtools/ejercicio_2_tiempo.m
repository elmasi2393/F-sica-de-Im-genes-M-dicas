% Importa la libreria AIRtools
addpath(genpath('/Users/maximilianogatto/OneDrive\ -\ Universidad\ Nacional\ de\ Cuyo/Balseiro/4\ semestre/Imagenes\ medicas/Practica/Practica\ 4/ejemplos/AIRtools'));

close all;
fprintf('Ejercicio 2 - Tiempos de cada algoritmo\n');

% Fijo el numero de angulos y de dectores, e itero para cada nivel de ruido

% Seteo los parametros de la cual quier calcular la reconstrucción de la imagen
N = 240;           % Imagen de salida
N_iter = 10;  % numero de iteraciones para obtener la imagen
% N_noises = 10;
angle_fijo = 100;
detect_fijo = 128;

fprintf('Prueba de tiempo\n');
fprintf('N = %d\n', N);
fprintf('N_iter = %d\n', N_iter);
fprintf('angle_fijo = %d\n', angle_fijo);
fprintf('detect_fijo = %d\n', detect_fijo);
fprintf('------------------------------\n');
% Creo la imagen
theta = linspace(0, 180, angle_fijo);
p = detect_fijo;
[A,b_ex,x_ex] = paralleltomo(N,theta,p);

% Guardo la imagen original para poder hacer la transformada de radon en python
image = reshape(x_ex, [N,N]);
save('/Users/maximilianogatto/OneDrive - Universidad Nacional de Cuyo/Balseiro/4 semestre/Imagenes medicas/Practica/Practica 4/src/resultados/tiempo/img_ex_0.mat', "image")

fprintf('Kaczmarz\n');
tic;
x_kaczmarz = kaczmarz(A,b_ex,N_iter);
toc;

fprintf('Kaczmarz simetrico\n');
tic;
x_kaczmarz_sim = symkaczmarz(A,b_ex,N_iter);
toc;

fprintf('Kaczmarz aleatorio\n');
tic;
x_kaczmarz_ale = randkaczmarz(A,b_ex,N_iter);
toc;

fprintf('Kaczmarz SART\n');
tic;
x_kaczmarz_sart = sart(A,b_ex,N_iter);
toc;


