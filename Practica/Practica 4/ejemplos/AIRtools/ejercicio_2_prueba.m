addpath(genpath('/Users/maximilianogatto/OneDrive\ -\ Universidad\ Nacional\ de\ Cuyo/Balseiro/4\ semestre/Imagenes\ medicas/Practica/Practica\ 4/ejemplos/AIRtools'));

close all;
fprintf('Prueba de ejercicio 2\n');


% Seteo los parametros de la cual quier calcular la reconstrucción de la imagen
N = 128;           % Imagen de salida
niter = 25;  % numero de iteraciones para obtener la imagen
N_angles = 10;    % No de angulos diferentes dentro del intervalo
N_deetct = 10;
N_noises = 10;
angle_fijo = 100;
detect_fijo = 128;


% Creo los vectores a iterar
n_angles = round(linspace(30, 200, N_angles));
n_detects = round(linspace(60, 300, N_deetct));
noises = linspace(0.01, 0.2, N_noises);

eta = 0.01;

for i = 1:N_angles
    theta = linspace(0, 180, n_angles(i));
    p = detect_fijo;
    [A,b_ex,x_ex] = paralleltomo(N,theta,p);

    if i ==1
        figure(1)
        imagesc(reshape(x_ex,N,N)), axis image, colormap gray
        title('Imagen original')
    end
    delta = eta*norm(b_ex);

    % Add noise to the rhs.
    randn('state',0);
    e = randn(size(b_ex));
    e = delta*e/norm(e);
    b = b_ex + e;

    % reconstruyo la imagen mediante kaaczmarz, simekaczmarz y randkaczmarz
    Xkacz = kaczmarz(A,b,niter);

    figure(i+1)
    imagesc(reshape(Xkacz,N,N)), axis image, colormap gray
    title('Reconstruccion con kaczmarz con %d angulos', n_angles(i))


end

