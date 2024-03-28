%EJERCICIO 2 - Expectation maximization

N=240; %
n_angles = 100; % 
n_detect = 128; % 
niter =10; % 10

fprintf('Comenzando reconstruccion\n');

theta = linspace(0,180,n_angles);
p = n_detect;
[A,b_ex,x_ex] = paralleltomo(N,theta,p);
x_ex = reshape(x_ex,N,N);
save('/Users/maximilianogatto/OneDrive - Universidad Nacional de Cuyo/Balseiro/4 semestre/Imagenes medicas/Practica/Practica 4/src/resultados/exp_max/original_em.mat', "x_ex")

figure,imagesc(x_ex), colormap gray, axis image, title('x original')

x_0 = zeros(size(x_ex));
it = round(linspace(1,niter,niter));

for i = it
    fprintf('Iteracion %d\n',i);
    tic
    [x_0,options] = ex_max(A,b_ex,i);
    toc
    x_0 = reshape(x_0,N,N);
    figure,
    save(sprintf('/Users/maximilianogatto/OneDrive - Universidad Nacional de Cuyo/Balseiro/4 semestre/Imagenes medicas/Practica/Practica 4/src/resultados/exp_max/img_em_it_%d.mat', i), "x_0")

    imagesc(x_0), colormap gray, axis image, title('reconstruccion con iteraciones',i)

    % save(sprintf('/Users/franpereyraaponte/Library/CloudStorage/OneDrive-UTNSanFrancisco/I. BALSEIRO/4to Cuatrimestre 2024/Imagenes Médicas/Práctica 4 /MATLAB/EXPMAX/EM_%d',K), 'x_0');
end

fprintf('Done!\n');