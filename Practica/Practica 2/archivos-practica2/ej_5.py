import cv2
import sys 
import numpy as np
import matplotlib.pyplot as plt
import os.path
import pandas as pd
import scienceplots

plt.style.use(['science', 'notebook', 'grid'])


def read_pgm_file(file_name):

    data_dir = os.path.dirname(os.path.abspath(__file__))

    # Test if file exists
    file_path = os.path.join(data_dir, file_name)
    assert os.path.isfile(file_path), 'file \'{0}\' does not exist'.format(file_path)

    img = cv2.imread(file_name,flags=cv2.IMREAD_GRAYSCALE)

    if img is not None:
        print('img.size: ', img.size)
    else:
        print('imread({0}) -> None'.format(file_path))

    return img


def show_img_hist(im):
    vmin = np.amin(im)
    vmax = np.max(im)
    print("Intensity Min: {}   Max:{}".format(vmin,vmax))

    L = int(vmax - vmin)
    print("Number of Levels: {}".format(L))
    fig = plt.figure(figsize=(16,6))
    ax1 = plt.subplot(1, 2, 1)
    ax2 = plt.subplot(1, 2, 2)

    # imgplot = plt.imshow(im/np.amax(im))
    imgplot = ax1.imshow(im,cmap='gray', vmin=vmin, vmax=vmax)
    fig.colorbar(imgplot, ax=ax1)
    # cv2.imshow(infile,img)
    # cv2.waitKey(0)

    hist, bin_edges = np.histogram(im.ravel(),bins=L)
    ax2.bar(bin_edges[:-1], hist)
    plt.show()


def process_pgm_file(im):
    imout = np.fft.fft2(im)

    # Shift la componente de frecuencia cero al centro del espectro
    fft_shifted = np.fft.fftshift(imout)

    # Obtengo la magnitud del espectro
    magnitude_spectrum = 20*np.log1p(np.abs(fft_shifted))

    # Imprimo la imagen transformada para poder guardarla
    plt.figure()
    plt.axis('off')
    plt.imshow(cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U) , cmap='gray')
    plt.show()
    
    # Hace una funcion que me permita dibujar un circulo en la imagen que imprima el espectro de magnitud
    def draw_circle(event, x, y, flags, param):
        nonlocal circle_center, is_drawing, radius, image_copy

        if event == cv2.EVENT_LBUTTONDOWN:
            circle_center = (x, y)
            is_drawing = True
        elif event == cv2.EVENT_MOUSEMOVE:
            if is_drawing:
                radius = int(np.sqrt((x - circle_center[0]) ** 2 + (y - circle_center[1]) ** 2))
                image_copy = np.copy(magnitude_image)
                cv2.circle(image_copy, circle_center, radius, (0, 255, 0), 1)
        elif event == cv2.EVENT_LBUTTONUP:
            is_drawing = False


    cv2.namedWindow("Circle Drawing")
    cv2.setMouseCallback("Circle Drawing", draw_circle)
    
    # Inicializar variables
    circle_center = (-1, -1)
    radius = 0
    is_drawing = False
    im_norm = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)   # Normalizo imagen para mostarla
    magnitude_image = cv2.merge([im_norm, im_norm, im_norm])    # Creo una imagen a color para imprimir el circulo
    image_copy = np.copy(magnitude_image)   # Copia de la imagen para dibujar el circulo
    
    # Creo un arreglo para guardar los circulos
    circles = []

    while True:
        cv2.imshow('Circle Drawing', image_copy)
        # Si presiono enter, guarda el circulo y lo imprime en magnitude_image
        if cv2.waitKey(1) & 0xFF == 13:
            if circle_center[0] != -1 and radius != 0:
                cv2.circle(magnitude_image, circle_center, radius, (0, 255, 0), 1)
                circles.append((circle_center, radius))
                # vuelvo a inicializar las variables
                circle_center = (-1, -1)
                radius = 0
                print("Circle drawn")
            else:
                print("No circle drawn")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

    print(circles)

    # Relleno con 0 la imagen en donde indica el arreglo de circulos
    mask = np.ones(fft_shifted.shape, dtype=np.uint8)
    # print(mask)
    for center, rad in circles:
        print(f'Drawing circle at {center} with radius {rad}')
        cv2.circle(mask, center, rad, 0, -1)

    fft_shifted_masked = fft_shifted * mask.astype(fft_shifted.dtype)

    #Mostrar la imagen con los circulos
    magnitude = 20*np.log1p(np.abs(fft_shifted_masked))
    plt.grid()
    plt.axis('off')
    plt.imshow(cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U), cmap='gray')
    plt.show()

    # Invertir la imagen al espacio original
    imout = np.fft.ifftshift(fft_shifted_masked)
    # Invertir la imagen al espacio original
    img_back = np.fft.ifft2(imout)
    img_back = np.real(img_back)

    return img_back


if __name__ == "__main__":
    
    if(len(sys.argv)<3):
        print("Usage: python read-write-pgm.py [infile.pgm] [outfile.pgm]")
        exit(1)

    infile = sys.argv[1]
    outfile = sys.argv[2]
    
    img = read_pgm_file(infile)

    im = np.array(img)
    print("Size of image: {}".format(im.shape))

    # show_img_hist(im)

    # imout = np.fft.fft2(im)

    # # Shift the zero-frequency component to the center of the spectrum
    # fft_shifted = np.fft.fftshift(imout)

    # # Take the 2D magnitude (absolute value)
    # magnitude_spectrum = 20*np.log1p(np.abs(fft_shifted))
    # print(magnitude_spectrum)

    # # Hace una funcion que me permita dibujar un circulo en la imagen que imprima el espectro de magnitud
    # def draw_circle(event, x, y, flags, param):
    #     global circle_center, is_drawing, radius, image_copy

    #     if event == cv2.EVENT_LBUTTONDOWN:
    #         circle_center = (x, y)
    #         is_drawing = True
    #         print("Mouse Down")
    #     elif event == cv2.EVENT_MOUSEMOVE:
    #         if is_drawing:
    #             radius = int(np.sqrt((x - circle_center[0]) ** 2 + (y - circle_center[1]) ** 2))
    #             image_copy = np.copy(magnitude_image)
    #             cv2.circle(image_copy, circle_center, radius, (0, 255, 0), 1)
    #     elif event == cv2.EVENT_LBUTTONUP:
    #         is_drawing = False
    #         print("Mouse Up")


    # cv2.namedWindow("Circle Drawing")
    # cv2.setMouseCallback("Circle Drawing", draw_circle)

    # # Inicializar variables
    # circle_center = (-1, -1)
    # radius = 0
    # # is_drawing = False
    # im_norm = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)   # Normalizo imagen para mostarla
    # magnitude_image = cv2.merge([im_norm, im_norm, im_norm])    # Creo una imagen a color para imprimir el circulo
    # image_copy = np.copy(magnitude_image)   # Copia de la imagen para dibujar el circulo
    
    # # Creo un arreglo para guardar los circulos
    # circles = []

    # while True:
    #     cv2.imshow('Circle Drawing', image_copy)
    #     # Si presiono enter, guarda el circulo y lo imprime en magnitude_image
    #     if cv2.waitKey(1) & 0xFF == 13:
    #         if circle_center[0] != -1 and radius != 0:
    #             cv2.circle(magnitude_image, circle_center, radius, (0, 255, 0), 1)
    #             circles.append((circle_center, radius))
    #             # vuelvo a inicializar las variables
    #             circle_center = (-1, -1)
    #             radius = 0
    #             print("Circle drawn")
    #         else:
    #             print("No circle drawn")

    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
    # cv2.destroyAllWindows()

    # print(circles)

    # # Relleno con 0 la imagen en donde indica el arreglo de circulos
    # mask = np.ones(fft_shifted.shape, dtype=np.uint8)
    # # print(mask)
    # for center, rad in circles:
    #     print(f'Drawing circle at {center} with radius {rad}')
    #     cv2.circle(mask, center, rad, 0, -1)

    # fft_shifted_masked = fft_shifted * mask.astype(fft_shifted.dtype)

    # print(fft_shifted_masked)

    # # Invertir la imagen al espacio original
    # imout = np.fft.ifftshift(fft_shifted_masked)
    # # Invertir la imagen al espacio original
    # img_back = np.fft.ifft2(imout)
    # img_back = np.real(img_back)

    # Normalizar la imagen para mostrarla
    # is_drawing = False

    img_back = process_pgm_file(im)
    # imprimo la imagen original
    plt.figure()
    plt.axis('off')
    plt.imshow(im, cmap='gray')

    # imprimo la imagen antitranformada
    plt.figure()
    plt.axis('off')
    plt.imshow(img_back, cmap='gray')

    plt.show()

    cv2.imwrite(outfile,img_back,[cv2.IMWRITE_PXM_BINARY,0])
