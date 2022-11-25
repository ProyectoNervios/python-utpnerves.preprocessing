"""
=======
Proceso
=======

Descripción de este archivo.
"""


import numpy as np
import cv2 as cv
import logging
from typing import TypeVar

Image = TypeVar('Image', bound=np.ndarray)


# Método Balance de Blancos
def limitar(im: Image) -> Image:
    """
    Descripción corta, una línea y termina en punto.

    Descripción tan larga como sea necesaria, multilinea.
    Descripción tan larga como sea necesaria, multilinea.
    Descripción tan larga como sea necesaria, multilinea.

    Parameters
    ----------
    im
        Description of parameter `im`.
    y_pred

    Returns
    -------
    im
        Descripción del objeto que retorna.
    """

    Mayores = im > 255
    im[Mayores] = 255
    Menores = im < 0
    im[Menores] = 0
    return im


def imadjust(imagen: Image, low_in: float, hig_in: float, low_out: float, hig_out: float, gamma: float) -> Image:
    """
    Descripción corta, una línea y termina en punto.

    Descripción tan larga como sea necesaria, multilinea.
    Descripción tan larga como sea necesaria, multilinea.
    Descripción tan larga como sea necesaria, multilinea.

    Parameters
    ----------
    imagen
        Description of parameter `imagen`.
    low_in
        Description of parameter
    hig_in
        Description of parameter
    low_out
        Description of parameter
    hig_out
        Description of parameter
    gamma
        Description of parameter

    Returns
    -------
    imagen_mod
        Descripción del objeto que retorna.
    """

    imagen_mod = np.zeros(imagen.shape, dtype=np.uint8)
    imagen_mod = low_out + (hig_out - low_out) * ((imagen - low_in) / (hig_in - low_in)) ** gamma
    imagen_mod = np.round(255 * imagen_mod)
    imagen_mod = limitar(imagen_mod)
    imagen_mod = np.uint8(imagen_mod)

    return imagen_mod


def white_balance(image: Image) -> Image:
    """
    Descripción corta, una línea y termina en punto.

    Descripción tan larga como sea necesaria, multilinea.
    Descripción tan larga como sea necesaria, multilinea.
    Descripción tan larga como sea necesaria, multilinea.

    Parameters
    ----------
    imagen
        Description of parameter.

    Returns
    -------
    canal_balance
        Descripción del objeto que retorna.
    """

    canal = image.astype('float') / 255.0
    min_canal = np.percentile(canal, 1)
    max_canal = np.percentile(canal, 99)
    canal_balance = imadjust(canal, min_canal, max_canal, 0.0, 1.0, 1)
    return canal_balance


def manipulacion_datos_prediccion(imagen: Image) -> Image:
    """
    Descripción corta, una línea y termina en punto.

    Descripción tan larga como sea necesaria, multilinea.
    Descripción tan larga como sea necesaria, multilinea.
    Descripción tan larga como sea necesaria, multilinea.

    Parameters
    ----------
    imagen
        Description of parameter.

    Returns
    -------
    imagen
        Descripción del objeto que retorna.
    """

    logging.debug('-' * 60)
    logging.debug('Loading and preprocessing train data...')
    logging.debug('-' * 60)

    imagen = imagen.reshape((1,
                             imagen.shape[0],
                             imagen.shape[1], 1))
    imagen = imagen / 255.
    return imagen


def imagen_proceso(imagen):
    balance = white_balance(imagen)
    resized_imagen = cv.resize(balance, (320, 180), interpolation=cv.INTER_AREA)
    resized_imagen = np.array(resized_imagen)
    imagenTensor = manipulacion_datos_prediccion(resized_imagen)
    return imagenTensor


def aumento_tam(imagen, tam_nuevo):
    resized_imagen = cv.resize(imagen,
                               (tam_nuevo[1], tam_nuevo[0]),
                               interpolation=cv.INTER_AREA)
    # resized_imagen = Limitar(resized_imagen)
    return resized_imagen


def remover_areas(imagen, min_size=500):
    nb_blobs, im_with_separated_blobs, stats, _ = cv.connectedComponentsWithStats(np.uint8(imagen))
    sizes = stats[:, -1]
    sizes = sizes[1:]
    nb_blobs -= 1
    im_result = np.zeros(imagen.shape)

    for blob in range(nb_blobs):
        if sizes[blob] >= min_size:
            # see description of im_with_separated_blobs above
            im_result[im_with_separated_blobs == blob + 1] = 1.0
    return im_result


def cuadrar_rect(mascara):
    img = np.array(mascara, dtype=np.uint8)
    imgRGB = cv.cvtColor(img.copy(), cv.COLOR_GRAY2RGB)
    a, ctrs = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # print(a)
    boxes = []
    for ctr in a:
        x, y, w, h = cv.boundingRect(ctr)
        boxes.append([x, y, w, h])

    for box in boxes:
        top_left = (box[0], box[1])
        bottom_right = (box[0] + box[2], box[1] + box[3])
        # para que quede solo el recuadro, cambiar el -1 por un 2
        cv.rectangle(imgRGB, top_left, bottom_right, (255, 255, 255), -1)

    return imgRGB[:, :, 0] / 255.0


def dim_rec(mascara, imagen):
    cnts, _ = cv.findContours(np.uint8(mascara), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    mask = imagen[cnts[0][0][0][1]: cnts[0][1][0][1], cnts[0][0][0][0]: cnts[0][2][0][0]]
    return mask
