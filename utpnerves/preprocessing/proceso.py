"""
=======
Proceso
=======

Este modulo provee funciones para procesar capturas de pantalla
de un ecográfo con el fin de segmentar y extraer la imagen
de ultrasonido y la información de los parámetros del ecógrafo.
"""


import numpy as np
import cv2 as cv
import logging
from typing import TypeVar

Image = TypeVar('Image', bound=np.ndarray)

class Proceso:

    def limitar(self, im: Image) -> Image:
        """
        Esta función permite acotar una imagen en el rango
        de 0 a 255. Esto se hace ncesario ya que al aplicar
        algunas funciones de opencv o de alguna librería de
        procesamiento de imágenes, la imagen resultante puede
        encontrarse en rangos comprendidos entre 0 y 255.

        Parameters
        ----------
        im
           Imagen de un Canal, por ejemplo R del RGB.

        Returns
        -------
        im
            Imagen acotada
        """

        Mayores = im > 255
        im[Mayores] = 255
        Menores = im < 0
        im[Menores] = 0
        return im


    def imadjust(
            self, imagen: Image, low_in: float, hig_in: float,
                        low_out: float, hig_out: float, gamma: float) -> Image:
        """
        Esta función permite aplicar una transformación
        lineal a una imagen(canal). Esta ecuación se encuentra
        especificada en el texto de Gonzales Wood
        "Digital Image Processing".

        Parameters
        ----------
        imagen
            Imagen de un canal, por ejemplo R del RGB.
        low_in
            El valor más pequeño que tiene la imagen(canal) original.
        hig_in
            El valor más grande que tiene la imagen(canal) original.
        low_out
            El valor más pequeño que desea mapear la imagen original.
        hig_out
            El valor más grande que desea mapear la imagen original.
        gamma
            Valor que controla el brillo de la imagen resultante.

        Returns
        -------
        imagen_mod
            Imagen con la transformación lineal aplicada.
        """

        imagen_mod = np.zeros(imagen.shape, dtype=np.uint8)
        imagen_mod = (
                    low_out
                        + (hig_out-low_out)
                        * ((imagen-low_in) / (hig_in-low_in))**gamma)

        imagen_mod = np.round(255 * imagen_mod)
        imagen_mod = self.limitar(imagen_mod)
        imagen_mod = np.uint8(imagen_mod)

        return imagen_mod


    def white_balance(self, image: Image) -> Image:
        """
        Esta función permite mapear una imagen a sus valores extremos.
        Es posible encontrar este algortimo implementado en muchos programas
        como GIMP o ADOBE PHOTOSHOP. Los valores de cada variable que se
        especifica en la función se extrajeron de la función "imadjust"
        que emplea Matlab.

        Parameters
        ----------
        imagen
            Imagen original con los 3 canales, por ejemplo RGB.

        Returns
        -------
            Imagen con la transformación balance de blancos.
        """

        canal = image.astype('float') / 255.0
        min_canal = np.percentile(canal, 1)
        max_canal = np.percentile(canal, 99)
        canal_balance = self.imadjust(canal, min_canal, max_canal, 0.0, 1.0, 1)
        return canal_balance


    def manipulacion_datos_prediccion(self, imagen: Image) -> Image:
        """
        Esta función permite realizar el procesamiento necesario
        para que la imagen sea leída por el modelo en la etapa
        de inferencia. Primero se cambian las dimensiones
        de la imagen y se transforma en un tensor y después se
        realiza un escalamiento.

        Parameters
        ----------
        imagen
            Imagen canal, por ejemplo R del RGB.

        Returns
        -------
        imagen
            Imagen mapeada al rango [0,1] y en forma de tensor.
        """

        logging.debug('-' * 60)
        logging.debug('Loading and preprocessing train data...')
        logging.debug('-' * 60)

        imagen = imagen.reshape((1,
                                         imagen.shape[0],
                                                                 imagen.shape[1], 1))
        imagen = imagen / 255.
        return imagen


    def imagen_proceso(self, imagen):
        """
        Esta función implementa otras funciones
        definidas anteriormente para obtener
        el procesamiento definitivo de las imagenes:
        Balance de Blancos, Reducción del tamaño de
        una imagen, Mapeo de la imagen al rango [0,1] y
        la transformación de la imagen en un tensor.

        Parameters
        ----------
        imagen
            Imagen en los 3 canales, por ejemplo RGB.

        Returns
        -------
        imagen
            Imagen procesada.
        """
        balance = self.white_balance(imagen)
        resized_imagen = cv.resize(
                    balance, (320, 180),
                        interpolation=cv.INTER_AREA)
        resized_imagen = np.array(resized_imagen)
        imagenTensor = self.manipulacion_datos_prediccion(resized_imagen)
        return imagenTensor


    def aumento_tam(self, imagen: Image, tam_nuevo: tuple) -> Image:
        """
        Esta función permite modificar el tamaño de la imagen.
        Este proceso es necesario ya que se requiere volver
        al tamaño original para analizar de forma visual
        el resultado obtenido.

        Parameters
        ----------
        imagen
            Imagen canal, por ejemplo R del RGB.
        tam_nuevo
            Tupla que define el nuevo ancho y largo de la imagen.

        Returns
        -------
        resized_imagen
            Imagen ampliada.
        """
        resized_imagen = cv.resize(
                    imagen,
                        (tam_nuevo[1], tam_nuevo[0]),
                        interpolation=cv.INTER_AREA)
        # resized_imagen = Limitar(resized_imagen)
        return resized_imagen


    def remover_areas(
            self,
                        imagen: Image,
                        min_size: int = 500) -> Image:
        """
        Función que permite mejorar el resultado de la segmentación
        o predicción, removiendo aquellas zonas que sumen cierto valor.

        Parameters
        ----------
        imagen
            Imagen binaria.
        min_size
            Valor del área que se quiere remover. El área se define
            como la cantidad de pixeles que son iguales a 1. El valor
            por defecto es 500, lo que quiere decir se van a remover
            todos los objetos de la imagen que sumen igual o menos de
            500 píxeles.

        Returns
        -------
        im_result
            Imagen binaria filtrada.
        """
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


    def cuadrar_rect(self, mascara: Image) -> Image:
        """
        Esta función permite calcular el rectángulo mas pequeño
        que encierra el objeto.

        Parameters
        ----------
        mascara
            Imagen binaria.

        Returns
        -------
            Imagen binaria (Rectángulo).
        """
        img = np.array(mascara, dtype = np.uint8)
        imgRGB = cv.cvtColor(img.copy(), cv.COLOR_GRAY2RGB)
        a, ctrs = cv.findContours(
                    img,
                        cv.RETR_EXTERNAL,
                        cv.CHAIN_APPROX_SIMPLE)

        # print(a)
        boxes = []
        for ctr in a:
            x, y, w, h = cv.boundingRect(ctr)
            boxes.append([x, y, w, h])

        for box in boxes:
            top_left = (box[0], box[1])
            bottom_right = (box[0] + box[2], box[1] + box[3])
            # para que quede solo el recuadro, cambiar el -1 por un 2
            cv.rectangle(
                            imgRGB, top_left, bottom_right,
                                (255, 255, 255), -1)

        return imgRGB[:, :, 0] / 255.0


    def dim_rec(self, mascara: Image, imagen: Image) -> Image:
        """
        Función para crear la máscara rectángulo.

        Parameters
        ----------
        mascara
            Imagen binaria.
        imagen
            Imagen canal, por ejemplo R del RGB.

        Returns
        -------
        mask
            Porción de la imagen.
        """
        cnts, _ = cv.findContours(
                    np.uint8(mascara), cv.RETR_EXTERNAL,
                        cv.CHAIN_APPROX_SIMPLE)
        mask = imagen[
                    cnts[0][0][0][1] : cnts[0][1][0][1],
                        cnts[0][0][0][0] : cnts[0][2][0][0]]
        return mask

