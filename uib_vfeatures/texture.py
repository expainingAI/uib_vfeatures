from skimage.feature import greycomatrix, greycoprops
import numpy as np

@staticmethod
def texture_features(distances, angles, properties, image):
    """
    Calc the texture properties of a greyscale image

    :param distances: List of pixel pair distance offsets
    :param angles: List of pixel pair angles in radians.
    :param properties:
    :param image: Greyscale image in uint8
    :return:
    """
    glcm = greycomatrix(image, distances=distances, angles=angles, symmetric=True, normed=True)

    return np.hstack([greycoprops(glcm, prop).ravel() for prop in properties])

