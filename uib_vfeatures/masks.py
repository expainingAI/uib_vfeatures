from matplotlib import pyplot as plt
from copy import copy
import cv2
from uib_vfeatures.contours import Contours


class Masks:
    @staticmethod
    def solidity(mask, screen=False):
        """
        Calculates the proportion between the area of the object in the mask and the convex-hull
        :param mask: 1 channel image
        :param screen:
        :return:
        """
        cnt = Masks.extract_contour(mask)

        if screen:
            hull = cv2.convexHull(cnt, returnPoints=False)
            defects = cv2.convexityDefects(cnt, hull)
            mask_cp = copy(mask)
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                start = tuple(cnt[s][0])
                end = tuple(cnt[e][0])
                cv2.line(mask_cp, start, end, [100, 100, 100], 10)

            plt.imshow(mask_cp)
            plt.show()

        return Contours.solidity(cnt)

    @staticmethod
    def convex_hull_perimeter(mask):
        """
        Calculates the perimeter of the convex hull of the biggest object into the mask
        :param mask: 1 channel image
        :return:
        """
        cnt = Masks.extract_contour(mask)

        return Contours.convex_hull_perimeter(cnt)

    @staticmethod
    def convex_hull_area(mask):
        """
        Calculates the area of the convex hull of the biggest object into the mask
        :param mask: 1 channel image
        :return:
        """
        cnt = Masks.extract_contour(mask)

        return Contours.convex_hull_area(cnt)

    @staticmethod
    def bounding_box_area(mask):
        """
        Calculates the area of the bounding box of the biggest object into the mask

        :param mask: 1 channel image
        :return:
        """
        cnt = Masks.extract_contour(mask)

        return Contours.bounding_box_area(cnt)

    @staticmethod
    def rectungalirity(mask):
        """
        Calculates the proportion between the real area of the mask and the bounding box
        :param mask: 1 channel image
        :return:
        """
        cnt = Masks.extract_contour(mask)

        return Contours.bounding_box_area(cnt)

    @staticmethod
    def min_r(mask):
        """
        Calculates the minor radius of the ellipse from the biggest object of the mask
        :param mask:
        :return:
        """
        cnt = Masks.extract_contour(mask)

        return Contours.min_r(cnt)

    @staticmethod
    def max_r(mask):
        """
        Calculates the radius of the enclosing circle of the contour
        :param contour:
        :return:
        """

        cnt = Masks.extract_contour(mask)

        return Contours.max_r(cnt)

    @staticmethod
    def feret(mask):
        """
        Calculates the major diagonal of the enclosing ellipse from the biggest object in the mask
        :param contour:
        :return:
        """

        cnt = Masks.extract_contour(mask)

        return Contours.feret(cnt)

    @staticmethod
    def breadth(mask):
        """
        Calculates the minor diagonal of the fitting ellipse. It's equal to the width of the object
        :param mask:
        :return:
        """
        cnt = Masks.extract_contour(mask)

        return Contours.breadth(cnt)

    @staticmethod
    def circularity(mask):
        """
        Calculates the likeliness of an object to a circle
        :param mask: 1 channel image
        :return:
        """
        cnt = Masks.extract_contour(mask)

        return Contours.circularity(cnt)

    @staticmethod
    def roundness(mask):
        """
        Circularity corrected by the aspect ratio
        ref : https://progearthplanetsci.springeropen.com/articles/10.1186/s40645-015-0078-x
        :param mask: 1 channel image
        :return:
        """
        cnt = Masks.extract_contour(mask)

        return Contours.roundness(cnt)

    @staticmethod
    def feret_angle(mask):
        """
        Calculates the feret angle from the horizontal
        :param mask:
        :return:
        """
        cnt = Masks.extract_contour(mask)

        return Contours.feret_angle(cnt)

    @staticmethod
    def eccentricity(mask, screen=False):
        """
        Calculates how much the conic section deviates from being circular

        :param mask: 1 channel image
        :param screen:
        :return:
        """
        cnt = Masks.extract_contour(mask)

        if screen:
            mask_cp = copy(mask)
            ellipse = cv2.fitEllipse(cnt)
            cv2.ellipse(mask_cp, ellipse, (100, 100, 100), 7)
            plt.imshow(mask_cp)
            plt.show()

        return Contours.eccentricity(cnt)

    @staticmethod
    def center(mask):
        """
        Calculates the center of the biggets object into the mask
        :param mask:
        :return:
        """
        cnt = Masks.extract_contour(mask)

        return Contours.center(cnt)

    @staticmethod
    def sphericity(mask):
        """
        Proportion between the major and the minor feret

        :param mask:
        :return:
        """
        cnt = Masks.extract_contour(mask)

        return Contours.sphericity(cnt)

    @staticmethod
    def aspect_ratio(mask):
        """
        Proportional relationship between its width and it's height
        :param contour:
        :return:
        """
        cnt = Masks.extract_contour(mask)

        return Contours.aspect_ratio(cnt)

    @staticmethod
    def area_equivalent_diameter(mask):
        """
        The diamater of the real area of the mask
        :param mask:
        :return:
        """
        cnt = Masks.extract_contour(mask)

        return Contours.area_equivalent_diameter(cnt)

    @staticmethod
    def perimeter_equivalent_diameter(mask):
        """
        The diameter of the real perimeter of the contour
        """
        cnt = Masks.extract_contour(mask)

        return Contours.perimeter_equivalent_diameter(cnt)

    @staticmethod
    def equivalent_ellipse_area(mask):
        """
        The area of the equivalent ellipse
        :param mask:
        :return:
        """
        cnt = Masks.extract_contour(mask)

        return Contours.equivalent_ellipse_area(cnt)

    @staticmethod
    def compactness(mask):
        """
        Proportion between area and the shape of the ellipse
        :param mask:
        :return:
        """
        cnt = Masks.extract_contour(mask)

        return Contours.compactness(cnt)

    @staticmethod
    def area(mask):
        """
        Calc the area of the object of the mask

        :param mask: 1 channel image
        :return:
        """
        cnt = Masks.extract_contour(mask)

        return cv2.contourArea(cnt)

    @staticmethod
    def convexity(mask):
        cnt = Masks.extract_contour(mask)

        return Contours.convexity(cnt)

    @staticmethod
    def shape(mask):
        """
        Relation between perimeter and area. Calc the elongation of an object
        :param mask:
        :return:
        """
        cnt = Masks.extract_contour(mask)

        return Contours.shape(cnt)

    @staticmethod
    def perimeter(mask):
        """
        Calc the perimeter of the object in the mask

        :param mask: 1 channel image
        :return:
        """
        cnt = Masks.extract_contour(mask)

        return Contours.perimeter(cnt)

    @staticmethod
    def extract_contour(mask):
        if len(mask.shape) != 2:
            raise ValueError('Image is not a maks, multiples channels of color')

        _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        return contours[0]

    @staticmethod
    def mean_sdv_lab(img, channel=0):
        """
        Calc the mean an the standard desviation of a channel of a CIE-LAB image
        :param img: Image with three channels
        :param channel: 0 => all three, 1 => L , 2 => A, 3 => B
        :return:
        """
        img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

        return Masks.mean_sdv(img_lab, channel)

    @staticmethod
    def mean_sdv_rgb(img, channel=0):
        """
        Calc the mean an the standard desviation of a channel of a RGB image
        :param img: Image with three channels
        :param channel: 0 => all three, 1 => L , 2 => A, 3 => B
        :return:
        """
        return Masks.mean_sdv(img, channel)

    @staticmethod
    def mean_sdv_hsv(img, channel=0):
        """
        Calc the mean an the standard desviation of a channel of a HSV image
        :param img: Image with three channels
        :param channel: 0 => all three, 1 => L , 2 => A, 3 => B
        :return:
        """
        img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        return Masks.mean_sdv(img_lab, channel)

    @staticmethod
    def mean_sdv(img, channel=0):
        if channel != 0:
            if channel == 1:
                chann, _, _ = cv2.split(img)
            elif channel == 2:
                _, chann, _ = cv2.split(img)
            else:
                _, _, chann = cv2.split(img)

        else:
            chann = copy(img)

        return cv2.meanStdDev(chann)
