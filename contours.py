import cv2
import math


class Contours:
    @staticmethod
    def area(contour):
        """
        Calc the area of the object of the contour

        :param contour:
        :return:
        """
        return cv2.contourArea(contour)

    @staticmethod
    def perimeter(contour):
        """
        Calc the perimeter of the contour

        :param contour:
        :return:
        """
        return cv2.arcLength(contour, True)

    @staticmethod
    def convex_hull(contour):
        return cv2.convexHull(contour)

    @staticmethod
    def convex_hull_perimeter(contour):
        """
        Return the perimeter of the convex hull of the contour
        :param contour:
        :return:
        """
        return Contours.perimeter(Contours.convex_hull(contour))

    @staticmethod
    def convex_hull_area(contour):
        """
        Return the area of the convex hull of the contour
        :param contour:
        :return:
        """
        return cv2.contourArea(Contours.convex_hull(contour))

    @staticmethod
    def bounding_box_area(contour):
        """
        Return the area of the bounding box of the contour

        :param contour:
        :return:
        """
        return Contours.feret(contour) * Contours.breadth(contour)

    @staticmethod
    def rectangularity(contour):
        """
        Return the proportion between the real area of the contour and the bounding box
        :param contour: 1 channel image
        :return:
        """
        return Contours.area(contour) / Contours.bounding_box_area(contour)

    @staticmethod
    def max_r(contour):
        """
        Return the radius of the enclosing circle of the contour
        :param contour:
        :return:
        """
        (_, _), radius = cv2.minEnclosingCircle(contour)
        return radius

    @staticmethod
    def min_r(contour):
        """
        Return the minor radius of the ellipse from the contour
        :param contour:
        :return:
        """
        return Contours.breadth(contour) / 2

    @staticmethod
    def feret(contour):
        """
        Return the major diagonal of the enclosing ellipse of the contour
        :param contour:
        :return:
        """
        (_, _), (major, _), _ = cv2.fitEllipse(contour)
        return major

    @staticmethod
    def breadth(contour):
        """
        Return the minor diagonal of the ellipse from the contour
        :param contour:
        :return:
        """
        (_, _), (_, minor), _ = cv2.fitEllipse(contour)
        return minor

    @staticmethod
    def feret_angle(contour):
        """
        Return the feret angle from the horizontal
        :param contour:
        :return:
        """

        (_, _), (_, _), angle = cv2.fitEllipse(contour)
        return angle

    @staticmethod
    def roundness(contour):
        """
        Circularity corrected by the aspect ratio
        ref : https://progearthplanetsci.springeropen.com/articles/10.1186/s40645-015-0078-x
        :param contour: 1 channel image
        :return:
        """
        return round(4 * Contours.area(contour) / (math.pi * Contours.feret(contour) * Contours.feret(contour)), 2)

    @staticmethod
    def circularity(contour):
        """
        Calc the likeliness of an object to a circle
        :param contour:
        :return:
        """
        return round(4 * math.pi * Contours.area(contour) / (Contours.perimeter(contour) * Contours.perimeter(contour)),
                     2)

    @staticmethod
    def solidity(contour):
        """
        Calc the proportion between the area of the contour and the convex-hull
        :param contour:
        :return:
        """
        return round(Contours.area(contour) / Contours.convex_hull_area(contour), 2)

    @staticmethod
    def sphericity(contour):
        """
        Proportion between the major diagonal and the minor diagonal
        :param contour:
        :return:
        """
        return Contours.min_r(contour) / Contours.max_r(contour)

    @staticmethod
    def aspect_ratio(contour):
        """
        Proportional relationship between its width and it's height
        :param contour:
        :return:
        """
        return round(Contours.feret(contour) / Contours.breadth(contour), 2)

    @staticmethod
    def area_equivalent_diameter(contour):
        """
        The diamater of the real area of the contour
        :param contour:
        :return:
        """
        return math.sqrt((4 / math.pi) * Contours.area(contour))

    @staticmethod
    def perimeter_equivalent_diameter(contour):
        """
        The diameter of the real perimeter of the contour
        ;param contour:
        """
        return Contours.area(contour) / math.pi

    @staticmethod
    def equivalent_ellipse_area(contour):
        """
        The area of the equivalent ellipse
        :param contour:
        :return:
        """
        return (math.pi * Contours.feret(contour) * Contours.breadth(contour)) / 4

    @staticmethod
    def compactness(contour):
        """
        Proportion between area and the shape of the ellipse
        :param contour:
        :return:
        """
        return math.sqrt((4 / math.pi * Contours.area(contour)) / Contours.feret(contour))

    @staticmethod
    def convexity(contour):
        """
        Calc the convexity of the contour
        :param contour:
        :return:
        """
        return Contours.convex_hull_perimeter(contour) / Contours.perimeter(contour)

    @staticmethod
    def shape(contour):
        """
        Relation between perimeter and area. Calc the elongation of an object
        :param contour:
        :return:
        """
        return math.pow(Contours.perimeter(contour), 2) / Contours.area(contour)

    @staticmethod
    def r_factor(contour):
        return Contours.convex_hull_perimeter(contour) / (Contours.feret(contour) * math.pi)

    @staticmethod
    def eccentricity(contour):
        """
        Calc how much the conic section deviates from being circular

        :param contour:
        :return:
        """
        ellipse = cv2.fitEllipse(contour)
        D = math.fabs((ellipse[0][0] - ellipse[1][0]))
        d = math.fabs(ellipse[0][1] - ellipse[1][1])

        return round(D / d, 2)

    @staticmethod
    def hu_moments(contour):
        return cv2.HuMoments(cv2.moments(contour)).flatten()

    @staticmethod
    def center(contour):
        """
        Center the center of a contour

        :param contour:
        :return:
        """
        m = cv2.moments(contour)
        cx = int(m['m10'] / m['m00'])
        cy = int(m['m01'] / m['m00'])
        return [cx, cy]