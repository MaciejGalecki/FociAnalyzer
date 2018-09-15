from . import imagetools as it
import cv2
import numpy as np
import tiffcapture as tc


class SelectNucleus(it.ImageTools):
    """Find each nucleus on the image, split them, enumarate and create new images with nucleus at the center"""
    tiff = None
    img = None

    def __init__(self, path):
        """Load image etc."""
        self.tiff = tc.opentiff(path)
        self.img = self.tiff.read()[1]
        self.img = np.array(self.img, dtype=np.uint8)

    def load_image(self, path):
        """Loads a single image to opencv format"""
        return None

    def convert_to_grey_scale(self):
        """creates a temporary greyscale image, later on we will need RGB one too"""
        return None

    def split_nucleus(self):
        """split image and into few parts, one neclues in each"""
        _, thresh = cv2.threshold(self.img, 100, 255, 0)
        kernel = np.ones((2, 2), np.uint8)
        thresh = cv2.erode(thresh, kernel, iterations=1)
        kernel = np.ones((10, 10), np.uint8)
        thresh = cv2.dilate(thresh, kernel, iterations=1)

        cv2.imshow('a', self.img)
        self.img, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        big_contour = []
        max = 0
        for i in contours:
            area = cv2.contourArea(i)  # --- find the contour having biggest area ---
            if (area > 5000):
                max = area
                big_contour.append(i)

        self.img = cv2.cvtColor(self.img, cv2.COLOR_GRAY2RGB)

        # draw rectangles around contours
        for contour in big_contour:
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(self.img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.drawContours(self.img, big_contour, -1, (255, 0, 0), 3)

        cv2.imshow('i', self.img)
        cv2.waitKey(0)

    def center_nucles(self):
        """Find the mass center of the nucleus and place it at the center of the image"""
        return None

    @property
    def transformation(self):
        """Performs mathematical transformation of coordinates based of the shape changes between referenced nucleus
        and current nucleus"""
        return None

    def show_image(self):
        """Displays frame of the image"""
        return None
