from . import imagetools as it
import cv2
import numpy as np
import tiffcapture as tc


class SelectNucleus(it.ImageTools):
    """Find each nucleus on the image, split them, enumarate and create new images with nucleus at the center"""

    def __init__(self, path, debug, output_name):
        """Load image etc."""
        self.debug = debug
        self.output_name = output_name
        self.current_frame = 0
        self.tiff = tc.opentiff(path)

        # just temporary
        self.grayscale = True
        self.big_contour = []
        self.centers = []
        self.images = []

        """Loads all images from TIFF"""
        _,temp_img = self.tiff.retrieve()
        temp_img = np.array(temp_img, dtype=np.uint8)
        self.images.append(temp_img)

        for temp_img in self.tiff:
            self.images.append(np.array(temp_img, dtype=np.uint8))

        self.img = np.array(self.images[0], dtype=np.uint8)
        self.org_img = self.img.copy()
        self.all_frames = len(self.images)



    def convert_to_grey_scale(self):
        """creates a temporary greyscale image, later on we will need RGB one too"""

        if self.grayscale:
            pass
        else:
            # convert to grayscale
            pass

        # We need some blurring to reduce high frequency noise
        self.img = cv2.GaussianBlur(self.img, (5, 5), 0)

    def split_nucleus(self):
        """split image and into few parts, one neclues in each"""
        _, thresh = cv2.threshold(self.img, 100, 255, 0)
        kernel = np.ones((2, 2), np.uint8)
        thresh = cv2.erode(thresh, kernel, iterations=1)
        kernel = np.ones((10, 10), np.uint8)
        thresh = cv2.dilate(thresh, kernel, iterations=1)

        self.img, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for i in contours:
            area = cv2.contourArea(i)  # --- find the contour having biggest area ---
            if area > 5000:
                self.big_contour.append(i)

        color = [255, 255, 255]
        stencil = np.zeros(self.img.shape).astype(self.img.dtype)
        cv2.fillPoly(stencil, self.big_contour, color)
        result = cv2.bitwise_and(self.org_img, stencil)

        if not self.debug:
            for idx, contour in enumerate(self.big_contour):
                (x, y, w, h) = cv2.boundingRect(contour)
                temp_img = result.copy()
                roi = temp_img[y:y+h, x:x+w]
                cv2.imwrite(str(self.output_name) + '_' + 'frame(' + str(self.current_frame)+ ')_' +  str(idx) + '_black' + '.jpg', roi)
                temp_img = self.org_img.copy()
                roi = temp_img[y:y+h, x:x+w]
                cv2.imwrite(str(self.output_name) + '_' + 'frame(' + str(self.current_frame)+ ')_' + str(idx)  + '.jpg', roi)                


        if self.debug:
            self.img = cv2.cvtColor(self.img, cv2.COLOR_GRAY2RGB)
            for contour in self.big_contour:
                (x, y, w, h) = cv2.boundingRect(contour)
                cv2.rectangle(self.org_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cv2.drawContours(self.org_img, self.big_contour, -1, (255, 0, 0), 3)
            cv2.imshow('i', self.org_img)
            cv2.waitKey(0)            



    def center_nucles(self):
        # calculate the center of the mass
        for c in self.big_contour:
            M = cv2.moments(c)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            self.centers.append([cX, cY])
            if self.debug:
                cv2.circle(self.org_img, (cX, cY), 7, (255, 255, 255), -1)

    @property
    def transformation(self):
        """Performs mathematical transformation of coordinates based of the shape changes between referenced nucleus
        and current nucleus"""
        return None

    def show_image(self):
        """Displays frame of the image"""
        cv2.imshow('i', self.org_img)
        cv2.waitKey(0)
        return None

    def set_frame(self, frame):
        self.current_frame = frame
        self.big_contour = []
        self.centers = []
        self.img = np.array(self.images[frame], dtype=np.uint8)
        self.org_img = self.img.copy()

