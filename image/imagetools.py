import cv2


class ImageTools():
    """Base class for image analysis with simple and universal functions."""

    def Threshold():
        """Converts image to binary format using same threshold for all pixels"""
        return None

    def LocalThreshold():
        """Converts image to binary format using local threshold value for each pixel"""
        return None

    def GaussianBlur():
        return None

    def SubtractBackground():
        """Lowers the intensity of the background"""
        return None

    def BinaryWatershed():
        """Performs Watershed algorithm (splitting circles), binary image on input"""
        return None

    def greyWatershed():
        """Performs Watershed algorithm (splitting circles), greyscale image on input"""
        return None
