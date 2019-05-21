import numpy as np

class DetectionFrame():
    def __init__(self, xy1, xy2):
        self.x1 = xy1[0]
        self.y1 = xy1[1]
        self.x2 = xy2[0]
        self.y2 = xy2[1]
    
    def __call__(self):
        return [(self.x1, self.y1), (self.x2, self.y2)]
    
    def get_frames_as_list(self, frames_list):
        result = []
        for df in frames_list:
            result.append(df())
        return result

    def merge_frames(self, other, img):
        c_x, c_y = self.weighted_frames_center(other, img)
        h, w = img.shape

        h1 = self.height()
        h2 = other.height()
        w1 = self.width()
        w2 = other.width()

        h_s = int(((h1+h2) / 4) * 1.1)
        w_s = int(((w1+w2) / 4) * 1.1)

        x1 = c_x - h_s
        if x1 < 0:
            x1 = 0
        x2 = c_x + h_s
        if x2 > h-1:
            x2 = h-1
        y1 = c_y - w_s
        if y1 < 0:
            y1 = 0
        y2 = c_y + w_s
        if y2 > w-1:
            y2 = w-1

        x1 = int(x1)
        x2 = int(x2)
        y1 = int(y1)
        y2 = int(y2)

        return DetectionFrame((y1, x1), (y2, x2))
    
    def height(self):
        return self.x2 - self.x1
    
    def width(self):
        return self.y2 - self.y1
    
    def center(self):
        return (( self.x1 + self.x2 ) / 2, (self.y1 + self.y2) / 2)
    
    def centers_distance(self, other):
        self_center = self.center()
        other_center = other.center()
        
        return ((self_center[0] - other_center[0])**2 + (self_center[1] - other_center[1])**2)**.5
    
    def image_fragment(self, img):
        return img[self.x1:self.x2, self.y1:self.y2]
    
    def weighted_image_center(self, img):
        h = self.height()
        w = self.width()

        img_fragment = self.image_fragment(img)
        
        img_sum = np.sum(img_fragment)
        
        h_sum = 0
        for i in range(h):
            h_sum += np.sum(img_fragment[i,:]) * i
        if img_sum == 0:
            h_mean = 0
        else:
            h_mean = h_sum / img_sum
        
        w_sum = 0
        for i in range(w):
            w_sum += np.sum(img_fragment[:,i]) * i
        if img_sum == 0:
            w_mean = 0
        else:
            w_mean = w_sum / img_sum

        return (int(h_mean + self.y1), int(w_mean + self.x1))
    
    def weighted_frames_center(self, other, img):
        img1 = self.image_fragment(img)
        swo1 = self.weighted_image_center(img)
        img2 = other.image_fragment(img)
        swo2 = other.weighted_image_center(img)

        w1 = np.sum(img1) / img1.flatten().shape
        w2 = np.sum(img2) / img2.flatten().shape

        x = (w1 * swo1[0] + w2 * swo2[0]) / (w1 + w2)
        y = (w1 * swo1[1] + w2 * swo2[1]) / (w1 + w2)

        return (int(x), int(y))