import cv2 as cv

class PanaceaFlow:
    img1 = ''
    img1_mss = ''
    img2 = ''
    def __init__(self, img1, img2):
        self.img1_path = img1
        self.img2_path = img2
        self.img1 = cv.imread(img1, cv.IMREAD_GRAYSCALE)
        self.img2 = cv.imread(img2, cv.IMREAD_GRAYSCALE)
    def mss_init(self, img1): # img2 is always attached to optical flow's
        self.img1_mss_path = img1
        self.img1_mss = cv.imread(img1, cv.IMREAD_GRAYSCALE)
    def change_image(self, img1, img2):
        self.__init__(img1, img2)
    def calculate(self):
        pass