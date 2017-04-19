"""
Image preprocessing
"""
import cv2
import matplotlib.pyplot as plt


CROPPED_SHAPE = (70, 320, 3)


def preprocess(image):
    """
    preprocess camera image
    """
    ret = image[60:130, 0:320]
    ret = cv2.cvtColor(ret, cv2.COLOR_BGR2HSV)
    ret = (ret / 255) - 0.5
    return ret


if __name__ == '__main__':
    base = ""
    imgfile = "IMG/center_2016_12_01_13_33_04_891.jpg"
    oimg = cv2.imread(base + imgfile)
    img = preprocess(oimg)
    plt.figure(0), plt.imshow(cv2.cvtColor(oimg, code=cv2.COLOR_BGR2RGB))
    plt.figure(1), plt.imshow(img)
    plt.show()
