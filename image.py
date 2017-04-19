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
    imgfile = "udacity/IMG/center_2016_12_01_13_30_48_287.jpg"
    imgfile = "run1/IMG/center_2017_04_15_18_19_55_334.jpg"
    oimg = cv2.imread(base + imgfile)
    img = preprocess(oimg, "comma.ai")
    plt.figure(0), plt.imshow(cv2.cvtColor(oimg, code=cv2.COLOR_BGR2RGB))
    plt.figure(1), plt.imshow(img)
    plt.show()
