import cv2
import numpy as np
import matplotlib.pyplot as plt


ORIGINAL_SHAPE = (160, 320, 3)
CROP_SLICES = (slice(50, 150), slice(0, 320))
CROPPED_SHAPE = (100, 320, 3)

model_image_params = {
    "comma.ai": {"shape": CROPPED_SHAPE},
    "trivial": {"shape": CROPPED_SHAPE},
    "nvidia": {"shape": CROPPED_SHAPE}
}


def cv2_normalize(img):
    zeros = np.zeros(img.shape)
    return cv2.normalize(img, zeros, alpha=0.0, beta=1.0,
                         norm_type=cv2.NORM_MINMAX,
                         dtype=cv2.CV_32F)


def cv2_hist_equalize(img):
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return final


def preprocess(image, model=""):
    """
    preprocess camera image
    """
    shape = model_image_params.get(model, {"shape": CROPPED_SHAPE})["shape"]
    resize = shape != CROPPED_SHAPE
    ret = image[CROP_SLICES[0], CROP_SLICES[1], :]
    ret = cv2_normalize(cv2_hist_equalize(ret))
    if resize:
        ret = cv2.resize(ret, shape[1::-1], interpolation=cv2.INTER_LINEAR)
    return ret


if __name__ == '__main__':
    imgfile = "udacity/IMG/center_2016_12_01_13_30_48_287.jpg"
    imgfile = "run1/IMG/center_2017_04_15_18_19_55_334.jpg"
    oimg = cv2.imread(imgfile)
    img = preprocess(oimg, "comma.ai")
    plt.figure(0), plt.imshow(cv2.cvtColor(oimg, code=cv2.COLOR_BGR2RGB))
    plt.figure(1), plt.imshow(cv2.cvtColor(img, code=cv2.COLOR_BGR2RGB))
    plt.show()
