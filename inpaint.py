from multiprocessing import freeze_support

import cv2
import os
import time
import concurrent.futures

root = "C:/Users/hyper/Downloads/process/"
dst = "C:/Users/hyper/Downloads/output/"
if not os.path.exists(dst):
    os.makedirs(dst)


def process_image(im):
    src = cv2.imread(root + im)

    # Convert the original image to grayscale
    grayScale = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)

    # Kernel for the morphological filtering
    kernel = cv2.getStructuringElement(1, (17, 17))

    # Perform the blackHat filtering on the grayscale image to find the
    # hair countours
    blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel)

    # intensify the hair countours in preparation for the inpainting
    # algorithm
    ret, thresh2 = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)

    # inpaint the original image depending on the mask
    dst_ = cv2.inpaint(src, thresh2, 1, cv2.INPAINT_TELEA)
    cv2.imwrite(dst + str(im)[:-4] + ".png", dst_, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    print(im)


if __name__ == '__main__':
    # with concurrent.futures.ProcessPoolExecutor(max_workers=12) as executor:
    #     filenames = os.listdir(root)
    #     executor.map(process_image, filenames)
    for image_name in os.listdir(root):
        process_image(image_name)
