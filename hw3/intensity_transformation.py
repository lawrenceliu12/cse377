import cv2
import numpy

def log_transformation(negative_image):
    log_transform = numpy.log(1 + negative_image)
    log_transform = cv2.normalize(log_transform, None, 0, 255, cv2.NORM_MINMAX)

    return numpy.uint8(log_transform)

def negative_image(image):
    L = 256
    return (L - 1) - image

def main():
    image = cv2.imread('Mammogram.png', cv2.IMREAD_GRAYSCALE)
    negative = negative_image(image)
    log_transformed = log_transformation(negative)

    cv2.imshow('Original', image)
    cv2.imshow('Negative', negative)
    cv2.imshow('Log Transformed', log_transformed)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()