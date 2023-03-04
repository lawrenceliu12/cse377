import cv2 #pip install opencv-python
import numpy #pip install numpy

def piecewise_transformation(negative_image):
    L = 256
    rmin = numpy.min(negative_image)
    rmax = numpy.max(negative_image)
    smin = 0
    smax = L - 1
    slope = (smax - smin) / (rmax - rmin)
    intercept = smin - slope * rmin
    return numpy.clip((slope * negative_image + intercept), 0, 255)

def power_law_transformation(negative_image):
    gamma = 1 #value of gamma
    power = numpy.power(negative_image, gamma) #power rule of the matrix
    power = cv2.normalize(power, None, 0, 255, cv2.NORM_MINMAX) #normalize picture

    return numpy.uint8(power) #return the matrix in 8 bits.

def log_transformation(negative_image):
    log_transform = cv2.normalize(numpy.array(numpy.log(1 + negative_image), dtype=numpy.float32), None, 0, 255, cv2.NORM_MINMAX) #use the log transformation formula to a new matrix
    return numpy.uint8(log_transform) #return the matrix in 8 bits.

def negative_image(image):
    L = 256 #L
    return (L - 1) - image #L - 1 - image

def main():
    image = cv2.imread('Mammogram.png')
    negative = negative_image(image)
    log_transformed = log_transformation(negative)
    power_law_transformed = power_law_transformation(negative)
    piecewise_transformed = piecewise_transformation(negative)

    cv2.imwrite('Negative.png', negative)
    cv2.imwrite('Log-Transformed.png', log_transformed)
    cv2.imwrite('Power-Law-Transformed.png', power_law_transformed)
    cv2.imwrite('Piecewise-Transformed.png', piecewise_transformed)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()