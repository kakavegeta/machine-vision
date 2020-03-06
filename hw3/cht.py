import cv2 as cv
import numpy as np

if __name__ == "__main__":
    img = cv.imread("shoeprint/circle.png")
    #img = cv.GaussianBlur(img, (5,5), 0)
    img = cv.medianBlur(img, 5)
    #cimg = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    edges = cv.Canny(img, 120, 180)
    cv.imwrite("output/edge.png", edges)
    cv.imwrite("output/smoothed.png", img)

    cimg = img
    circles = cv.HoughCircles(edges, cv.HOUGH_GRADIENT, 1, 20,
                             param1=50, param2=30, minRadius=0, maxRadius=0)
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        cv.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0),2)
        cv.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)

    cv.imwrite("output/cht.png", cimg)