import cv2 as cv

if __name__ == "__main__":
    img = cv.imread("shoeprint/circle.png")
    cv.circle(img, (210, 170), 1, (0,0,255), 3)
    cv.imwrite("output/draw_circle_test.png", img)