import cv2 as cv
import numpy as np

def hough_circle_foo(edges, r_min, r_max, threshold, region):
    M, N = edges.shape
    x_min, x_max, y_min, y_max = region
    # extract edges coordinate
    edges = np.argwhere(edges[:,:])
    edges = [(x, y) for x, y in edges if x>x_min and x<x_max and y<y_max and y>y_min]
    # accumulator matrix A[a, b, r]
    A = np.zeros((M+2*r_max, N+2*r_max, r_max))

    # pre calculate theta
    theta = np.arange(0,360)*np.pi/180

    #voting
    for r in range(r_min, r_max, 1):
        for t in theta:
            for x,y in edges:
                a = x - int(np.round(r * np.cos(t)))
                b = y - int(np.round(r * np.sin(t)))
                A[a, b, r] += 1
        print(r, np.max(A), np.where(A==np.max(A)))
    circles = np.argwhere(A > np.max(A)*threshold)
    return circles    



          

if __name__ == "__main__":
    cimg = cv.imread("shoeprint/circle.png")
    gray = cv.cvtColor(cimg, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (5,5),0)
    edges = cv.Canny(gray, 60, 110) 
    print(edges.shape)
    cv.imwrite("output/edges.png", edges)
    np.savetxt('output/edges.txt', edges)
     
    circles = hough_circle_foo(edges, 25, 51, 0.7, (120,230,160,270)) 
    np.save("output/circles", circles)

    for a, b, r in circles:
        cv.circle(cimg, (b, a), r, (0,255,0), 1, lineType=4)
        cv.circle(cimg, (b, a), 1, (0, 0, 255), 3)
    
    cv.imwrite("output/hough_circle.png", cimg)
    
    