import numpy as np
import cv2



class CV249:
    def cvt_to_gray(self, img):
        # Note that cv2.imread will read the image to BGR space rather than RGB space

        # TODO: your implementation
        return np.rint(img.dot(np.array([0.114, 0.587, 0.299])))

    def blur(self, img, kernel_size=(3, 3)):
        """smooth the image with box filter
        
        Arguments:
            img {np.array} -- input array
        
        Keyword Arguments:
            kernel_size {tuple} -- kernel size (default: {(3, 3)})
        
        Returns:
            np.array -- blurred image
        """
        # TODO: your implementation
        kernel = np.ones(kernel_size, np.float32)/9
        return cv2.filter2D(img, -1, kernel)
        

    def sharpen_laplacian(self, img):
        """sharpen the image with laplacian filter
        
        Arguments:
            img {np.array} -- input image
        
        Returns:
            np.array -- sharpened image
        """

        # subtract the laplacian from the original image 
        # when have a negative center in the laplacian kernel

        # TODO: your implementation
        lap_kernel = np.array([[0,1,0],[1,-4,1],[0,1,0]])
        return img - cv2.filter2D(img, -1, lap_kernel)        

    def unsharp_masking(self, img):
        """sharpen the image via unsharp masking
        
        Arguments:
            img {np.array} -- input image
        
        Returns:
            np.array -- sharpened image
        """
        # use don't use cv2 in this function
        
        # TODO: your implementation
        smoothed = self.blur(img, (3,3))
        masked = img - smoothed
        return img + masked

    def edge_det_sobel(self, img):
        """detect edges with sobel filter
        
        Arguments:
            img {np.array} -- input image
        
        Returns:
            [np.array] -- edges
        """

        # TODO: your implementation
        kx = np.flip(np.array([[-1,0,1],[-2,0,2],[-1,0,1]]))
        ky = np.flip(np.array([[-1,-2,-1],[0,0,0],[1,2,1]]))
        g_x = cv2.filter2D(img, -1, kx)
        g_y = cv2.filter2D(img, -1, ky)
        g = np.sqrt(g_x**2 + g_y**2)
        return g.astype(np.uint8)

if __name__ == "__main__":
    cv249 = CV249()
    img = cv2.imread('../../data/lena.tiff')
    my_gray = cv249.cvt_to_gray(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(gray.shape)
    print(np.mean(np.abs(gray - my_gray)))
    print(img[0, 0, 0], img[0, 0, 1], img[0, 0, 2])
    #cv2.imshow('image', my_gray)
    if(np.mean(np.abs(gray - my_gray)) < 2e-3):
        print("pass")
    blur = cv2.blur(img, ksize=(3,3))
    my_blur = cv249.blur(img,(3,3))
    if(np.all(my_blur==blur)):
        print("pass")
    
    lap_sharp = img - cv2.Laplacian(img, -1, ksize=1)
    my_lap_sharp = cv249.sharpen_laplacian(img)
    if(np.all(my_lap_sharp == lap_sharp)):
        print("pass lap sharp")
    
    masked_img = img - cv2.blur(img, ksize=(3, 3))
    unsharp_mask_img = img + masked_img
    my_unsharp_mask_img = cv249.unsharp_masking(img)
    if(np.all(my_unsharp_mask_img==unsharp_mask_img)):
        print("pass unsharped mask")

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    g_x = cv2.Sobel(gray_img, -1, 1, 0, ksize=3)
    g_y = cv2.Sobel(gray_img, -1, 0, 1, ksize=3)
    g = np.sqrt(g_x ** 2 + g_y ** 2).astype(np.uint8)
    my_g = cv249.edge_det_sobel(img)
    print(g.shape, my_g.shape)
    if(np.all(g==my_g)):
        print("pass sobel")

