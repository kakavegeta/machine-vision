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


