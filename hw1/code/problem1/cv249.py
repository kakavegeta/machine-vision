import numpy as np
import cv2



class CV249:
    def cvt_to_gray(self, img):
        # Note that cv2.imread will read the image to BGR space rather than RGB space

        # TODO: your implementation
        return img.dot(np.array([0.114, 0.587, 0.299]))

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
        kernel = np.ones(kernel_size)*(1/9);
        
        pass

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
        pass

    def unsharp_masking(self, img):
        """sharpen the image via unsharp masking
        
        Arguments:
            img {np.array} -- input image
        
        Returns:
            np.array -- sharpened image
        """
        # use don't use cv2 in this function
        
        # TODO: your implementation
        pass

    def edge_det_sobel(self, img):
        """detect edges with sobel filter
        
        Arguments:
            img {np.array} -- input image
        
        Returns:
            [np.array] -- edges
        """

        # TODO: your implementation
        pass

if __name__ == "__main__":
    cv249 = CV249()
    img = cv2.imread('../../data/lena.tiff')
    my_gray = cv249.cvt_to_gray(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(gray.shape)
    print(np.mean(np.abs(gray - my_gray)))
    cv2.imshow('image',gray)
    #cv2.imshow('image', my_gray)
    if(np.mean(np.abs(gray - my_gray)) < 2e-3):
        print("pass")
