import numpy as np


class Conv2D(object):
    """2D convolutional layer.

    Arguments:
        kernel_size (tuple): the shape of the kernel. It is a tuple = (
            out_channels, in_channels, kernel_height, kernel_width).
        strides (int or tuple): the strides of the convolution operation.
            padding (int or tuple): number of zero paddings.
        weights_init (obj):  an object instantiated using any initializer class
                in the "initializer" module.
        bias_init (obj):  an object instantiated using any initializer class
                in the "initializer" module.
        name (str): the name of the layer.

    Attributes:
        W (np.array): the weights of the layer. A 4D array of shape (
            out_channels, in_channels, kernel_height, kernel_width).
        b (np.array): the biases of the layer. A 1D array of shape (
            out_channels).
        kernel_size (tuple): the shape of the filter. It is a tuple = (
            out_channels, in_channels, kernel_height, kernel_width).
        strides (tuple): the strides of the convolution operation. A tuple = (
            height_stride, width_stride).
        padding (tuple): the number of zero paddings along the height and
            width. A tuple = (height_padding, width_padding).
        name (str): the name of the layer.

    """

    def __init__(
            self, kernel_size, stride, padding, name = "Cov2D"):
        self.W = np.random.randn(*kernel_size)
        self.b = np.random.randn(kernel_size[0], 1)
        self.kernel_size = kernel_size
        self.stride = (stride, stride) if type(stride) == int else stride
        self.padding = (padding, padding) if type(padding) == int else padding
        self.name = name


    def __repr__(self):
        return "{}({}, {}, {})".format(
            self.name, self.kernel_size, self.stride, self.padding
        )

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """Compute the layer output.

        Args:
            x (np.array): the input of the layer. A 3D array of shape (
                in_channels, in_heights, in_weights).

        Returns:
            The output of the layer. A 3D array of shape (out_channels,
                out_heights, out_weights).

        """
        p, s = self.padding, self.stride
        x_padded = np.pad(
            x, ((0, 0), (p[0], p[0]), (p[1], p[1])), mode='constant'
        )

        # check dimensions
        assert (x.shape[1] - self.W.shape[2] + 2 * p[0]) / s[0] + 1 > 0, \
                'Height doesn\'t work'
        assert (x.shape[2] - self.W.shape[3] + 2 * p[1]) / s[1] + 1 > 0, \
                'Width doesn\'t work'

        # TODO: Put your code below
        H, W, F_H, F_W = x.shape[1], x.shape[2], self.W.shape[2], self.W.shape[3]
        CC = self.W.shape[0] # output channels

        HH = (H - F_H + 2*p[0]) // s[0] + 1 # output height
        WW = (W - F_W + 2*p[1]) // s[1] + 1 # output width
        
        OO = np.zeros((CC, HH, WW))
        # convolve
        for hh in range(HH):
            for ww in range(WW):
                for cc in range(CC):
                    _x = x_padded[:, hh*s[0]:hh*s[0]+F_H, ww*s[1]:ww*s[1]+F_W]
                    OO[cc, hh, ww] = np.sum(_x * self.W[cc, :, :, :]) + self.b[cc]
        return OO

class MaxPool2D:
    def __init__(self, kernel_size, stride, padding, name="MaxPool2D"):
        self.kernel_size = kernel_size # len(kernel_size == 2)
        self.stride = (stride, stride) if type(stride) == int else stride
        self.padding = (padding, padding) if type(padding) == int else padding
        self.name = name

    def __repr__(self):
        return "{}({}, {}, {})".format(
        self.name, self.kernel_size, self.stride, self.padding
    )

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """Compute the layer output.

        Arguments:
            x {[np.array]} -- the input of the layer. A 3D array of shape (
                              in_channels, in_heights, in_weights).
        Returns:
            The output of the layer. A 3D array of shape (out_channels,
                out_heights, out_weights).
        """
        p, s = self.padding, self.stride
        x_padded = np.pad(
            x, ((0, 0), (p[0], p[0]), (p[1], p[1])), mode='constant'
        )

        # check dimensions
        assert (x.shape[1] - self.kernel_size[0] + 2 * p[0]) / s[0] + 1 > 0, \
                'Height doesn\'t work'
        assert (x.shape[2] - self.kernel_size[1] + 2 * p[1]) / s[1] + 1 > 0, \
                'Width doesn\'t work'

        # TODO: Put your code below
        H, W, F_H, F_W = x.shape[1], x.shape[2], self.kernel_size[0], self.kernel_size[1]
        CC = x_padded.shape[0]
        WW = (W - F_W + 2*p[1]) // s[1] + 1 # output width
        HH = (H - F_H + 2*p[0]) // s[0] + 1 # output height
        OO = np.zeros((CC, HH, WW))
        for hh in range(HH):
            for ww in range(WW):
                for cc in range(CC):
                    OO[cc, hh, ww] = np.max(x_padded[cc, hh*s[0]:hh*s[0]+F_H, ww*s[1]:ww*s[1]+F_W])
        return OO



class AvgPool2D:
    def __init__(self, kernel_size, stride, padding, name="MaxPool2D"):
        self.kernel_size = kernel_size
        self.stride = (stride, stride) if type(stride) == int else stride
        self.padding = (padding, padding) if type(padding) == int else padding
        self.name = name

    def __repr__(self):
        return "{}({}, {}, {})".format(
        self.name, self.kernel_size, self.stride, self.padding
    )

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """Compute the layer output.

        Arguments:
            x {[np.array]} -- the input of the layer. A 3D array of shape (
                              in_channels, in_heights, in_weights).
        Returns:
            The output of the layer. A 3D array of shape (out_channels,
                out_heights, out_weights).
        """
        p, s = self.padding, self.stride
        x_padded = np.pad(
            x, ((0, 0), (p[0], p[0]), (p[1], p[1])), mode='constant'
        )

        # check dimensions
        assert (x.shape[1] - self.kernel_size[0] + 2 * p[0]) / s[0] + 1 > 0, \
                'Height doesn\'t work'
        assert (x.shape[2] - self.kernel_size[1] + 2 * p[1]) / s[1] + 1 > 0, \
                'Width doesn\'t work'

        # TODO: Put your code below
                # TODO: Put your code below
        H, W, F_H, F_W = x.shape[1], x.shape[2], self.kernel_size[0], self.kernel_size[1]
        CC = x_padded.shape[0]
        WW = (W - F_W + 2*p[1]) // s[1] + 1 # output width
        HH = (H - F_H + 2*p[0]) // s[0] + 1 # output height
        OO = np.zeros((CC, HH, WW))
        for hh in range(HH):
            for ww in range(WW):
                for cc in range(CC):
                    OO[cc, hh, ww] = np.mean(x_padded[cc, hh*s[0]:hh*s[0]+F_H, ww*s[1]:ww*s[1]+F_W])
        return OO
