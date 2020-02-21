"""All the layer functions go here.

"""

from __future__ import print_function, absolute_import
import numpy as np


class FullyConnected(object):
    """Fully connected layer 'y = Wx + b'.

    Arguments:
        shape (tuple): the shape of the fully connected layer. shape[0] is the
            output size and shape[1] is the input size.
        weights_init (obj):  an object instantiated using any initializer class
                in the "initializer" module.
        bias_init (obj):  an object instantiated using any initializer class
                in the "initializer" module.
        name (str): the name of the layer.

    Attributes:
        W (np.array): the weights of the fully connected layer.
        b (np.array): the biases of the fully connected layer.
        shape (tuple): the shape of the fully connected layer. shape[0] is the
            output size and shape[1] is the input size.
        name (str): the name of the layer.

    """

    def __init__(
        self, d_in, d_out, weights_init=None, bias_init=None, name="FullyConnected"
    ):
        shape = (d_out, d_in)
        self.W = weights_init.initialize(shape) \
            if weights_init else np.random.randn(*shape).astype(np.float32)
        self.b = bias_init.initialize((shape[0])) \
            if bias_init else np.random.randn(shape[0]).astype(np.float32)
        self.shape = shape
        self.name = name

    def __repr__(self):
        return "{}({}, {})".format(self.name, self.shape[0], self.shape[1])

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """Compute the layer output.

        Args:
            x (np.array): the input of the layer.

        Returns:
            The output of the layer.

        """
        Y = np.dot(self.W, x) + self.b
        return Y

    def backward(self, x, dv_y):
        """Compute the gradients of weights and biases and the gradient with
        respect to the input.

        Args:
            x (np.array): the input of the layer.
            dv_y (np.array): The derivative of the loss with respect to the
                output.

        Returns:
            dv_x (np.array): The derivative of the loss with respect to the
                input.
            dv_W (np.array): The derivative of the loss with respect to the
                weights.
            dv_b (np.array): The derivative of the loss with respect to the
                biases.

        """

        # TODO: write your implementation below
        dv_x = np.empty(x.shape, dtype=np.float32)
        dv_W = np.empty(self.W.shape, dtype=np.float32)
        dv_b = np.empty(self.b.shape, dtype=np.float32)

        dv_x = np.dot(dv_y, self.W).T
        dv_W = np.dot(dv_y.reshape(dv_y.shape[0], 1), x.reshape(1, x.shape[0]))
        dv_b = dv_y

        # don't change the order of return values
        return dv_x, dv_W, dv_b


class Conv2D(object):
    """2D convolutional layer.

    Arguments:
        filter_size (tuple): the shape of the filter. It is a tuple = (
            out_channels, in_channels, filter_height, filter_width).
        strides (int or tuple): the strides of the convolution operation.
            padding (int or tuple): number of zero paddings.
        weights_init (obj):  an object instantiated using any initializer class
                in the "initializer" module.
        bias_init (obj):  an object instantiated using any initializer class
                in the "initializer" module.
        name (str): the name of the layer.

    Attributes:
        W (np.array): the weights of the layer. A 4D array of shape (
            out_channels, in_channels, filter_height, filter_width).
        b (np.array): the biases of the layer. A 1D array of shape (
            in_channels).
        filter_size (tuple): the shape of the filter. It is a tuple = (
            out_channels, in_channels, filter_height, filter_width).
        strides (tuple): the strides of the convolution operation. A tuple = (
            height_stride, width_stride).
        padding (tuple): the number of zero paddings along the height and
            width. A tuple = (height_padding, width_padding).
        name (str): the name of the layer.

    """

    def __init__(
            self, in_channel, out_channel, kernel_size, stride, padding,
            weights_init=None, bias_init=None, name="Conv2D"):
        filter_size = (out_channel, in_channel, *kernel_size)

        self.W = weights_init.initialize(filter_size) \
            if weights_init else np.random.randn(*filter_size).astype(np.float32)
        self.b = bias_init.initialize((filter_size[0], 1)) \
            if bias_init else np.random.randn(out_channel, 1).astype(np.float32)

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

        y_shape = (
            self.W.shape[0],
            int((x.shape[1] - self.W.shape[2] + 2 * p[0]) / s[0]) + 1,
            int((x.shape[2] - self.W.shape[3] + 2 * p[1]) / s[1]) + 1,
        )
        y = np.empty(y_shape, dtype=np.float32)

        for k in range(y.shape[0]):
            for i in range(y.shape[1]):
                for j in range(y.shape[2]):
                    y[k, i, j] = np.sum(
                        x_padded[
                            :,
                            i * s[0] : i * s[0] + self.W.shape[2],
                            j * s[1] : j * s[1] + self.W.shape[3]
                        ] * self.W[k]
                    ) + self.b[k]
        return y

    def backward(self, x, dv_y):
        """Compute the gradients of weights and biases and the gradient with
        respect to the input.

        Args:
            x (np.array): the input of the layer. A 3D array of shape (
                in_channels, in_heights, in_weights).
            dv_y (np.array): The derivative of the loss with respect to the
                output. A 3D array of shape (out_channels, out_heights,
                out_weights).

        Returns:
            dv_x (np.array): The derivative of the loss with respect to the
                input. It has the same shape as x.
            dv_W (np.array): The derivative of the loss with respect to the
                weights. It has the same shape as self.W
            dv_b (np.array): The derivative of the loss with respect to the
                biases. It has the same shape as self.b

        """
        p, s = self.padding, self.stride
        x_padded = np.pad(
            x, ((0, 0), (p[0], p[0]), (p[1], p[1])), mode='constant'
        )

        # TODO: write your implementation below

        dv_W = np.zeros(self.W.shape, dtype=np.float64)
        dv_b = np.zeros(self.b.shape, dtype=np.float64)
        dv_x = np.zeros(x.shape, dtype=np.float64)

        dv_x_padded = np.pad(
            dv_x, ((0, 0), (p[0], p[0]), (p[1], p[1])), mode = 'constant'
        )


        fh, fw = self.W.shape[2], self.W.shape[3]
        HH, WW = dv_y.shape[1], dv_y.shape[2] # output height, weight
        H, W = x.shape[1], x.shape[2] # input height, weight

        for o in range(self.W.shape[0]):
                for h in range(HH):
                    for w in range(WW):
                        dv_W[o, :, :, :] += dv_y[o, h, w] * x_padded[:, h*s[0]:h*s[0]+fh, w*s[1]:w*s[1]+fw]
                        dv_x_padded[:, h*s[0]:h*s[0]+fh, w*s[1]:w*s[1]+fw] += dv_y[o, h, w] * self.W[o, :, :, :]
                        dv_b[o] += dv_y[o, h, w]
        # don't change the order of return values
        dv_x = dv_x_padded[:, p[0]:H-p[0], p[1]:W-p[1]]
        return dv_x, dv_W, dv_b


class MaxPool2D:
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

        y_shape = (
            x.shape[0],
            int((x.shape[1] - self.kernel_size[0] + 2 * p[0]) / s[0]) + 1,
            int((x.shape[2] - self.kernel_size[1] + 2 * p[1]) / s[1]) + 1,
        )
        y = np.empty(y_shape, dtype=np.float32)

        for i in range(y.shape[1]):
            for j in range(y.shape[2]):
                y[:, i, j] = np.max(x_padded[
                                    :,
                                    i * s[0]: i * s[0] + self.kernel_size[0],
                                    j * s[1]: j * s[1] + self.kernel_size[1]
                                    ].reshape(-1, self.kernel_size[0] * self.kernel_size[1]),
                                    axis=1
                                    )

        return y

    def backward(self, x, dv_y):
        """Compute the gradients of weights and biases and the gradient with
                respect to the input.

                Args:
                    x (np.array): the input of the layer. A 3D array of shape (
                        in_channels, in_heights, in_weights).
                    dv_y (np.array): The derivative of the loss with respect to the
                        output. A 3D array of shape (out_channels, out_heights,
                        out_weights).

                Returns:
                    dv_x (np.array): The derivative of the loss with respect to the
                        input. It has the same shape as x.
                """
        p, s = self.padding, self.stride
        x_padded = np.pad(
            x, ((0, 0), (p[0], p[0]), (p[1], p[1])), mode='constant'
        )


        fh, fw = self.kernel_size[0], self.kernel_size[1]
        HH, WW = dv_y.shape[1], dv_y.shape[2]
        H, W = x.shape[1], x.shape[2]

        # TODO: write your implementation below
        dv_x = np.zeros(x.shape, dtype=np.float64)

        dv_x_padded = np.pad(
            dv_x, ((0, 0), (p[0], p[0]), (p[1], p[1])), mode='constant'
        )

        mask = lambda x: x==np.max(x)

        for h in range(HH):
            for w in range(WW):
                for o in range(dv_y.shape[0]): # dv_y.shape==dv_x.shape
<<<<<<< HEAD
                    #masked = mask(x_padded[:, h*s[0]:h*s[0]+fh, w*s[1]:w*s[1]+fw])
                    #dv_x_padded[:, h*s[0]:h*s[0]+fh, w*s[1]:w*s[1]+fw] += np.multiply(masked, dv_y[o,h,w])
<<<<<<< HEAD
                    masked = mask(x[:, h:h+fh, w:w+fw])
                    dv_x[:, h:h+fh, w:w+fw] += np.multiply(masked, dv_y[o,h,w])
=======
                    masked = mask(x[o, h:h+fh, w:w+fw])
                    dv_x[o, h:h+fh, w:w+fw] += np.multiply(masked, dv_y[o,h,w])
>>>>>>> 8c776efb2eab71f6a39113a86d66705266395342

        #dv_x = dv_x_padded[:, p[0]:H-p[0], p[1]:W-p[1]]

=======
                    masked = mask(x[o, h:h+fh, w:w+fw])
                    dv_x[o, h:h+fh, w:w+fw] += np.multiply(masked, dv_y[o,h,w])
>>>>>>> 20bbfddb393a45609d86c45d6bf37ce9ba2bad04
        return dv_x

