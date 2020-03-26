from keras.engine import InputSpec, Layer
from keras.layers import Dense
import keras.backend as K


class DenseTied(Dense):
    def __init__(self, tied_layer, **kwargs):
        self.tied_layer = tied_layer
        super().__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]

        if self.use_bias:
            self.bias = self.add_weight(shape=(self.tied_layer.units,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True

    def call(self, inputs):
        self.W = K.transpose(self.tied_layer.kernel)
        output = K.dot(inputs, self.W)
        if self.use_bias:
            print('out', output)
            output = K.bias_add(output, self.bias)
        if self.activation is not None:
            output = self.activation(output)
        return output
