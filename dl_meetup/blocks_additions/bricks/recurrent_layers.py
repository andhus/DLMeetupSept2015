from __future__ import division, print_function

from blocks.bricks.base import(
    application,
    lazy
)
from blocks.bricks import(
    Initializable,
    Linear,
    Feedforward,
    Tanh,
)
from blocks.bricks.recurrent import(
    LSTM,
    SimpleRecurrent,
)
from blocks import initialization as init


class SimpleRecurrentLayer(Initializable, Feedforward):
    """ Blocks implementation of SimpleRecurrent is general and only
    handles the "recursive part". This class wraps the SimpleRecurrent
    class and adds linear input transformation. It can be used for most basic
    cases as a layer in a sequence of layers.

    Parameters
    ----------
    input_dim : int
    state_dim : int
    activation : Brick
    state_weights_init : NdarrayInitialization
        Initialization of weights in LSTM (including gates).
    input_weights_init : NdarrayInitialization
        Initialization of weights in linear transformation of input.
    biases_init : NdarrayInitialization
        Initialization of biases in linear transformation of input.
    """

    @lazy()
    def __init__(
        self,
        input_dim,
        state_dim,
        activation=Tanh(),
        state_weights_init=None,
        input_weights_init=None,
        biases_init=None,
        **kwargs
    ):
        super(SimpleRecurrentLayer, self).__init__(
            biases_init=biases_init,
            **kwargs
        )
        if state_weights_init is None:
            state_weights_init = init.IsotropicGaussian(0.01)
        if input_weights_init is None:
            input_weights_init = init.IsotropicGaussian(0.01)
        if biases_init is None:
            biases_init = init.Constant(0)

        self.input_transformation = Linear(
            input_dim=input_dim,
            output_dim=state_dim,
            weights_init=input_weights_init,
            biases_init=biases_init
        )
        self.rnn = SimpleRecurrent(
            dim=state_dim,
            activation=activation,
            weights_init=state_weights_init
        )
        self.children = [self.input_transformation, self.rnn]

    @application
    def apply(
        self,
        inputs,
        *args,
        **kwargs
    ):
        """ Transforms input, sends to BasicRecurrent and returns output.

        Parameters
        ----------
        inputs : tensor.TensorVariable
            The 3 dimensional tensor of inputs in the shape (timesteps,
            batch_size, features).

        Returns
        -------
        outputs : tensor.TensorVariable
            The 3 dimensional tensor of outputs in the shape (timesteps,
            batch_size, features).
        """
        rnn_inputs = self.input_transformation.apply(inputs)
        outputs = self.rnn.apply(inputs=rnn_inputs, *args, **kwargs)

        return outputs

    @apply.delegate
    def apply_delegate(self):
        return self.children[0].apply

    @property
    def input_dim(self):
        return self.input_transformation.input_dim

    @input_dim.setter
    def input_dim(self, value):
        self.input_transformation.input_dim = value

    @property
    def output_dim(self):
        return self.rnn.dim

    @output_dim.setter
    def output_dim(self, value):
        self.rnn.dim = value


class LSTMLayer(Initializable, Feedforward):
    """ Blocks implementation of LSTM is general and only handles the "recursive
    part". This class wraps the LSTM class and adds the input transformation as
    well as does not return cell activations. It can be used for most basic
    cases as a layer in a sequence of layers.

    Parameters
    ----------
    input_dim : int
    state_dim : int
    activation : Brick
    state_weights_init : NdarrayInitialization
        Initialization of weights in LSTM (including gates).
    input_weights_init : NdarrayInitialization
        Initialization of weights in linear transformation of input.
    biases_init : NdarrayInitialization
        Initialization of biases in linear transformation of input.
    """

    @lazy()
    def __init__(
        self,
        input_dim,
        state_dim,
        activation=Tanh(),
        state_weights_init=None,
        input_weights_init=None,
        biases_init=init.Constant(0),
        **kwargs
    ):
        super(LSTMLayer, self).__init__(
            biases_init=biases_init,
            **kwargs
        )
        if state_weights_init is None:
            state_weights_init = init.IsotropicGaussian(0.01)
        if input_weights_init is None:
            input_weights_init = init.IsotropicGaussian(0.01)
        if biases_init is None:
            biases_init = init.Constant(0)

        self.input_transformation = Linear(
            input_dim=input_dim,
            output_dim=state_dim * 4,
            weights_init=input_weights_init,
            biases_init=biases_init
        )
        self.lstm = LSTM(
            dim=state_dim,
            activation=activation,
            weights_init=state_weights_init
        )
        self.children = [self.input_transformation, self.lstm]

    @application
    def apply(
        self,
        inputs,
        *args,
        **kwargs
    ):
        """ Transforms input, sends to LSTM and returns output excluding cell
        activations.

        Parameters
        ----------
        inputs : tensor.TensorVariable
            The 3 dimensional tensor of inputs in the shape (timesteps,
            batch_size, features).

        Returns
        -------
        outputs : tensor.TensorVariable
            The 3 dimensional tensor of outputs in the shape (timesteps,
            batch_size, features).
        """
        lstm_inputs = self.input_transformation.apply(inputs)
        outputs, _ = self.lstm.apply(inputs=lstm_inputs, *args, **kwargs)
        # by default LSTM.apply also output cell activations

        return outputs

    @apply.delegate
    def apply_delegate(self):
        return self.children[0].apply

    @property
    def input_dim(self):
        return self.input_transformation.input_dim

    @input_dim.setter
    def input_dim(self, value):
        self.input_transformation.input_dim = value

    @property
    def output_dim(self):
        return self.lstm.dim

    @output_dim.setter
    def output_dim(self, value):
        self.lstm.dim = value