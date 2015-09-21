from __future__ import division, print_function

from theano import tensor

from blocks.bricks import Initializable, MLP, Linear, Tanh
from blocks import initialization as init

from dl_meetup.blocks_additions.bricks.recurrent_layers import LSTMLayer


class SeqToSeqLSTM(Initializable):

    def __init__(
        self,
        input_dim,
        h0_dim,
        s0_dim,
        h1_dim,
        output_dim,
    ):
        super(SeqToSeqLSTM, self).__init__()
        self.h0__input = MLP(
            [Tanh()],
            dims=[
                input_dim,
                h0_dim
            ],
            weights_init=init.IsotropicGaussian(0.01),
            biases_init=init.IsotropicGaussian(0.3),
            name='MLP:h0__input'
        )
        self.s0__h0_input = LSTMLayer(
            input_dim=h0_dim + input_dim,
            state_dim=s0_dim,
            name='LSTMLayer:s0__h0_input'
        )

        self.h1__s0_h0_input = MLP(
            [Tanh()],
            dims=[
                s0_dim + h0_dim + input_dim,
                h1_dim
            ],
            weights_init=init.IsotropicGaussian(0.01),
            biases_init=init.Constant(0.0),
            name='MLP:h1__s0_h0_input'
        )
        self.output__h1_s0_h0_input = Linear(
            input_dim=h1_dim + s0_dim + h0_dim + input_dim,
            output_dim=output_dim,
            weights_init=init.IsotropicGaussian(0.01),
            biases_init=init.Constant(0.0),
            name='Linear:output__h1_s0_h0_input'
        )
        self.children = [
            self.h0__input,
            self.s0__h0_input,
            self.h1__s0_h0_input,
            self.output__h1_s0_h0_input
        ]

    def apply(self, input_):
        h0 = self.h0__input.apply(input_)
        s0 = self.s0__h0_input.apply(
            tensor.concatenate([h0, input_], axis=2)
        )
        h1 = self.h1__s0_h0_input.apply(
            tensor.concatenate([s0, h0, input_], axis=2)
        )
        output = self.output__h1_s0_h0_input.apply(
            tensor.concatenate([h1, s0, h0, input_], axis=2)
        )

        return output
