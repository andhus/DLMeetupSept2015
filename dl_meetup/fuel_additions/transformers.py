from __future__ import division, print_function

import numpy as np
import theano

from fuel.transformers import Transformer, SingleMapping
from dl_meetup.utils.convert import tsf_of_stf


class TSFFromSTF(SingleMapping):
    """Applies a shape transform mapping to the source.
    source Shape=(
        <n_samples>,
        <n_timesteps>,
        <n_features>
    )
    Stream(output) shape=(
        <n_timesteps>,
        <n_samples_in_batch>,
        <n_features>
    )

    Parameters
    ----------
    data_stream : instance of :class:`DataStream`
        The wrapped data stream.
    which_sources : tuple of str, optional
        Which sources to apply the mapping to. Defaults to `None`, in
        which case the mapping is applied to all sources.

    """
    def __init__(self, data_stream, **kwargs):
        super(TSFFromSTF, self).__init__(data_stream, **kwargs)
        self.batch_input = True

    def mapping(self, source):
        return tsf_of_stf(source)


class ForceTheanoFloatX(Transformer):
    """Force all floating point numpy arrays to be theano.config.floatX.

    The fuel native ForceFloatX transformer reads fuel.config.floatX, by using
    this transformer instead one only needs to have the .theanorc set up
    correctly.
    """
    def __init__(self, data_stream):
        super(ForceTheanoFloatX, self).__init__(
            data_stream, axis_labels=data_stream.axis_labels)

    def get_data(self, request=None):
        if request is not None:
            raise ValueError
        data = next(self.child_epoch_iterator)
        result = []
        for piece in data:
            if (isinstance(piece, np.ndarray) and
                    piece.dtype.kind == "f" and
                    piece.dtype != theano.config.floatX):
                result.append(piece.astype(theano.config.floatX))
            else:
                result.append(piece)
        return tuple(result)
