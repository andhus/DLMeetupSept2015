from __future__ import division, print_function

import os
import numpy as np

from warnings import warn

from blocks.extensions import SimpleExtension
from blocks.serialization import secure_pickle_dump


class PickleBestMainBrick(SimpleExtension):
    """Saves best and latest models into disk based on current cost.

    Parameters
    ----------
    cost : theano.Variable
    main_brick : blocks.Model
    save_path: str
        Target director for pickling main brick.
    """

    def __init__(
        self,
        cost,
        main_brick,
        save_path,
        file_name='best_main_brick.pkl',
    ):
        super(PickleBestMainBrick, self).__init__(
            every_n_epochs=1  # TODO this could be passed via kwargs
                              # but some extra logic is needed if we don't
                              # check cost every epoch.
        )
        self.cost = cost
        self.main_brick = main_brick
        self.save_file_path = os.path.join(
            save_path,
            file_name
        )
        self.best_cost = np.inf

    def do(self, callback_name, *args):
        """Pickles the main_brick object.
        """
        cur_cost = self.main_loop.log.current_row[self.cost.name]
        if cur_cost <= self.best_cost:
            self.best_cost = cur_cost
            try:
                secure_pickle_dump(
                    self.main_brick,
                    self.save_file_path
                )
            except:
                warn('Failed to save "best main brick", continuing training.')