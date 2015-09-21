from __future__ import division, print_function

import os
import shutil


def safe_mkdir(path):
    """ Checks if directory already exists, and requires user input before
    overwriting.

    Parameters
    ----------
    :param path: str
    """
    if os.path.isdir(path):  # TODO what if file?
        answer = raw_input(
            "Directory {} already exists, if you continue all data will be "
            "overwritten. Do you wish to continue? [y/n]".format(path)
        )
        allowed = answer in ['Y', 'y', 'yes']
        if allowed:
            shutil.rmtree(path)
            os.mkdir(path)
        else:
            raise Exception('Overwrite not permitted by user.')
    else:
        os.mkdir(path)