from __future__ import division, print_function

import numpy as np
import h5py

from fuel.datasets import H5PYDataset
from theano import config


def write_h5py_dataset(
    nested_dataset_dict,
    sources_dim_labels,
    file_path,
    dtype=config.floatX
):
    """ Creates a h5py file based dataset (writes this to disk).

    Parameters
    ----------
    nested_dataset_dict : {
        <subset>: {
            <source>: numpy.ndarray()
        }
    }, where
        <subset> : str
        <source> : str
    file_path : str
    sources_dim_labels = {
        <source> : [str]  # length of list = to number of dimensions of source
    }
    name : str

    Returns
    -------
    h5py_file_path
    """
    previous_end_idx = 0
    source_to_list_of_subsets = {}
    split_dict = {}
    for subset_name in nested_dataset_dict.keys():
        if subset_name not in split_dict:
                split_dict[subset_name] = {}
        n_samples = nested_dataset_dict[subset_name].values()[0].shape[0]
        for source_name in nested_dataset_dict[subset_name].keys():
            assert nested_dataset_dict[
                subset_name
            ][source_name].shape[0] == n_samples
            split_dict[subset_name][source_name] = (
                previous_end_idx,
                previous_end_idx + n_samples
            )
            if source_name not in source_to_list_of_subsets:
                source_to_list_of_subsets[source_name] = [
                    nested_dataset_dict[subset_name][source_name]
                ]
            else:
                source_to_list_of_subsets[source_name].append(
                    nested_dataset_dict[subset_name][source_name]
                )
        previous_end_idx += n_samples

    concatenated_subsets = {}
    for source_name in source_to_list_of_subsets.keys():
        concatenated_subsets[source_name] = np.concatenate(
            source_to_list_of_subsets[source_name],
            axis=0
        )

    def write_one_source(
        source_name,
        source_data,
        source_dim_labels,
        h5py_file
    ):
        """ Writes the content for one source to the passed H5PY File.

        Parameters
        ----------
        source_name : str
        source_data : ndarray(shape=S)
        source_dim_labels : [str]
            len(source_dim_labels) = len(S)
        h5py_file : h5py.File
        """
        source_handle = h5py_file.create_dataset(
            source_name,
            source_data.shape,
            dtype=dtype
        )
        source_handle[...] = source_data
        for dim, label in zip(source_handle.dims, source_dim_labels):
            dim.label = label

    with h5py.File(file_path, mode='w') as f:
        for source_name in concatenated_subsets.keys():
            write_one_source(
                source_name=source_name,
                source_data=concatenated_subsets[source_name],
                source_dim_labels=sources_dim_labels[source_name],
                h5py_file=f
            )
        f.attrs['split'] = H5PYDataset.create_split_array(split_dict)
        f.flush()
        f.close()


def h5py_dataset_summary(
    h5py_dataset
):
    """
    Parameters
    ----------
    :param h5py_dataset: fuel.datasets.H5PYDataset

    Returns
    -------
    :return: summary : str
    """
    summary = 'Dataset: {which_set}\n' \
        'sources: {sources}\n' \
        'number of samples: {num_examples}'.format(
            which_set=h5py_dataset.which_set,
            sources=h5py_dataset.sources,
            num_examples=h5py_dataset.num_examples
        )

    return summary