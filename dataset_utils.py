

#!/usr/bin/env python

import numpy
import os
import h5py

import thousand_genomes, asw_dataset
import aggregate_dataset as opensnp


def shuffle(data_sources, seed=23):
    """
    Shuffles multiple data sources (numpy arrays) together so the
    correspondance between data sources (such as inputs and targets) is
    maintained.
    """

    numpy.random.seed(seed)
    indices = numpy.arange(data_sources[0].shape[0])
    numpy.random.shuffle(indices)

    return [d[indices] for d in data_sources]


def split(data_sources, splits):
    """
    Splits the given data sources (numpy arrays) according to the provided
    split boundries.

    Ex : if splits is [0.6], every data source will be separated in two parts,
    the first containing 60% of the data and the other containing the
    remaining 40%.
    """

    if splits is None:
        return data_sources

    split_data_sources = []
    nb_elements = data_sources[0].shape[0]
    start = 0
    end = 0

    for s in splits:
        end += int(nb_elements * s)
        split_data_sources.append([d[start:end] for d in data_sources])
        start = end
    split_data_sources.append([d[end:] for d in data_sources])

    return split_data_sources


def prune_splits(splits, nb_prune):
    """
    Takes as input a list of split points in a dataset and produces a new list
    where the last split has been removed and the other splits are expanded
    to encompass the whole data but keeping the same proportions.

    Ex : splits = [0.6, 0.2] corresponds to 3 splits : 60%, 20% and 20%.
         with nb_prune=1, the method would return [0.75] which corresponds to
         2 splits : 75% and 25%. Hence, the proportions between the remaining
         splits remain the same.
    """
    if nb_prune > 1:
        normalization_constant = (1.0 / sum(splits[:-(nb_prune-1)]))
    else:
        normalization_constant = 1.0 / sum(splits)
    return [s * normalization_constant for s in splits[:-nb_prune]]


def load_1000_genomes(path, transpose=False, label_splits=None, feature_splits=None,
                      nolabels='raw', fold=0, norm=True,
                      return_subject_ids=False, return_snp_names=False,
                      return_label_names=False):

    #print path

    if nolabels == 'raw' or not transpose:
        # Load raw data either for supervised or unsupervised part
        x, y, subject_ids, snp_names, label_names = thousand_genomes.load_data(path)
        x = x.astype("float32")

        (x, y, subject_ids) = shuffle((x, y, subject_ids))  # seed is fixed, shuffle is always the same

        # Prepare training and validation sets
        assert len(label_splits) == 1  # train/valid split
        # 5-fold cross validation: this means that test will always be 20%
        all_folds = split([x, y, subject_ids], [.2, .2, .2, .2])
        assert fold >= 0
        assert fold < 5

        # Separate choosen test set
        test = all_folds[fold]
        all_folds = all_folds[:fold] + all_folds[(fold + 1):]

        x = numpy.concatenate([el[0] for el in all_folds])
        y = numpy.concatenate([el[1] for el in all_folds])
        subject_ids = numpy.concatenate([el[2] for el in all_folds])

    # Preprocess the inputs to replace the missing values with the corresponding
    # feature's mean
    values_not_missing_mask = (x >= 0)
    per_feature_mean = (x * values_not_missing_mask).sum(0) / (values_not_missing_mask.sum(0))
    for i in range(x.shape[0]):
        x[i] = values_not_missing_mask[i] * x[i] + (1 - values_not_missing_mask[i]) * per_feature_mean

    # Data used for supervised training
    if not transpose:
        train, valid = split([x, y, subject_ids], label_splits)
        if norm:
            mu = x.mean(axis=0)
            sigma = x.std(axis=0) + 1e-6
            train[0] = (train[0] - mu[None, :]) / sigma[None, :]
            valid[0] = (valid[0] - mu[None, :]) / sigma[None, :]
            test[0] = (test[0] - mu[None, :]) / sigma[None, :]

        if return_subject_ids:
            rvals = [train, valid, test]
        else:
            rvals = [train[:2], valid[:2], test[:2]]
    else:
        rvals = []

    # Data used for transpose part or unsupervised training
    if nolabels == 'raw' and not transpose:
        unsupervised_data = None  # x.transpose()
    elif nolabels == 'raw' and transpose:
        unsupervised_data = x.transpose()
    elif nolabels == 'histo3':
        unsupervised_data = numpy.load(os.path.join(path, 'histo3_fold' +
                                    str(fold) + '.npy'))
    elif nolabels == 'histo3x26':
        unsupervised_data = numpy.load(os.path.join(path, 'histo3x26_fold' +
                                    str(fold) + '.npy'))
    elif nolabels == 'bin':
        unsupervised_data = numpy.load(os.path.join(path, 'snp_bin_fold' +
                                        str(fold) + '.npy'))
    elif nolabels == 'w2v':
        raise NotImplementedError
    else:
        try:
            unsupervised_data = numpy.load(nolabels)
        except:
            raise ValueError('Could not load specified embedding source')

    if transpose:
        assert len(feature_splits) == 1  # train/valid split feature-wise
        (unsupervised_data, ) = shuffle((unsupervised_data,))
        rvals += split([unsupervised_data], feature_splits)
    else:
        rvals += [unsupervised_data]

    if return_snp_names:
        rvals += [snp_names]

    if return_label_names:
        rvals += [label_names]

    return rvals


def load_1000G_WTCCC(path, fold=0, norm=True):
    # Load the data file
    thgen_x, thgen_y, wtccc_x, wtccc_y = numpy.load(path)

    thgen_x = thgen_x.astype("float32")
    wtccc_x = wtccc_x.astype("float32")

    test = [wtccc_x, wtccc_y]

    # Split the 1000G data into folds and then into train and valid
    (thgen_x, thgen_y) = shuffle((thgen_x, thgen_y))  # seed is fixed, shuffle is always the same

    all_folds = split([thgen_x, thgen_y], [.2, .2, .2, .2])

    valid = all_folds[fold]
    remaining_folds = all_folds[:fold] + all_folds[(fold + 1):]
    train = [numpy.concatenate([el[0] for el in all_folds]),
             numpy.concatenate([el[1] for el in all_folds])]

    # Normalize the data, if needed
    if norm:
        mu = thgen_x.mean(axis=0)
        sigma = thgen_x.std(axis=0) + 1e-6

        train[0] = (train[0] - mu[None, :]) / sigma[None, :]
        valid[0] = (valid[0] - mu[None, :]) / sigma[None, :]
        test[0] = (test[0] - mu[None, :]) / sigma[None, :]

    return train, valid, test, None


def load_1000G_ASW(path, fold=0, norm=True, return_subject_ids=False,
                   return_snp_names=False, return_label_names=False):
    # Load the data file
    (thgen_x, asw_x, thgen_y, asw_y, thgen_subject_ids, asw_subject_ids,
     snp_names, label_names) = asw_dataset.load_data(path)

    thgen_x = thgen_x.astype("float32")
    asw_x = asw_x.astype("float32")

    test = [asw_x, None, asw_subject_ids] # No test labels

    # Split the 1000G data into folds and then into train and valid
    # (seed is fixed, shuffle is always the same)
    (thgen_x, thgen_y, thgen_subject_ids) = shuffle((thgen_x, thgen_y, thgen_subject_ids))
    all_folds = split([thgen_x, thgen_y, thgen_subject_ids], [.2, .2, .2, .2])

    valid = all_folds[fold]
    remaining_folds = all_folds[:fold] + all_folds[(fold + 1):]
    train = [numpy.concatenate([el[0] for el in all_folds]),
             numpy.concatenate([el[1] for el in all_folds]),
             numpy.concatenate([el[2] for el in all_folds])]

    # Normalize the data, if needed
    if norm:
        mu = thgen_x.mean(axis=0)
        sigma = thgen_x.std(axis=0) + 1e-6

        train[0] = (train[0] - mu[None, :]) / sigma[None, :]
        valid[0] = (valid[0] - mu[None, :]) / sigma[None, :]
        test[0] = (test[0] - mu[None, :]) / sigma[None, :]

    if return_subject_ids:
        rvals = [train, valid, test, None]
    else:
        rvals = [train[:2], valid[:2], test[:2], None]

    if return_snp_names:
        rvals += [snp_names]

    if return_label_names:
        rvals += [label_names]

    return rvals


def load_antibiotic_resistance(path, dataset, transpose=False, label_splits=None, feature_splits=None,
                               nolabels='raw', fold=0, norm=False):

    assert dataset in ['amikacin__pseudomonas_aeruginosa',
                       'beta-lactam__streptococcus_pneumoniae',
                       'carbapenem__acinetobacter_baumannii',
                       'gentamicin__staphylococcus_aureus',
                       'isoniazid__mycobacterium_tuberculosis']

    hdf5_file_name = os.path.join(path, dataset+'.h5')
    print (hdf5_file_name)

    fold_name = 'train_0.800_seed_' + str(fold)

    if nolabels == 'raw' or not transpose:
        # Load raw data either for supervised or unsupervised part
        f = h5py.File(hdf5_file_name, 'r')
        x = f['X'][()]
        y = f['y'][()]
        splits = f['splits']

        train_idx = splits[fold_name]['train_example_idx'][()]
        test_idx = splits[fold_name]['test_example_idx'][()]

        train_x = x[train_idx, :]
        train_y = y[train_idx]
        test_x = x[test_idx, :]
        test_y = y[test_idx]
        del x,y

        test = (test_x, test_y)
        del test_x,test_y
        f.close()

    # Data used for supervised training
    if not transpose:
        train, valid = split([train_x, train_y], label_splits)
        if norm:
            # TODO: Doesn't make sense with uint8, recheck if ever used
            raise ValueError('This normalization might not make any sense, please check!')
            mu = x.mean(axis=0)
            sigma = x.std(axis=0)
            train[0] = (train[0] - mu[None, :]) / sigma[None, :]
            valid[0] = (valid[0] - mu[None, :]) / sigma[None, :]
            test[0] = (test[0] - mu[None, :]) / sigma[None, :]
        rvals = [train, valid, test]
    else:
        rvals = []

    # Data used for transpose part or unsupervised training
    if nolabels == 'raw' and not transpose:
        unsupervised_data = None  # x.transpose()
    elif nolabels == 'raw' and transpose:
        unsupervised_data = x.transpose()
    elif nolabels == 'histo3':
        raise NotImplementedError('histo3 has to be generated!')
    elif nolabels == 'histo3x26':
        raise NotImplementedError('histo3x26 has to be generated!')
    elif nolabels == 'bin':
        raise NotImplementedError('bin (snp2vec) has to be generated!')
    else:
        raise ValueError('Unknown feature embedding method')

    if transpose:
        assert len(feature_splits) == 1  # train/valid split feature-wise
        (unsupervised_data, ) = shuffle((unsupervised_data,))
        rvals += split([unsupervised_data], feature_splits)
    else:
        rvals += [unsupervised_data]

    return rvals
