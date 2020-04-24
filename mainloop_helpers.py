#!/usr/bin/env python
from __future__ import print_function
import math
import numpy as np
import os
import random
import dataset_utils as du

# Function to load data
def load_data(dataset, dataset_path, embedding_source,
              which_fold=0, keep_labels=1., missing_labels_val=1.,
              embedding_input='raw', transpose=False, norm=True):

    # Load data from specified dataset
    splits = [.6, .2]  # this will split the data into [60%, 20%, 20%]
    if dataset == '1000_genomes':
        # This will split the training data into 75% train, 25%
        # this corresponds to the split 60/20 of the whole data,
        # test is considered elsewhere as an extra 20% of the whole data
        splits = [.75]
        data = du.load_1000_genomes(dataset_path,
                                    transpose=transpose,
                                    label_splits=splits,
                                    feature_splits=[.8],
                                    fold=which_fold,
                                    nolabels=embedding_input,
                                    norm=norm, return_subject_ids=True,
                                    return_snp_names=True,
                                    return_label_names=True)

        if transpose:
            return data
        else:
            ((x_train, y_train, exmpl_ids_train), 
             (x_valid, y_valid, exmpl_ids_valid),
             (x_test, y_test, exmpl_ids_test),
             x_nolabel, feature_names, label_names) = data

    elif dataset in  ['amikacin__pseudomonas_aeruginosa',
                      'beta-lactam__streptococcus_pneumoniae',
                      'carbapenem__acinetobacter_baumannii',
                      'gentamicin__staphylococcus_aureus',
                      'isoniazid__mycobacterium_tuberculosis']:
        data = du.load_antibiotic_resistance(dataset_path, dataset,
                                             transpose=transpose,
                                             label_splits=[.8],
                                             feature_splits=[.8],
                                             fold=which_fold,
                                             nolabels=embedding_input,
                                             norm=False)

        if transpose:
            return data
        else:
            (x_train, y_train), (x_valid, y_valid), (x_test, y_test), x_nolabel = data
            exmpl_ids_train = ["" for i in range(x_train.shape[0])]
            exmpl_ids_valid = ["" for i in range(x_valid.shape[0])]
            exmpl_ids_test = ["" for i in range(x_test.shape[0])]
            feature_names = ["" for i in range(x_train.shape[1])]
            label_names = None

    elif dataset == "1000G_WTCCC":
        data = du.load_1000G_WTCCC(dataset_path, fold=which_fold, norm=norm)
        ((x_train, y_train, exmpl_ids_train), 
         (x_valid, y_valid, exmpl_ids_valid),
         (x_test, y_test, exmpl_ids_test),
         x_nolabel, feature_names, label_names) = data
    elif dataset == "1000G_ASW":
        data = du.load_1000G_ASW(dataset_path, fold=which_fold, norm=norm,
                                 return_subject_ids=True, return_snp_names=True,
                                 return_label_names=True)
        ((x_train, y_train, exmpl_ids_train), 
         (x_valid, y_valid, exmpl_ids_valid),
         (x_test, y_test, exmpl_ids_test),
         x_nolabel, feature_names, label_names) = data
    else:
        print("Unknown dataset")
        return

    if not embedding_source:
        if x_nolabel is None:
            x_unsup = x_train.transpose()
        else:
            x_unsup = x_nolabel
    else:
        x_unsup = None

    # If needed, remove some of the training labels
    if keep_labels <= 1.0:
        training_labels = y_train.copy()
        random.seed(23)
        nb_train = len(training_labels)

        indices = range(nb_train)
        random.shuffle(indices)

        indices_discard = indices[:int(nb_train * (1 - keep_labels))]
        for idx in indices_discard:
            training_labels[idx] = missing_labels_val
    else:
        training_labels = y_train

    return (x_train, y_train, exmpl_ids_train,
            x_valid, y_valid, exmpl_ids_valid,
            x_test, y_test, exmpl_ids_test,
            x_unsup, training_labels, feature_names, label_names)


def load_external_test_inputs(filename):
    """
    The external test inputs are expected to be in the following
    tab-separated-values format

    subjects    snp_id1 snp_id2 snp_id3
    subject_id1    0   0   0
    subject_id2    2   1   0
    subject_id3    0   NA  0

    SNP values must be provided in additive coding format (0, 1 or 2) with
    missing values signaled by the characters "NA". The subject ids can be any
    string of consecutive alphanumerical characters and as well as "-" and "_".
    """

    # Load the genomic data file
    nb_header_columns = 1
    with open(filename, "r") as f:
        lines = f.readlines()
        snp_names = np.array(lines[0].split()[nb_header_columns:])
        data_lines = lines[1:]
    headers = [l.split()[:nb_header_columns] for l in data_lines]
    subject_ids = np.array([h[0] for h in headers])

    nb_features = len(l.split()[nb_header_columns:])
    genomic_data = np.empty((len(data_lines), nb_features), dtype="int8")
    for idx, line in enumerate(data_lines):
        genomic_data[idx] = [int(e) for e in line.replace("NA", "-1").split()[nb_header_columns:]]

    return genomic_data, snp_names, subject_ids


def define_exp_name(keep_labels, alpha, beta, gamma, lmd, n_hidden_u,
                    n_hidden_t_enc, n_hidden_t_dec, n_hidden_s, which_fold,
                    lr, dni, eni, batchnorm, input_dropout, embedding_noise,
                    earlystop, anneal, input_decoder_mode):
    # Define experiment name from parameters
    exp_name = 'our_model' + str(keep_labels) + \
        '_lr-' + str(lr) + '_anneal-' + str(anneal)  +\
        ('_eni-' + str(eni) if eni > 0 else '') + \
        ('_dni-' + str(dni) if dni > 0 else '') + \
        '_' + earlystop + \
        '_BN-' + str(batchnorm) + \
        '_Inpdrp-' + str(input_dropout) + \
        '_EmbNoise-' + str(embedding_noise) + \
        ('_Ri'+str(gamma) if gamma > 0 else '') + ('_Rwenc' if alpha > 0 else '') + \
        ('_Rwdec' if beta > 0 else '') + \
        (('_l2-' + str(lmd)) if lmd > 0. else '') + \
        ('_decmode-' + input_decoder_mode )
    exp_name += '_hu'
    for i in range(len(n_hidden_u)):
        exp_name += ("-" + str(n_hidden_u[i]))
    exp_name += '_tenc'
    for i in range(len(n_hidden_t_enc)):
        exp_name += ("-" + str(n_hidden_t_enc[i]))
    exp_name += '_tdec'
    for i in range(len(n_hidden_t_dec)):
        exp_name += ("-" + str(n_hidden_t_dec[i]))
    exp_name += '_hs'
    for i in range(len(n_hidden_s)):
        exp_name += ("-" + str(n_hidden_s[i]))
    exp_name += '_fold' + str(which_fold)

    return exp_name


# Mini-batch iterator function
def iterate_minibatches(inputs, targets, batchsize,
                        shuffle=False):
    assert inputs.shape[0] == targets.shape[0]
    indices = np.arange(inputs.shape[0])
    if shuffle:
        indices = np.random.permutation(inputs.shape[0])

    for i in range(0, inputs.shape[0], batchsize):
        yield inputs[indices[i:i+batchsize], :],\
            targets[indices[i:i+batchsize]]

def iterate_minibatches_unsup(x, batch_size, shuffle=False):
    indices = np.arange(x.shape[0])
    if shuffle:
        indices = np.random.permutation(x.shape[0])
    for i in range(0, x.shape[0]-batch_size+1, batch_size):
        yield x[indices[i:i+batch_size], :]


def iterate_testbatches(inputs, batchsize, shuffle=False):
    indices = np.arange(inputs.shape[0])
    if shuffle:
        indices = np.random.permutation(inputs.shape[0])
    for i in range(0, inputs.shape[0]-batchsize+1, batchsize):
        yield inputs[indices[i:i+batchsize], :]


def get_precision_recall_cutoff(predictions, targets):

    prev_threshold = 0.00
    threshold_inc = 0.10

    while True:
        if prev_threshold > 1.000:
            cutoff = 0.0
            break

        threshold = prev_threshold + threshold_inc
        tp = ((predictions >= threshold) * (targets == 1)).sum()
        fp = ((predictions >= threshold) * (targets == 0)).sum()
        fn = ((predictions < threshold) * (targets == 1)).sum()

        precision = float(tp) / (tp + fp + 1e-20)
        recall = float(tp) / (tp + fn + 1e-20)

        if precision > recall:
            if threshold_inc < 0.001:
                cutoff = recall
                break
            else:
                threshold_inc /= 10
        else:
            prev_threshold += threshold_inc

    return cutoff


# Monitoring function
def monitoring(minibatches, which_set, error_fn, monitoring_labels,
               prec_recall_cutoff=True, start=1, return_pred=False):
    print('-'*20 + which_set + ' monit.' + '-'*20)
    prec_recall_cutoff = False if start == 0 else prec_recall_cutoff
    monitoring_values = np.zeros(len(monitoring_labels), dtype="float32")
    nb_monitoring_examples = 0

    targets = []
    predictions = []

    for batch in minibatches:
        # Update monitored values
        if start == 0:
            out = error_fn(batch)
            import pdb; pdb.set_trace()
            # minibatch_size = batch.shape[0]
        else:
            out = error_fn(*batch)
            minibatch_size = batch[0].shape[0]

        monitoring_values += np.array(out[start:]) * minibatch_size
        if start == 1:
            predictions.append(out[0])
        targets.append(batch[1])
        nb_monitoring_examples += minibatch_size

    # Print monitored values
    monitoring_values /= nb_monitoring_examples
    for (label, val) in zip(monitoring_labels, monitoring_values):
        print ("  {} {}:\t\t{:.6f}".format(which_set, label, val))

    # If needed, compute and print the precision-recall breakoff point
    if prec_recall_cutoff:
        predictions = np.vstack(predictions)
        targets = np.vstack(targets)
        cutoff = get_precision_recall_cutoff(predictions, targets)
        print ("  {} precis/recall cutoff:\t{:.6f}".format(which_set, cutoff))

    if return_pred:
        return monitoring_values, np.vstack(predictions), np.vstack(targets)
    else:
        return monitoring_values


def eval_prediction(minibatches, dataset_name, predict_fn, norm_mus=None,
                    norm_sigmas=None, nb_evals=1, rescale_inputs=False):
    """
    Attempt to compute some curve of prediction accuracy
    For each value in list_percent_missing_snp, go through the
    whole test set and, for each example, take turn in removing
    a certain % of the input values

    Params:
        minibatches : iterator providing minibatches over which to iterate to
            compute prediction accuracy
        dataset_name : string. Used when printing the results of the prediction
            evaluation
        predict_fn : theano function that gives the network's ethnicity
            prediction
        norm_mus and norm_sigmas : if provided, will be used to preprocess the
            input minibatches before feeding them to the network.
        nb_evals : number of times to perform the evaluation and average over
        rescale_inputs : boolean. True if inputs values must be rescaled to
            compensate for the proportion of missing SNPs so that the total
            "amount of signal" will be constant (like dropout). This scaling
            will be applied after the normalization (if norm_mus and
            norm_sigmas are provided)
        """
        
    print("\nEvaluating performance on %s set with missing SNPs" % dataset_name)
    
    # Make the minibatches into a list so we can iterate over it as many times
    # as we need to.
    minibatches = [m for m in minibatches]

    list_percent_missing_snp = [0.01, 0.02, 0.05, 0.10, 0.20, 0.25, 0.50,
                                0.75, 0.80, 0.90, 0.95, 0.98, 0.99]

    for percent_missing_snp in list_percent_missing_snp:

        per_batch_correct_predictions = []

        for batch in minibatches:
            inputs, targets_onehot = batch

            targets = targets_onehot.argmax(1)
            n_feats = inputs.shape[1]

            batch_correct_predictions = []

            for eval_idx in range(nb_evals):

                # If the proportion of missing SNPs is higher than 0.50,
                # instead of separating all the SNPs into groups we will remove
                # one by one, we separate the SNPs into groups we will *keep*
                # one by one. This is done by changing the group sizes here
                # and flipping the SNP binary mask later
                if percent_missing_snp > 0.50:
                    percent_kept_snp = 1 - percent_missing_snp
                    snp_group_size = int(math.ceil(n_feats * percent_kept_snp))
                    reverse_mask = True
                else:
                    snp_group_size = int(math.ceil(n_feats * percent_missing_snp))
                    reverse_mask = False

                # Separate all the Snps into random groups of identical size
                snp_indices = np.arange(n_feats)
                np.random.shuffle(snp_indices)

                snp_groups = [snp_indices[i:i + snp_group_size] for i in
                              xrange(0, len(snp_indices), snp_group_size)]

                for snp_group_idx in range(len(snp_groups)):
                    group_indices = snp_groups[snp_group_idx]

                    # Create masks that will be used to identify SNPs to impute
                    input_mask = np.zeros((n_feats,), dtype="float32")
                    np.put(input_mask, group_indices, 1)

                    # If the groups represent SNPs to keep and not SNPs to
                    # discard, flip the SNP mask
                    if reverse_mask:
                        input_mask = (1 - input_mask)
                        
                    if norm_mus is not None and norm_sigmas is not None:
                        normed_inputs = (inputs - norm_mus) / norm_sigmas
                    else:
                        normed_inputs = inputs

                    # If needed, rescale the inputs to compensate for the
                    # proportion of missing SNPs
                    if rescale_inputs:
                        scale_factor = 1 / (1 - percent_missing_snp)
                    else:
                        scale_factor = 1.0

                    # Obtain the model's prediction with masked inputs
                    masked_inputs = normed_inputs * (1 - input_mask) * scale_factor
                    predictions = predict_fn(masked_inputs)

                    # Compute the number of successful ethnicity predictions
                    batch_correct_predictions.append(predictions == targets)

            batch_correct_predictions = np.vstack(batch_correct_predictions)
            per_batch_correct_predictions.append(batch_correct_predictions)

        per_batch_correct_predictions = np.hstack(per_batch_correct_predictions)

        print("%s prediction accuracy at with %i percent of missing SNPs : %f +- %f" %
              (dataset_name, percent_missing_snp * 100,
               per_batch_correct_predictions.mean(),
               per_batch_correct_predictions.mean(1).std()))


def parse_int_list_arg(arg):
    if isinstance(arg, str):
        arg = eval(arg)

    if isinstance(arg, list):
        return arg
    if isinstance(arg, int):
        return [arg]
    else:
        raise ValueError("Following arg value could not be cast as a list of"
                         "integer values : " % arg)


def parse_string_int_tuple(arg):
    if isinstance(arg, (list, tuple)):
        return arg
    elif isinstance(arg, str):
        tmp = arg.strip("()[]").split(",")
        assert (len(tmp) == 2)
        return (tmp[0], eval(tmp[1]))
    else:
        raise NotImplementedError()


def get_grads_wrt_inputs(inputs, grad_fn, input_names, class_names):
    # For each class, get the gradients of that class' prediction wrt the 
    # inputs and then sort by average value of that gradient
    grads_wrt_inputs = {}
    for i in range(len(class_names)):
        gs = grad_fn(inputs, i)
        sorted_gs = sorted(zip(gs, input_names))
        grads_wrt_inputs[class_names[i]] = sorted_gs

    return grads_wrt_inputs


def get_integrated_gradients(inputs, grad_from_normed_fn, input_names, class_names,
                             norm_mus, norm_sigmas, m=100):

    # Normalize inputs and obtain reference inputs
    normed_inputs = ((inputs == -1) * norm_mus +
                     (inputs != -1) * inputs.astype("float32"))
    normed_inputs = (normed_inputs - norm_mus) / norm_sigmas
    ref_inputs = normed_inputs * 0

    # For each class, get the integrated gradients of that class' predictions
    # wrt the inputs and then sort by average value of that gradient
    grads_wrt_inputs = []
    interpolation_factors = (np.arange(m, dtype="float32") / m)[:, None]
    itt_inputs = ref_inputs + interpolation_factors * (normed_inputs - ref_inputs)
    for i in range(len(class_names)):

        # Compute integrated gradients
        gs = grad_from_normed_fn(itt_inputs, i) * (normed_inputs - ref_inputs)
        grads_wrt_inputs.append(gs)

    return np.stack(grads_wrt_inputs)
