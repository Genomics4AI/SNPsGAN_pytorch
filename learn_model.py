#!/usr/bin/env python
from __future__ import print_function
import argparse
import time
import os
from dir_util import copy_tree

import sys
import numpy as np
import theano
from theano import config
import theano.tensor as T

import mainloop_helpers as mlh
import model_helpers as mh

import utils_helpers
import lasagne

from lasagne.regularization import apply_penalty, l2

print ("config floatX: {}".format(config.floatX))

####need to give password somewhere
import getpass

# Main program
def execute(dataset, n_hidden_u, n_hidden_t_enc, n_hidden_t_dec, n_hidden_s,
            embedding_source=histo_GenotypicFrequency_perclass, additional_unsup_input=None,
            num_epochs=500, learning_rate=.001, learning_rate_annealing=1.0,
            alpha=1, beta=1, delta=1, gamma=1, lmd=.0001,
            disc_nonlinearity="sigmoid", encoder_net_init=0.2,
            decoder_net_init=0.2, optimizer="rmsprop", max_patience=100,
            batchnorm=0, input_dropout=1.0, embedding_noise=0.0, keep_labels=1.0,
            prec_recall_cutoff=True, missing_labels_val=-1.0, which_fold=0,
            early_stop_criterion='loss_sup_det',
            input_decoder_mode="regression",
            save_path='melyse@cedar.calculcanada.ca:/home/melyse/projects/def-awazana/melyse/', 
            save_copy='melyse@cedar.calculcanada.ca:/home/melyse/projects/def-awazana/melyse/',
            dataset_path='melyse@cedar.calculcanada.ca:/home/melyse/projects/def-awazana/melyse/',
            resume=False, exp_name='', random_proj=0,
            bootstrap_snp_embeddings=0, bootstrap_cutoff=0.9):

    # Prepare embedding information :
    # - If no embedding is specified, use the transposed input matrix
    # - If a file is specified, use it's content as feature embeddings
    # - Else (a embedding category like  'histo3x26' is provided), load a
    #   pregenerated embedding of the specified category
    if embedding_source is None or embedding_source == "raw":
        embedding_source = None
        embedding_input = 'raw'
    elif os.path.exists(embedding_source):
        embedding_input = embedding_source
    else:
        embedding_input = embedding_source
        embedding_source = os.path.join(dataset_path, embedding_input + '_fold' + str(which_fold) + '.npy')
        
    # Load the dataset
    print("Loading data")
    (x_train, y_train, exmpl_ids_train,
     x_valid, y_valid, exmpl_ids_valid,
     x_test, y_test, exmpl_ids_test,
     x_unsup, training_labels, feature_names, label_names) = mlh.load_data(
            dataset, dataset_path, embedding_source,
            which_fold=which_fold, keep_labels=keep_labels,
            missing_labels_val=missing_labels_val,
            embedding_input=embedding_input, norm=False)

    # Load the additional unsupervised data, if some is specified
    if additional_unsup_input is not None:
        print("Adding additional data to the model's unsupervised inputs")
        paths = additional_unsup_input.split(";")
        additional_unsup_data = [np.load(p) for p in paths]
        print(x_unsup.shape)
        x_unsup = np.hstack(additional_unsup_data + [x_unsup])
        print(x_unsup.shape)
    
    if x_unsup is not None:
        n_samples_unsup = x_unsup.shape[1]
    else:
        n_samples_unsup = 0
    
    original_x_train = x_train.copy()
    original_x_valid = x_valid.copy()
    original_x_test = x_test.copy()
    
    # Change how the missing data values are encoded. Right now they are
    # encoded as being the mean of the corresponding feature so that, after
    # feature normalization, they will be 0s. However, this prevents us from
    # transfering the minibatch data as int8 so we replace those values with -1s.
    for i in range(x_train.shape[1]):
        feature_mean = x_train[:, i].mean()
        x_train[:, i] = mh.replace_arr_value(x_train[:, i], feature_mean, -1)
        x_valid[:, i] = mh.replace_arr_value(x_valid[:, i], feature_mean, -1)
        x_test[:, i] = mh.replace_arr_value(x_test[:, i], feature_mean, -1)
    x_train = x_train.astype("int8")
    x_valid = x_valid.astype("int8")
    x_test = x_test.astype("int8")

    # Normalize the input data. The mlh.load_data() function already offers
    # this feature but we need to do it here so that we will have access to
    # both the normalized and unnormalized input data.
    norm_mus = original_x_train.mean(axis=0)
    norm_sigmas = original_x_train.std(axis=0) + 1e-6

    #x_train = (x_train - norm_mus[None, :]) / norm_sigmas[None, :]
    #x_valid = (x_valid - norm_mus[None, :]) / norm_sigmas[None, :]
    #x_test = (x_test - norm_mus[None, :]) / norm_sigmas[None, :]

    #x_train *= (315345. / 553107)
    #x_valid *= (315345. / 553107)
    #x_test *= (315345. / 553107)


    # Setup variables to build the right type of decoder bases on the value of
    # `input_decoder_mode`
    assert input_decoder_mode in ["regression", "classification"]
    if input_decoder_mode == "regression":
        # The size of the input reconstruction will be the same as the number
        # of inputs
        decoder_encoder_unit_ratio = 1
    elif input_decoder_mode == "classification":
        # # The size of the input reconstruction will be the N times larger as
        # the number of inputs where N is the number of distinct discrete
        # values that each input can take. For SNP input data with an additive
        # coding scheme, N=3 because the 3 possible values are : {0, 1, 2}.
        nb_discrete_vals_by_input = int(original_x_train.max() + 1)
        decoder_encoder_unit_ratio = nb_discrete_vals_by_input
        
        # Print baseline accuracy for the imputation of genes
        print("Distribution of input values in valid: %f %f %f" %
              ((original_x_train == 0).mean(), (original_x_train == 1).mean(),
               (original_x_train == 2).mean()))
        print("Distribution of input values in test: %f %f %f" %
              ((original_x_test == 0).mean(), (original_x_test == 1).mean(),
               (original_x_test == 2).mean()))

    # Extract required information from data
    n_samples, n_feats = x_train.shape
    print("Number of features : ", n_feats)
    print("Glorot init : ", 2.0 / (n_feats + n_hidden_t_enc[-1]))
    n_targets = y_train.shape[1] if y_train.ndim == 2 else y_train.max()+1

    # Set some variables
    batch_size = 138
    beta = gamma if (gamma == 0) else beta

    # Generate an name for the experiment based on the hyperparameters used
    if embedding_source is None:
        embedding_name = embedding_input
    else:
        embedding_name = embedding_source.replace("_", "").split(".")[0]
        exp_name += embedding_name.rsplit('/', 1)[::-1][0] + '_'

    exp_name += mlh.define_exp_name(keep_labels, alpha, beta, gamma, lmd,
                                    n_hidden_u, n_hidden_t_enc, n_hidden_t_dec,
                                    n_hidden_s, which_fold,
                                    learning_rate, decoder_net_init,
                                    encoder_net_init, batchnorm, input_dropout,
                                    embedding_noise, early_stop_criterion,
                                    learning_rate_annealing, input_decoder_mode)
    print("Experiment: " + exp_name)
    
    # Ensure that the folders where the results of the experiment will be
    # saved do exist. Create them if they don't.
    save_path = os.path.join(save_path, dataset, exp_name)
    save_copy = os.path.join(save_copy, dataset, exp_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not os.path.exists(save_copy):
        os.makedirs(save_copy)

    # Prepare Theano variables for inputs and targets
    input_var_sup = T.bmatrix('input_sup')
    input_var_unsup = theano.shared(x_unsup, 'input_unsup')  # x_unsup TBD
    target_var_sup = T.matrix('target_sup')
    lr = theano.shared(np.float32(learning_rate), 'learning_rate')

    # Use the provided mus and sigmas to process the missing values and
    # normalize the inputs
    b_input_var_sup = input_var_sup.astype("float32")
    normed_input_sup = (T.eq(b_input_var_sup, -1) * norm_mus +
                        T.neq(b_input_var_sup, -1) * b_input_var_sup)
    normed_input_sup = (normed_input_sup - norm_mus) / norm_sigmas

    reconst_target_sup = T.cast(input_var_sup, "int32")

    # Build model
    print("Building model")

    # Some checkings
    # assert len(n_hidden_u) > 0
    assert len(n_hidden_t_enc) > 0
    assert len(n_hidden_t_dec) > 0
    assert n_hidden_t_dec[-1] == n_hidden_t_enc[-1]

    # Build feature embedding networks (encoding and decoding if gamma > 0)
    nets, embeddings, pred_feat_emb = mh.build_feat_emb_nets(
        embedding_source, n_feats, n_samples_unsup,
        input_var_unsup, n_hidden_u, n_hidden_t_enc,
        n_hidden_t_dec, gamma, encoder_net_init,
        decoder_net_init, save_path, random_proj,
        decoder_encoder_unit_ratio, embedding_noise)

    # Build feature embedding reconstruction networks (if alpha > 0, beta > 0)
    nets += mh.build_feat_emb_reconst_nets(
            [alpha, beta], n_samples_unsup, n_hidden_u,
            [n_hidden_t_enc, n_hidden_t_dec],
            nets, [encoder_net_init, decoder_net_init])

    # Supervised network
    discrim_net, hidden_rep = mh.build_discrim_net(
        batch_size, n_feats, normed_input_sup, n_hidden_t_enc,
        n_hidden_s, embeddings[0], disc_nonlinearity, n_targets, batchnorm,
        input_dropout)

    # Reconstruct network
    nets += [mh.build_reconst_net(hidden_rep,
                                  embeddings[1] if len(embeddings) > 1 else None,
                                  n_feats * decoder_encoder_unit_ratio, gamma,
                                  decoder_encoder_unit_ratio)]

    # Load weights if we are resuming job
    if resume:
        # Load best model
        with np.load(os.path.join(save_copy, 'dietnet_best.npz')) as f:
            param_values = [f['arr_%d' % i]
                            for i in range(len(f.files))]
        nlayers = len(lasagne.layers.get_all_params(filter(None, nets) +
                                                    [discrim_net]))
        #lasagne.layers.set_all_param_values(filter(None, nets) +
        #                                    [discrim_net],
        #                                    param_values[:nlayers])

        params = lasagne.layers.get_all_params(filter(None, nets) + [discrim_net])
        for p, v in zip(params, param_values[:nlayers]):
            # Do not overwrite embedding value with old embedding. Removing
            # the following condition will prevent a trained model from being
            # tested on a different dataset
            if p.name != "feat_emb":
                p.set_value(v)


    print("Building and compiling training functions")

    # Build and compile training functions
    predictions, predictions_det = mh.define_predictions(nets, start=2)
    prediction_sup, prediction_sup_det = mh.define_predictions([discrim_net])
    prediction_sup = prediction_sup[0]
    prediction_sup_det = prediction_sup_det[0]

    # Define losses
    # reconstruction losses
    if input_decoder_mode == "regression":
        reconst_losses, reconst_losses_det = mh.define_reconst_losses(
            predictions, predictions_det,
            [input_var_unsup, input_var_unsup, normed_input_sup])
    elif input_decoder_mode == "classification":
        # Obtain regular reconstruction losses for every reconstruction
        # but the reconstruction of the supervised input data
        reconst_losses1, reconst_losses_det1 = mh.define_reconst_losses(
            predictions[:-1], predictions_det[:-1],
            [input_var_unsup, input_var_unsup])
        
        # Obtain a "classification" reconstruction loss for the reconstruction
        # of the supervised input data. This classification loss will be
        # performed on the input data without normalization
        reconst_losses2, reconst_losses_det2 = mh.define_classif_reconst_losses(
            predictions[-1:], predictions_det[-1:], [reconst_target_sup],
            [decoder_encoder_unit_ratio])
        
        reconst_losses = reconst_losses1 + reconst_losses2
        reconst_losses_det = reconst_losses_det1 + reconst_losses_det2
    
    # supervised loss
    sup_loss, sup_loss_det = mh.define_sup_loss(
        disc_nonlinearity, prediction_sup, prediction_sup_det, keep_labels,
        target_var_sup, missing_labels_val)

    # Define inputs
    inputs = [input_var_sup, target_var_sup]

    # Define parameters
    params = lasagne.layers.get_all_params(
        [discrim_net]+filter(None, nets), trainable=True, unwrap_shared=False)
    params_to_freeze= \
        lasagne.layers.get_all_params(filter(None, nets), trainable=False,
                                      unwrap_shared=False)

    # Remove unshared variables from params and params_to_freeze
    params = [p for p in params if isinstance(p, theano.compile.sharedvalue.SharedVariable)]
    params_to_freeze = [p for p in params_to_freeze if isinstance(p, theano.compile.sharedvalue.SharedVariable)]
    print("Params : ", params)

    feat_emb_var = next(p for p in lasagne.layers.get_all_params([discrim_net]) if p.name == 'input_unsup' or p.name == 'feat_emb')
    # feat_emb_var = lasagne.layers.get_all_params([discrim_net])[0]
    print(feat_emb_var)
    feat_emb_val = feat_emb_var.get_value()
    feat_emb_norms = (feat_emb_val ** 2).sum(0) ** 0.5
    feat_emb_var.set_value(feat_emb_val / feat_emb_norms)

    print('Number of params discrim: '+str(len(params)))
    print('Number of params to freeze: '+str(len(params_to_freeze)))

    for p in params_to_freeze:
        new_params = [el for el in params if el != p]
        params = new_params

    print('Number of params to update: '+ str(len(params)))

    # Combine losses
    loss = delta*sup_loss + alpha*reconst_losses[0] + beta*reconst_losses[1] + \
        gamma*reconst_losses[2]
    loss_det = delta*sup_loss_det + alpha*reconst_losses_det[0] + \
        beta*reconst_losses_det[1] + gamma*reconst_losses_det[2]

    l2_penalty = apply_penalty(params, l2)
    loss = loss + lmd*l2_penalty
    loss_det = loss_det + lmd*l2_penalty

    # Compute network updates
    assert optimizer in ["rmsprop", "adam", "amsgrad"]
    if optimizer == "rmsprop":
        updates = lasagne.updates.rmsprop(loss,
                                        params,
                                        learning_rate=lr)
    elif optimizer == "adam":
        updates = lasagne.updates.adam(loss,
                                       params,
                                       learning_rate=lr)
    elif optimizer == "amsgrad":
        updates = lasagne.updates.amsgrad(loss,
                                          params,
                                          learning_rate=lr)
    #updates = lasagne.updates.sgd(loss,
    #                              params,
    #                              learning_rate=lr)
    # updates = lasagne.updates.momentum(loss, params,
    #                                    learning_rate=lr, momentum=0.0)

    # Apply norm constraints on the weights
    for k in updates.keys():
        if updates[k].ndim == 2:
            updates[k] = lasagne.updates.norm_constraint(updates[k], 1.0)

    # Compile training function
    train_fn = theano.function(inputs, loss, updates=updates,
                               on_unused_input='ignore')

    # Monitoring Labels
    monitor_labels = ["reconst. feat. W_enc",
                      "reconst. feat. W_dec",
                      "reconst. loss"]
    monitor_labels = [i for i, j in zip(monitor_labels, reconst_losses)
                      if j != 0]
    monitor_labels += ["feat. W_enc. mean", "feat. W_enc var"]
    monitor_labels += ["feat. W_dec. mean", "feat. W_dec var"] if \
        (embeddings[1] is not None) else []
    monitor_labels += ["loss. sup.", "total loss"]

    # Build and compile test function
    val_outputs = reconst_losses_det
    val_outputs = [i for i, j in zip(val_outputs, reconst_losses) if j != 0]
    val_outputs += [embeddings[0].mean(), embeddings[0].var()]
    val_outputs += [embeddings[1].mean(), embeddings[1].var()] if \
        (embeddings[1] is not None) else []
    val_outputs += [sup_loss_det, loss_det]

    # Compute supervised accuracy and add it to monitoring list
    test_acc, test_pred = mh.define_test_functions(
        disc_nonlinearity, prediction_sup, prediction_sup_det, target_var_sup)
    monitor_labels.append("accuracy")
    val_outputs.append(test_acc)
    
    # If appropriate, compute the input reconstruction accuracy and add it to
    # the monitoring list
    if input_decoder_mode == "classification":
        input_reconst_acc = mh.define_classif_reconst_acc(predictions_det[-1],
                                reconst_target_sup, decoder_encoder_unit_ratio)
        #import pdb; pdb.set_trace()
        monitor_labels.append("input_reconst_acc")
        val_outputs.append(input_reconst_acc)


    # Compile prediction function
    predict = theano.function([input_var_sup], test_pred)
    predict_from_normed_inps = theano.function([normed_input_sup], test_pred)
    
    predict_scores = theano.function([input_var_sup], prediction_sup_det)
    predict_scores_from_normed_inps = theano.function([input_var_sup],
                                                      prediction_sup_det)

    # Compile validation function
    val_fn = theano.function(inputs,
                             [prediction_sup_det] + val_outputs,
                             on_unused_input='ignore')

    # Finally, launch the training loop.
    print("Starting training...")

    # Some variables
    patience = 0

    train_monitored = []
    valid_monitored = []
    train_loss = []

    # Pre-training monitoring
    print("Epoch 0 of {}".format(num_epochs))

    train_minibatches = mlh.iterate_minibatches(x_train, y_train,
                                                batch_size, shuffle=False)
    train_err = mlh.monitoring(train_minibatches, "train", val_fn,
                               monitor_labels, prec_recall_cutoff)

    valid_minibatches = mlh.iterate_minibatches(x_valid, y_valid,
                                                batch_size, shuffle=False)
    valid_err = mlh.monitoring(valid_minibatches, "valid", val_fn,
                               monitor_labels, prec_recall_cutoff)

    # Before starting training, save a copy of the model in case
    np.savez(os.path.join(save_path, 'dietnet_best.npz'),
             *lasagne.layers.get_all_param_values(filter(None, nets) +
                                                  [discrim_net]))

    # Training loop
    start_training = time.time()
    for epoch in range(num_epochs):
        start_time = time.time()
        print("Epoch {} of {}".format(epoch+1, num_epochs))
        nb_minibatches = 0
        loss_epoch = 0

        # Train pass
        for batch in mlh.iterate_minibatches(x_train, training_labels,
                                             batch_size,
                                             shuffle=True):
            loss_epoch += train_fn(*batch)
            nb_minibatches += 1

        loss_epoch /= nb_minibatches
        train_loss += [loss_epoch]

        # Monitoring on the training set
        train_minibatches = mlh.iterate_minibatches(x_train, y_train,
                                                    batch_size, shuffle=False)
        train_err = mlh.monitoring(train_minibatches, "train", val_fn,
                                   monitor_labels, prec_recall_cutoff)
        train_monitored += [train_err]

        # Monitoring on the validation set
        valid_minibatches = mlh.iterate_minibatches(x_valid, y_valid,
                                                    batch_size, shuffle=False)

        valid_err = mlh.monitoring(valid_minibatches, "valid", val_fn,
                                   monitor_labels, prec_recall_cutoff)
        valid_monitored += [valid_err]

        try:
            early_stop_val = valid_err[
                monitor_labels.index(early_stop_criterion)]
        except:
            raise ValueError("There is no monitored value by the name of %s" %
                             early_stop_criterion)

        valid_loss_sup_hist = [v[monitor_labels.index("loss. sup.")] for v in valid_monitored]
        valid_loss_sup = valid_loss_sup_hist[-1]

        # Early stopping
        if epoch == 0:
            best_valid = early_stop_val
        elif ((early_stop_val > best_valid and early_stop_criterion == 'input_reconst_acc') or
              (early_stop_val > best_valid and early_stop_criterion == 'accuracy') or
              (early_stop_val >= best_valid and early_stop_criterion == 'accuracy' and
               valid_loss_sup == min(valid_loss_sup_hist)) or
              (early_stop_val < best_valid and early_stop_criterion == 'loss. sup.')):
            best_valid = early_stop_val
            patience = 0

            # Save stuff
            np.savez(os.path.join(save_path, 'dietnet_best.npz'),
                     *lasagne.layers.get_all_param_values(filter(None, nets) +
                                                          [discrim_net]))
            np.savez(save_path + "/errors_supervised_best.npz",
                     zip(*train_monitored), zip(*valid_monitored))

            # Monitor on the test set now because sometimes the saving doesn't
            # go well and there isn't a model to load at the end of training
            if y_test is not None:
                test_minibatches = mlh.iterate_minibatches(x_test, y_test,
                                                           138,
                                                           shuffle=False)

                test_err = mlh.monitoring(test_minibatches, "test", val_fn,
                                          monitor_labels, prec_recall_cutoff)
        else:
            patience += 1
            # Save stuff
            np.savez(os.path.join(save_path, 'dietnet_last.npz'),
                     *lasagne.layers.get_all_param_values(filter(None, nets) +
                                                          [discrim_net]))
            np.savez(save_path + "/errors_supervised_last.npz",
                     zip(*train_monitored), zip(*valid_monitored))

        print("  epoch time:\t\t\t{:.3f}s \n".format(time.time() - start_time))

        # End training if needed
        if patience == max_patience or epoch == num_epochs-1:
            break

        # Anneal the learning rate
        lr.set_value(np.array(lr.get_value() * learning_rate_annealing, dtype="float32"))


    # End training with a final monitoring step on the best model
    print("Ending training")

    # Load best model
    with np.load(os.path.join(save_path, 'dietnet_best.npz')) as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        nlayers = len(lasagne.layers.get_all_params(filter(None, nets) +
                                                    [discrim_net]))

        #lasagne.layers.set_all_param_values(filter(None, nets) +
        #                                    [discrim_net],
        #                                    param_values[:nlayers])
        params = lasagne.layers.get_all_params(filter(None, nets) + [discrim_net])
        for p, v in zip(params, param_values[:nlayers]):
            # Do not overwrite embedding value with old embedding. Removing
            # the following condition will prevent a trained model from being
            # tested on a different dataset
            if p.name != "feat_emb":
                p.set_value(v)

        if embedding_source is None:
            # Save embedding
            pred = pred_feat_emb()
            np.savez(os.path.join(save_path, 'feature_embedding.npz'), pred)

        # Training set results
        train_minibatches = mlh.iterate_minibatches(x_train, y_train,
                                                    batch_size, shuffle=False)
        train_err = mlh.monitoring(train_minibatches, "train", val_fn,
                                   monitor_labels, prec_recall_cutoff)

        # Validation set results
        valid_minibatches = mlh.iterate_minibatches(x_valid, y_valid,
                                                    batch_size, shuffle=False)
        valid_err = mlh.monitoring(valid_minibatches, "valid", val_fn,
                                   monitor_labels, prec_recall_cutoff)

        # Test set results
        if y_test is not None:
            test_minibatches = mlh.iterate_minibatches(x_test, y_test,
                                                       138, shuffle=False)

            test_err = mlh.monitoring(test_minibatches, "test", val_fn,
                                      monitor_labels, prec_recall_cutoff)

            # Test the model's accuracy with varying levels of provided SNPs
            test_minibatches = mlh.iterate_minibatches(x_test, y_test,
                                                       138, shuffle=False)
            mlh.eval_prediction(test_minibatches, "test (rescaled)",
                                predict_from_normed_inps, norm_mus, norm_sigmas,
                                nb_evals=1, rescale_inputs=True)

        # Save the model's test predictions to file
        print(x_test.shape)
        test_predictions = []
        for minibatch in mlh.iterate_testbatches(x_test, 1, shuffle=False):
            test_predictions += [predict(minibatch)]
        print(len(test_predictions))
        print(sum([t.shape[0] for t in test_predictions]))
        np.savez(os.path.join(save_path, 'test_predictions.npz'),
                 test_predictions)

        # Get the scores assigned by the model to each class for each test sample
        test_scores = []
        for minibatch in mlh.iterate_testbatches(x_test, 1, shuffle=False):
            test_scores += [predict_scores(minibatch)]
        np.savez(os.path.join(save_path, 'test_scores.npz'), test_scores)

        # Generate new SNP embeddings using test examples labeled according
        # to the model's predictions
        if bootstrap_snp_embeddings:

            if bootstrap_cutoff == "soft":
                bootstrap_snp_data = np.hstack((x_train.transpose(),
                                                x_valid.transpose(),
                                                x_test.transpose()))
                bootstrap_labels = np.vstack((y_train,
                                              y_valid,
                                              np.array(test_scores)[:,0,:]))

                filename_genotypic = 'bootstrap_gen_snp_embeddings_softlabels.npy'
                filename_allelic = 'bootstrap_all_snp_embeddings_softlabels.npy'

            else: # Hard cutoff
                sure_test_idxs = np.argwhere((np.array(test_scores)[:,0,:] > bootstrap_cutoff).sum(1)).flatten()
                sure_test_inputs = x_test[sure_test_idxs]
                sure_test_preds = np.array(test_scores)[sure_test_idxs, 0].argmax(1)

                bootstrap_snp_data = np.hstack((x_train.transpose(),
                                                x_valid.transpose(),
                                                sure_test_inputs.transpose()))
                bootstrap_labels = np.hstack((y_train.argmax(1),
                                            y_valid.argmax(1),
                                            sure_test_preds))
                
                filename_genotypic = 'bootstrap_gen_snp_embeddings_cutoff%f.npy' % bootstrap_cutoff
                filename_allelic = 'bootstrap_all_snp_embeddings_cutoff%f.npy' % bootstrap_cutoff

            utils_helpers.generate_snp_hist(bootstrap_snp_data, bootstrap_labels,
                label_names=label_names, perclass=True, sum_to_one=True,
                filename_genotypic=os.path.join(save_path, filename_genotypic),
                filename_allelic=os.path.join(save_path, filename_allelic))

    # Print all final errors for train, validation and test
    print("Training time:\t\t\t{:.3f}s".format(time.time() - start_training))

    # Analyse the model gradients to determine the influence of each SNP on
    # each of the model's prediction
    print(label_names)
    class_idx = T.iscalar("class index")
    grad_fn = theano.function([input_var_sup, class_idx],
                              T.grad(prediction_sup_det[:, class_idx].mean(), input_var_sup).mean(0))
    grads_wrt_inputs = mlh.get_grads_wrt_inputs(x_test, grad_fn, feature_names,
                                                label_names)

    # Obtain function that takes as inputs normed inputs and returns the
    # gradient of a class score wrt the normed inputs themselves (this is
    # requird because computing the integrated gradients requires to be able
    # to interpolate between an example where all features are missing and an
    # example where any number of features are provided)
    grad_from_normed_fn = theano.function(
                            [normed_input_sup, class_idx],
                            T.grad(prediction_sup_det[:, class_idx].sum(), normed_input_sup).mean(0))
    
    # Collect integrated gradients over the whole test set. Obtain, for each
    # SNP, for each possible value (0, 1 or 2), the average contribution of that
    # value for what SNP to the score of each class.
    avg_int_grads = np.zeros((x_test.shape[1], 3, len(label_names)), dtype="float32")
    counts_int_grads = np.zeros((x_test.shape[1], 3), dtype="int32")
    for test_idx in range(x_test.shape[0]):
        int_grads = mlh.get_integrated_gradients(x_test[test_idx], grad_from_normed_fn,
                                                 feature_names, label_names, norm_mus,
                                                 norm_sigmas, m=100)

        snp_value_mask = np.arange(3) == x_test[test_idx][:, None]
        avg_int_grads += snp_value_mask[:, :, None] * int_grads.transpose()[:, None, :]
        counts_int_grads += snp_value_mask
    avg_int_grads = avg_int_grads / counts_int_grads[:, :, None]

    # Save all the additional information required for model analysis :
    # - Test predictions
    # - SNP IDs
    # - Subject IDs
    # - Normalization parameters for the input minibatches
    np.savez(os.path.join(save_path, 'additional_data.npz'),
             test_labels=y_test,
             test_scores=np.array(test_scores)[:, 0],
             test_predictions=np.array(test_predictions)[:, 0],
             norm_mus=norm_mus, norm_sigmas=norm_sigmas,
             grads_wrt_inputs=grads_wrt_inputs, exmpl_ids_train=exmpl_ids_train,
             exmpl_ids_valid=exmpl_ids_valid, exmpl_ids_test=exmpl_ids_test,
             feature_names=feature_names, label_names=label_names,
             avg_int_grads=avg_int_grads)

    # Copy files to loadpath (only if some training has beeen done so there
    # is a local saved version)
    if save_path != save_copy and num_epochs > 0:
        print('Copying model and other training files to {}'.format(save_copy))
        copy_tree(save_path, save_copy)


def main():
    parser = argparse.ArgumentParser(description="""Train Diet Networks""")
    parser.add_argument('--dataset',
                        default='1000_genomes',
                        help='Dataset.')
    parser.add_argument('--n_hidden_u',
                        default=[100],
                        help='List of unsupervised hidden units.')
    parser.add_argument('--n_hidden_t_enc',
                        default=[100],
                        help='List of theta transformation hidden units.')
    parser.add_argument('--n_hidden_t_dec',
                        default=[100],
                        help='List of theta_prime transformation hidden units')
    parser.add_argument('--n_hidden_s',
                        default=[100],
                        help='List of supervised hidden units.')
    parser.add_argument('--embedding_source',
                        default='histo3x26',
                        help='Source for the feature embedding. Either' +
                             'None or the name of a file from which' +
                             'to load a learned embedding')
    parser.add_argument('--additional_unsup_input',
                        default=None,

                        help="List of numpy files to load and add to the " +
                             "dataset's unsupervised inputs (separated by ';')")
    parser.add_argument('--num_epochs',
                        '-ne',
                        type=int,
                        default=500,
                        help="""Int to indicate the max'
                        'number of epochs.""")
    parser.add_argument('--learning_rate',
                        '-lr',
                        type=float,
                        default=0.0001,
                        help="""Float to indicate learning rate.""")
    parser.add_argument('--learning_rate_annealing',
                        '-lra',
                        type=float,
                        default=.99,
                        help="Float to indicate learning rate annealing rate.")
    parser.add_argument('--alpha',
                        '-a',
                        type=float,
                        default=0.,
                        help="""reconst_loss coeff. for auxiliary net W_enc""")
    parser.add_argument('--beta',
                        '-b',
                        type=float,
                        default=0.,
                        help="""reconst_loss coeff. for auxiliary net W_dec""")
    parser.add_argument('--delta',
                        '-d',
                        type=float,
                        default=1.,
                        help="""supervised classification loss coefficient""")
    parser.add_argument('--gamma',
                        '-g',
                        type=float,
                        default=10.,
                        help="""reconst_loss coeff. (used for aux net W-dec as well)""")
    parser.add_argument('--lmd',
                        '-l',
                        type=float,
                        default=.0,
                        help="""Weight decay coeff.""")
    parser.add_argument('--disc_nonlinearity',
                        '-nl',
                        default="softmax",
                        help="""Nonlinearity to use in disc_net's last layer""")
    parser.add_argument('--encoder_net_init',
                        '-eni',
                        type=float,
                        default=0.01,
                        help="Bounds of uniform initialization for " +
                             "encoder_net weights")
    parser.add_argument('--decoder_net_init',
                        '-dni',
                        type=float,
                        default=0.01,
                        help="Bounds of uniform initialization for " +
                             "decoder_net weights")
    parser.add_argument('--optimizer',
                        type=str,
                        default="rmsprop",
                        help="Optimizer to use for training (rmsprop or adam)")
    parser.add_argument('--patience',
                        type=int,
                        default=100,
                        help="Number of epochs without validation improvement" +
                             "after which to stop training")
    parser.add_argument('--batchnorm',
                        '-bn',
                        type=int,
                        default=0,
                        help="Whether to use BatchNorm in the main network")
    parser.add_argument('--input_dropout',
                        type=float,
                        default=1.0,
                        help="Whether to use dropout on the input of the main" +
                             "network. 1.0 means no dropout and a value " +
                             "between 0.0 and 1.0 corresponds to the " +
                             "probability of keeping each input value at " +
                             "every iteration")
    parser.add_argument('--embedding_noise',
                        type=float,
                        default=0.0,
                        help="Standard deviation of gaussion noise to apply "
                             "on feature embedding values during training.")
    parser.add_argument('--keep_labels',
                        type=float,
                        default=1.0,
                        help='Fraction of training labels to keep')
    parser.add_argument('--prec_recall_cutoff',
                        type=int,
                        default=0,
                        help='Whether to compute the precision-recall cutoff' +
                             'or not')
    parser.add_argument('--which_fold',
                        type=int,
                        default=0,
                        help='Which fold to use for cross-validation (0-4)')
    parser.add_argument('--early_stop_criterion',
                        default='accuracy',
                        help='What monitored variable to use for early-stopping')
    parser.add_argument('--input_decoder_mode',
                        default='regression',
                        help='What type of reconstruction to use for the ' +
                             'supervised inputs of the network. "regression ' +
                             'or "classification"')
    parser.add_argument('--save_tmp',
                        default= '/Users/Marie-Elyse/Downloads/embedding2/',
                        help='Path to save results.')
    parser.add_argument('--save_perm',
                        default='/Users/Marie-Elyse/Downloads/embedding2/',
                        help='Path to save results.')
    parser.add_argument('--dataset_path',
                        default='/Users/Marie-Elyse/Downloads/embedding2/',
                        help='Path to dataset')
    parser.add_argument('--save_tmp',
                        default= 'melyse@cedar.calculcanada.ca:/home/melyse/projects/def-awazana/melyse/'+ os.environ+'/DietNetworks/',
                        help='Path to save results.')
    parser.add_argument('--save_perm',
                        default='melyse@cedar.calculcanada.ca:/home/melyse/projects/def-awazana/melyse/'+ os.environ+'/DietNetworks/',
                        help='Path to save results.')
    parser.add_argument('--dataset_path',
                        default='melyse@cedar.calculcanada.ca:/home/melyse/projects/def-awazana/melyse/',
                        help='Path to dataset')
    parser.add_argument('-resume',
                        type=bool,
                        default=False,
                        help='Whether to resume job')
    parser.add_argument('-exp_name',
                        type=str,
                        default='dietnets_final_',
                        help='Experiment name that will be concatenated at the beginning of the generated name')
    parser.add_argument('--random_proj',
                        '-rp',
                        type=int,
                        default=0,
                        help="Whether to use random projections as embedding")
    parser.add_argument('--bootstrap_snp_embeddings',
                        type=int,
                        default=0,
                        help="Generate new SNP embeddings using test predictions")
    parser.add_argument('--bootstrap_cutoff',
                        type=str,
                        default="0.9",
                        help=("Cutoff to transform model predictions into hard labels. " +
                              "Float values like 0.5, 0.6 are used as cutoff; model test predictions that have higher " + 
                              "certainty than the cutoff are takes as ground truth labels for the boostrapped embeddings. " + 
                              "'soft' means that test examples won't be assigned to a specific class according to the model's " +
                              "predictions and instead will count as weighted examples for each class, proportional to the " +
                              "model's predicted score."))

    args = parser.parse_args()
    print ("Printing args")
    print (args)
    
    if args.bootstrap_cutoff == "soft":
        bootstrap_cutoff = "soft"
    else:
        bootstrap_cutoff = float(args.bootstrap_cutoff)

    execute(args.dataset,
            mlh.parse_int_list_arg(args.n_hidden_u),
            mlh.parse_int_list_arg(args.n_hidden_t_enc),
            mlh.parse_int_list_arg(args.n_hidden_t_dec),
            mlh.parse_int_list_arg(args.n_hidden_s),
            args.embedding_source,
            args.additional_unsup_input,
            int(args.num_epochs),
            args.learning_rate,
            args.learning_rate_annealing,
            args.alpha,
            args.beta,
            args.delta,
            args.gamma,
            args.lmd,
            args.disc_nonlinearity,
            args.encoder_net_init,
            args.decoder_net_init,
            args.optimizer,
            args.patience,
            args.batchnorm,
            args.input_dropout,
            args.embedding_noise,
            args.keep_labels,
            args.prec_recall_cutoff != 0, -1,
            args.which_fold,
            args.early_stop_criterion,
            args.input_decoder_mode,
            args.save_tmp,
            args.save_perm,
            args.dataset_path,
            args.resume,
            args.exp_name,
            int(args.random_proj),
            bootstrap_snp_embeddings=args.bootstrap_snp_embeddings,
            bootstrap_cutoff=bootstrap_cutoff)


if __name__ == '__main__':
    main()
