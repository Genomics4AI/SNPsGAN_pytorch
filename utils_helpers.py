#!/usr/bin/env python
import os
import numpy as np
import sys

import dataset_utils as du


def generate_1000_genomes_hist(path, fold, perclass, transpose=False,
        label_splits=None, feature_splits=None, sum_to_one=True, is_asw_dataset=False):
    """
    Can generate embeddings based on populations using genotypic and allelic
    frequencies

    Parameters:
        perclass: If this is true we will compute the frequency_type for each
            population
        sum_to_one: By default (the default is false), the function will return
            the miminum amount of information per SNPs. For exemple, if we have
            a SNP with a major allelic frequency of 0.8, then we will only
            store and use 0.8 (since 0.2, can be infered easily). At the time
            of writing this, it remains to be shown that putting this parameter
            to false doesn't harm the results.
    """
    if is_asw_dataset:
        train, valid, test, _, label_names = du.load_1000G_ASW(
            path=path, fold=fold, norm=False, return_label_names=True)
    else:
        train, valid, test, _, label_names = du.load_1000_genomes(
            transpose=transpose, label_splits=label_splits,
            feature_splits=feature_splits, fold=fold, norm=False,
            nolabels='raw', path=path, return_label_names=True)
    """
    train, valid, test, _, label_names = du.load_1000G_ASW(
            path=path, fold=fold, norm=False, return_label_names=True)
    """

    # Generate no_label: fuse train and valid sets
    nolabel_orig = (np.vstack([train[0], valid[0]])).transpose()
    nolabel_y = np.vstack([train[1], valid[1]])
    nolabel_y = nolabel_y.argmax(axis=1)

    filename_suffix = ( '_perclass' if perclass else '') + \
                ( '' if sum_to_one else '_SumToOne') + \
                '_fold' + str(fold) + '.npy'
    filename_genotypic = 'histo_' + 'GenotypicFrequency' + filename_suffix
    filename_allelic = 'histo_' + 'AllelicFrequency' + filename_suffix

    generate_snp_hist(nolabel_orig, nolabel_y, label_names, perclass,
                      sum_to_one,
                      os.path.join(path, filename_genotypic),
                      os.path.join(path, filename_allelic))


def generate_snp_hist(snp_data, subject_labels, label_names, perclass,
                      sum_to_one=True, filename_genotypic=None,
                      filename_allelic=None):
    """
    Can generate embeddings based on populations using genotypic and allelic
    frequencies

    Parameters:
        perclass: If this is true we will compute the frequency_type for each
            population
        sum_to_one: By default (the default is false), the function will return
            the miminum amount of information per SNPs. For exemple, if we have
            a SNP with a major allelic frequency of 0.8, then we will only
            store and use 0.8 (since 0.2, can be infered easily). At the time
            of writing this, it remains to be shown that putting this parameter
            to false doesn't harm the results.
    """

    assert subject_labels.ndim in [1, 2]

    NB_POPULATION = len(label_names)
    NB_POSSIBLE_GENOTYPE = 3 # homref, alt, homalt

    if perclass:
        # the first dimension of the following is length 'number of snps'
        nolabel_x = np.zeros((snp_data.shape[0], NB_POSSIBLE_GENOTYPE*NB_POPULATION))
        nolabel_x_allelic = np.zeros((snp_data.shape[0], 2*NB_POPULATION))
        class_indices = zip(NB_POSSIBLE_GENOTYPE*np.arange(NB_POPULATION),
                NB_POSSIBLE_GENOTYPE*np.arange(NB_POPULATION) + NB_POSSIBLE_GENOTYPE);
	#import pdb;pdb.set_trace()
        # Compute counts per SNP per genotype
        for lower_index, upper_index in class_indices:

            if subject_labels.ndim == 1:
                # Using hard labels for examples; pick only the examples of the
                # right class and compute the SNP counts on those examples.
                class_examples = snp_data[:, subject_labels == lower_index/3].astype('int32')
                for genotype in range(NB_POSSIBLE_GENOTYPE):
                    nolabel_x[:, lower_index + genotype] = (class_examples == genotype).sum(1)

            elif subject_labels.ndim == 2:
                # Using soft labels for examples: compute the SNP counts from
                # all examples but
                for genotype in range(NB_POSSIBLE_GENOTYPE):
                    nolabel_x[:, lower_index + genotype] = ((snp_data == genotype) *
                                                            subject_labels[:,lower_index/3]).sum(1)

        # We normalize genotypic counts for all populations and compute the
        # allelic frequencies
        for lower_index, upper_index in class_indices:
            nolabel_x[:, lower_index:upper_index] = \
                nolabel_x[:, lower_index:upper_index] / np.array([nolabel_x[:,
                    lower_index:upper_index].sum(axis=1)]).T
            nolabel_x_allelic[:, lower_index/3*2] = \
                0.5*(2*nolabel_x[:, lower_index] + nolabel_x[:, lower_index+1])
            nolabel_x_allelic[:, lower_index/3*2 + 1] = \
                    1 - nolabel_x_allelic[:, lower_index/3*2]
    else:
        nolabel_x = np.zeros((snp_data.shape[0], NB_POSSIBLE_GENOTYPE))
        nolabel_x_allelic = np.zeros((snp_data.shape[0], 2))
        for i in range(nolabel_x.shape[0]):
            nolabel_x[i, :] += np.bincount(snp_data[i, :].astype('int32'),
                                           minlength=NB_POSSIBLE_GENOTYPE)
            nolabel_x[i, :] /= nolabel_x[i, :].sum()
        nolabel_x_allelic[:, 0] = \
            0.5*(2*nolabel_x[:, 0] + nolabel_x[:, 1])
        nolabel_x_allelic[:, 1] = \
                1 - nolabel_x_allelic[:, 0]

    nolabel_x = nolabel_x.astype('float32')

    # We remove redundant information if sum_to_one is False
    if not sum_to_one:
        if perclass:
            nolabel_x = nolabel_x[:,np.array(zip(np.arange(NB_POPULATION)*NB_POSSIBLE_GENOTYPE,np.arange(NB_POPULATION)*NB_POSSIBLE_GENOTYPE + 1)).reshape(2*NB_POPULATION)]
            nolabel_x_allelic =  nolabel_x_allelic[:, np.arange(NB_POPULATION)*2]
        else:
            nolabel_x = nolabel_x[:,0:-1]
            nolabel_x_allelic =  nolabel_x_allelic[:,-1]

    # We determine the filename based on frequency_type, perclass and
    # sum_to_one params
    if filename_genotypic is not None:
        np.save(filename_genotypic, nolabel_x)
    if filename_allelic is not None:
        np.save(filename_allelic, nolabel_x_allelic)


def generate_1000_genomes_bag_of_genes(
        transpose=False, label_splits=None,
        feature_splits=[0.8], fold=0,
        path = '/data/milatmp1/carriepl/datasets/1000_Genome_project/'):

    train, valid, test, _ = du.load_1000_genomes(transpose, label_splits,
                                                 feature_splits, fold,
                                                 norm=False)

    nolabel_orig = (np.vstack([train[0], valid[0]]))

    if not os.path.isdir(path):
        os.makedirs(path)

    filename = 'unsupervised_bag_of_genes'
    filename += '_fold' + str(fold) + '.npy'

    nolabel_x = np.zeros((nolabel_orig.shape[0], nolabel_orig.shape[1]*2))

    mod1 = np.zeros(nolabel_orig.shape)
    mod2 = np.zeros(nolabel_orig.shape)

    for i in range(nolabel_x.shape[0]):
        mod1[i, :] = np.where(nolabel_orig[i, :] > 0)[0]*2 + 1
        mod2[i, :] = np.where(nolabel_orig == 2)[0]*2

    nolabel_x[mod1] += 1
    nolabel_x[mod2] += 1
    # import ipdb; ipdb.set_trace()


def generate_1000_genomes_snp2bin(transpose=False, label_splits=None,
                                  feature_splits=None, fold=0,
                                  path = '/data/milatmp1/carriepl/datasets/1000_Genome_project/'):

    train, valid, test, _ = du.load_1000_genomes_old(transpose, label_splits,
                                                     feature_splits, fold,
                                                     norm=False)

    # Generate no_label: fuse train and valid sets
    nolabel_orig = (np.vstack([train[0], valid[0]]))
    nolabel_x = np.zeros((nolabel_orig.shape[0], nolabel_orig.shape[1]*2),
                         dtype='uint8')


    filename = 'unsupervised_snp_bin_fold' + str(fold) + '.npy'

    # SNP to bin
    nolabel_x[:, ::2] += (nolabel_orig == 2)
    nolabel_x[:, 1::2] += (nolabel_orig >= 1)

    np.save(os.path.join(path, filename), nolabel_x)

if __name__ == '__main__':
    from os import sys, path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    # parameters
    folds = range(5)
    perclass = True
    path = sys.argv[1]

    if len(sys.argv) > 2:
        # Value specified for is_asw_dataset
        is_asw_dataset = sys.argv[2] == "asw"
    else:
        is_asw_dataset = False

    print("Generating embeddings for the provided dataset")
    for f in folds:
        print 'Processing fold:', str(f)
        generate_1000_genomes_hist(transpose=False, label_splits=[.75],
                                   feature_splits=[1.], fold=f, perclass=True,
                                   path=path, is_asw_dataset=is_asw_dataset)

    """
    print("Generate embedding for the 'MAF0.05 Prunned0.5' dataset")
    for f in folds:
        print 'Processing fold:', str(f)
        generate_1000_genomes_hist(transpose=False, label_splits=[.75],
                                   feature_splits=[1.], fold=f, perclass=True,
                                   path='/data/milatmp1/carriepl/datasets/1000_Genome_project/data_files/maf005_prunned05/')

    print("Generate embedding for the 'Prunned0.35' dataset")
    for f in folds:
        print 'Processing fold:', str(f)
        generate_1000_genomes_hist(transpose=False, label_splits=[.75],
                                   feature_splits=[1.], fold=f, perclass=True,
                                   path='/data/milatmp1/carriepl/datasets/1000_Genome_project/data_files/prunned035/')

    print("Generate embedding for the 'Prunned0.5' dataset")
    for f in folds:
        print 'Processing fold:', str(f)
        generate_1000_genomes_hist(transpose=False, label_splits=[.75],
                                   feature_splits=[1.], fold=f, perclass=True,
                                   path='/data/milatmp1/carriepl/datasets/1000_Genome_project/data_files/prunned05/')

    print("Generate embedding for the complementary dataset to the 400k SNPs dataset with imputation and rare SNPs")
    for f in folds:
        print 'Processing fold:', str(f)
        generate_1000_genomes_hist(transpose=False, label_splits=[.75],
                                   feature_splits=[1.], fold=f, perclass=True,
                                   path='/data/milatmp1/carriepl/datasets/1000_Genome_project/data_files/900k_imputation_minus_400_imputation_with_rare/')

    for f in folds:
        print 'processing fold:', str(f)
        generate_1000_genomes_hist(transpose=False, label_splits=[.75],
                                   feature_splits=[1.], fold=f,
                                   perclass=perclass, path=path,
                                   sum_to_one=True)

    print("Generate embedding for the 300k SNPs dataset")
    for f in folds:
        print 'Processing fold:', str(f)
        generate_1000_genomes_hist(transpose=False, label_splits=[.75],
                                   feature_splits=[1.], fold=f, perclass=True,
                                   path='/data/milatmp1/carriepl/datasets/1000_Genome_project/data_files/300k/')

    print("Generate embedding for the 400k SNPs dataset with imputation and rare SNPs")
    for f in folds:
        print 'Processing fold:', str(f)
        generate_1000_genomes_hist(transpose=False, label_splits=[.75],
                                   feature_splits=[1.], fold=f, perclass=True,
                                   path='/data/milatmp1/carriepl/datasets/1000_Genome_project/data_files/400k_imputation_with_rare/')


    print("Generate embedding for the 600k SNPs dataset")
    for f in folds:
        print 'Processing fold:', str(f)
        generate_1000_genomes_hist(transpose=False, label_splits=[.75],
                                   feature_splits=[1.], fold=f, perclass=True,
                                   path='/data/milatmp1/carriepl/datasets/1000_Genome_project/data_files/600k_imputation/')

    print("Generate embedding for the pruned 600k SNPs dataset")
    for f in folds:
        print 'Processing fold:', str(f)
        generate_1000_genomes_hist(transpose=False, label_splits=[.75],
                                   feature_splits=[1.], fold=f, perclass=True,
                                   path='/data/milatmp1/carriepl/datasets/1000_Genome_project/data_files/600k_imputation_pruned/')

    print("Generate embedding for the 900k SNPs dataset (no imputation")
    for f in folds:
        print 'Processing fold:', str(f)
        generate_1000_genomes_hist(transpose=False, label_splits=[.75],
                                   feature_splits=[1.], fold=f, perclass=True,
                                   path='/data/milatmp1/carriepl/datasets/1000_Genome_project/data_files/900k_noimputation/')

    print("Generate embedding for the 900k SNPs dataset (with imputation)")
    for f in folds:
        print 'Processing fold:', str(f)
        generate_1000_genomes_hist(transpose=False, label_splits=[.75],
                                   feature_splits=[1.], fold=f, perclass=True,
                                   path='/data/milatmp1/carriepl/datasets/1000_Genome_project/data_files/900k_imputation/')

    print("Generate embeddings for the dataset split by chromosome")
    for c in range(1, 23):
        for f in folds:
            print("Generating embedding for chromosome %i (no imputation), fold %i" % (c, f))
            path = '/data/milatmp1/carriepl/datasets/1000_Genome_project/data_files/per_chromosome_noimputation/chr%i/' % c
            generate_1000_genomes_hist(transpose=False, label_splits=[.75],
                                       feature_splits=[1.], fold=f, perclass=True,
                                       path=path)

    for c in range(1, 23):
        for f in folds:
            print("Generating embedding for chromosome %i (imputation), fold %i" % (c, f))
            path = '/data/milatmp1/carriepl/datasets/1000_Genome_project/data_files/per_chromosome_imputation/chr%i/' % c
            generate_1000_genomes_hist(transpose=False, label_splits=[.75],
                                       feature_splits=[1.], fold=f, perclass=True,
                                       path=path)
    """
