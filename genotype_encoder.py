#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 13:39:11 2020

@author: chemarestrepo
"""
import os
import numpy as np
import allel
import pandas as pd
import random
import pickle
import easydict


def generate_encoded_genotypes(path = '', geno_file='', subset=False,
                               subset_geno_file = ''):

    '''
    Generates genotype input for external_dataset.py
    Genotypes are encoded as either 0, 1, or 2, denoting the number of reference
    alleles in the sample's genotype
    Arg:
        if subset = True, encodes 2000 snps for 250 subjects
        This subset was generated using vcf_subset_generator.py
    Returns:
        snps.txt
    '''


    if subset:
        filepath = os.path.join(path,subset_geno_file)
        with open(file_path, 'rb') as f:
            callset = pickle.load(f)
    else:
        filepath = os.path.join(path,geno_file)
        with open(file_path, 'rb') as f:
            callset = pickle.load(f)

    #create allel.model.GenotypeArray
    gt = allel.GenotypeArray(callset['calldata/GT'])

    #trim repeated name in samples
    samples = callset['samples'].tolist()
    samples_trimmed = []
    for name in samples:
        samples_trimmed.append(name.split('_')[0])

    #Populate df with genotypes (encoded as 0, 1, or 2)
    df = pd.DataFrame()
    df['subjects'] = samples_trimmed
    ids = callset['variants/ID'].tolist()
    for snp, genotype in enumerate(gt):
        genotypes_per_snp = []
        for subject in genotype:
            genotype_per_subject = []
            if subject[0] == subject[1]:
                if subject[0] == 0:
                    genotype_per_subject = 2
                elif subject[0] == 1:
                    genotype_per_subject = 0
            elif subject[0] != subject[1]:
                genotype_per_subject = 1
            else:
                genotype_per_subject = 'NA'
        genotypes_per_snp.append(genotype_per_subject)
    df[ids[snp]] = genotypes_per_snp

    #save 'snps.txt' file
    df.to_csv(os.path.join(path,'snps.txt'), index=None, sep = '\t')


if __name__=="__main__":
    """
    fold=0: determines which fold to make test set. will not be used to construct
    the histogram. test is 20% data perclass: true => histogram embedding will be
    (px78) 78 = 3*26classes label_splits: train will be 75% val will be 25% of 80%
    total (since test)
    """
    generate_encoded_genotypes(path = '/Users/chemarestrepo/Desktop/AI_for_genomics/SNPsGAN/Dataset',
                               geno_file='affy_harmonized_maf005_pruned.pkl',subset=False,
                               subset_geno_file = 'sample_affy_harmonized_maf005_pruned.pkl')

