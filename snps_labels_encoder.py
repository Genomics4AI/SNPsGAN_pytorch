#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 13:39:11 2020

@author: chemarestrepo
"""
import os
import numpy as np
import allel
import json
import pandas as pd
import random
import pickle
import easydict


args = easydict.EasyDict({
    "nb_sub_variants": 2000,   # number of subsampled variants
    "nb_sub_subjects": 250,    # number of subsampled samples
    "seed": 313            # seed used for reproducibility
})


path = '/Users/chemarestrepo/Desktop/AI_for_genomics/SNPsGAN/Dataset'
#geno_file = 'affy_harmonized_maf005_pruned.vcf.recode.vcf' #use when full dataset is required
geno_file = 'sample_affy_harmonized_maf005_pruned.pkl'
pheno_file = 'pheno_file_1KGP_subset.csv'

#load sample_affy_harmonized_maf005_pruned.pkl
file_path = os.path.join(path,geno_file)
with open(file_path, 'rb') as f:
    callset_subset = pickle.load(f)
gt_subset = allel.GenotypeArray(callset_subset['calldata/GT'])

#when loading full dataset, use this block to load callset
#additionally, replace 'callset_subset' for 'callset' in the subsequent code blocks
file_path = os.path.join(path,geno_file) #make sure geno_file is not 'sample_affy...'
callset = allel.read_vcf(file_path)
gt = allel.GenotypeArray(callset['calldata/GT'])

#trim repeat in samples
samples = callset_subset['samples'].tolist() #remove '_subset' when using full dataset
samples_trimmed = []
for name in samples:
    samples_trimmed.append(name.split('_')[0])

#Populate df with genotypes (encoded as 0, 1, or 2)
df = pd.DataFrame()
df['subjects'] = samples_trimmed
ids = callset_subset['variants/ID'].tolist() #remove '_subset' when using full dataset
for snp, genotype in enumerate(gt_subset): #remove '_subset' when using full dataset
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
