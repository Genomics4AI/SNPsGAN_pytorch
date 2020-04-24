#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 20:56:43 2020

@author: chemarestrepo
"""

import os
import numpy as np
import pandas as pd
import allel
import random
import pickle
import easydict


args = easydict.EasyDict({
    "nb_sub_variants": 2000,   # number of subsampled variants
    "nb_sub_subjects": 250,    # number of subsampled samples
    "seed": 313            # seed used for reproducibility
})

random.seed(args.seed)
np.random.seed(args.seed)

path = '/Users/chemarestrepo/Desktop/AI_for_genomics/SNPsGAN/Dataset'
geno_file = 'affy_harmonized_maf005_pruned.vcf.recode.vcf'
pheno_file = 'pheno_file_1KGP.csv'
file_path = os.path.join(path, geno_file)

callset = allel.read_vcf(file_path, numbers={'ALT': 1})
gt = allel.GenotypeArray(callset['calldata/GT'])

indx_variants = sorted(np.random.choice(range(gt.n_variants), args.nb_sub_variants, replace=False))
indx_samples = sorted(np.random.choice(range(gt.n_samples), args.nb_sub_subjects, replace=False))

callset_subset = {} # an empty dictionary
callset_subset['calldata/GT'] = np.array(gt.subset(indx_variants, indx_samples))
callset_subset['samples'] = callset['samples'][indx_samples]
callset_subset['variants/ALT'] = callset['variants/ALT'][indx_variants]
callset_subset['variants/CHROM'] = callset['variants/CHROM'][indx_variants]
callset_subset['variants/FILTER_PASS'] = callset['variants/FILTER_PASS'][indx_variants]
callset_subset['variants/ID'] = callset['variants/ID'][indx_variants]
callset_subset['variants/POS'] = callset['variants/POS'][indx_variants]
callset_subset['variants/QUAL'] = callset['variants/QUAL'][indx_variants]
callset_subset['variants/REF'] = callset['variants/REF'][indx_variants]

with open("sample_affy_harmonized_maf005_pruned.pkl", 'wb') as f:
    pickle.dump(callset_subset, f)

#create subset for phenotype file containing labels
pheno = pd.read_csv(os.path.join(path,pheno_file))
pheno = pheno.iloc[indx_samples,[0,-1]]
pheno.rename(columns={'Sample': 'sample', 'Population':'class'}, inplace=True)
pheno.to_csv(os.path.join(path,'labels.txt'), index=None, sep = '\t')

