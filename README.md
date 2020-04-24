# SNPsGAN_pytorch

Overview: The majority of studies in population genetics make use of data consisting of single nucleotide polymorphisms (SNPs); however, in many circumstances, the number of required data samples (i.e.,individuals) is not sufficient enough to train a supervised deep learning model; that is mainly due to a large number of features present in the data, compared to the number of samples (widely known as the curse of dimensionality). Moreover, the issue of privacy restricts the public distribution of real data samples. Overcoming the inability to share genetic data across different projects and groups can help accelerate research and enhance outcomes and predictions. In this study, by leveraging deep generative models (namely conditional generative adversarial networks (cGANS) and conditional variational autoencoders (cVAEs)) we attempt to deal with such scarcity and privacy problems by synthesizing data samples of SNPs with particular attributes of interest.


## Dataset

Initially, we will use the 1000 genomes project dataset, consisting of 3450 subjects from 26 different populations, and for each subject we have 294,427 single nucleotide variants with a minor allele frequency > 0.05 (i.e. genetic variation that consist of a single nucleotide change, for which the less frequent allele occurs in at least 5% of the sample). Additionally, we obtained information regarding which of the 26 populations under study is each subject coming from, in order to use this information as our labels in the histogram embedding.

## Preprocessing

Using scikit-allel we extracted sample ids, snp ids, and genotypes in order to construct an initial array of shape (# of snps, # of subjects) where each cell can take the value 0, 1, or 2, representing the number of reference alleles that each subject has in their genotype. 

* In order to obtain the initial array, run *genotype_encoder.py*. It will output the files *snps.txt*. Before you continue, make sure you have the labels in a file named 'labels.txt' with 2 columns ('sample','class'). Also, make sure the sample IDs match in both 'snps.txt' and 'labels.txt'.

* In order for the code to run faster in the next steps, run *external_dataset.py* (for now, use python version 2). This will parse our dataset to a single binary file which is smaller and faster to work with. The output of this script is a single *dataset.npy* file.

* Once *dataset.npy* is generated, run *utils_helpers.py* to generate the feature embeddings for the dataset. This will output 5 files in with the names  *histo_AllelicFrequency_perclass_fold{0-4}*

