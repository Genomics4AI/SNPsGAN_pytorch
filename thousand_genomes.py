
#!/usr/bin/env python
import pickle
import numpy
import os





def load_data(path, force_binary_recreation=False, onehot_encoding=False):

    """
    force_binary_recreation : If true, will force the recreation of the binary
        dataset file even if it already exists
    onehot_encoding : If false, the values for a SNP will be encoded as a single
        continuous value from {-1, 0, 1, 2}. If true, the values for a SNP will
        be encoded as a onehot vector :
            {-1 : [0,0,0], 0: [1, 0, 0], 1: [0, 1, 0], 2: [0, 0, 1]}
    """

    """
    if os.getenv('USER') ==  "barakatm":
        dataset_file = "affy_6_biallelic_snps_maf005_thinned_aut_imputed_dataset.pkl"
        genome_file = "affy_6_biallelic_snps_maf005_thinned_aut_imputed_A.raw.gz"
    else:
        dataset_file = "dataset.npy"
        genome_file = "snps.raw"
    """
    dataset_file = "dataset.npy"
    genome_file = "snps.raw"
    label_file = "affy_samples.20141118.panel"

    # If this function has already been called in the past, it will have saved
    # the result of the data parsing. These results are simply read and
    # returned to the called.
    # WARNING : Previous versions of this method only saved the genomic data
    # and label data to file. When loading previously parsed data, we need
    # to account for the fact that it might not contain subject_ids and
    # snp_ids. If that is the case, dummy values are provided so as to not
    # break the interface.
    if os.path.exists(path + dataset_file) and not force_binary_recreation:
        with open(path + dataset_file, "rb") as f:

            data = numpy.load(f)
            genomic_data = data['inputs']
            label_data = data['labels']
            try:
                subject_ids = data['subject_ids']
                snp_names = data['snp_names']
            except:
                subject_ids = numpy.array(["" for i in range(genomic_data.shape[0])])
                snp_names = numpy.array(["" for i in range(genomic_data.shape[1])])

            try:
                label_names = data['label_names']
            except:
                label_names = ['ACB', 'ASW', 'BEB', 'CDX', 'CEU', 'CHB',
                               'CHS', 'CLM', 'ESN', 'FIN', 'GBR', 'GIH',
                               'GWD', 'IBS', 'ITU', 'JPT', 'KHV', 'LWK',
                               'MSL', 'MXL', 'PEL', 'PJL', 'PUR', 'STU',
                               'TSI', 'YRI']

        return genomic_data, label_data, subject_ids, snp_names, label_names

    print("No binary file has been found for this dataset. The data will "
          "be parsed to produce one. This will take a few minutes.")

    # Load the genomic data file
    with open(path + genome_file, "r") as f:
        lines = f.readlines()
        snp_names = numpy.array(lines[0].split()[6:])
        data_lines = lines[1:]
    headers = [l.split()[:6] for l in data_lines]
    subject_ids = numpy.array([h[0] for h in headers])

    nb_features = len(l.split()[6:])
    genomic_data = numpy.empty((len(data_lines), nb_features), dtype="int8")
    for idx, line in enumerate(data_lines):
        if idx % 100 == 0:
            print("Parsing subject %i out of %i" % (idx, len(data_lines)))
        genomic_data[idx] = [int(e) for e in line.replace("NA", "-1").split()[6:]]

    # If using onehot_encoding, change the data format
    if onehot_encoding:
        new_genomic_data = numpy.empty((len(data_lines), nb_features * 3), dtype="int8")
        new_genomic_data[:,::3] = genomic_data == 0
        new_genomic_data[:,1::3] = genomic_data == 1
        new_genomic_data[:,2::3] = genomic_data == 2
        genomic_data = new_genomic_data

    # Load the label file
    label_dict = {}
    with open(path + label_file, "r") as f:
        for line in f.readlines()[1:]:
            patient_id, ethnicity, _ = line.split()
            label_dict[patient_id] = ethnicity

    # Transform the label into a one-hot format
    all_labels = list(set(label_dict.values()))
    all_labels.sort()

    label_data = numpy.zeros((genomic_data.shape[0], len(all_labels)),
                             dtype="float32")
    for subject_idx in range(len(headers)):
        subject_id = headers[subject_idx][0]
        subject_label = label_dict[subject_id]
        label_idx = all_labels.index(subject_label)
        label_data[subject_idx, label_idx] = 1.0

    # Save the parsed data to the filesystem
    print("Saving parsed data to a binary format for faster loading in the future.")
    with open(path + dataset_file, "wb") as f:
        numpy.savez(f, inputs=genomic_data, labels=label_data,
                    subject_ids=subject_ids, snp_names=snp_names,
                    label_names=all_labels)

    return genomic_data, label_data, subject_ids, snp_names, all_labels


if __name__ == '__main__':

    load_data(path="/data/milatmp1/carriepl/datasets/1000_Genome_project/data_files/300k/")
    print("Load '300k no imputation' done")

    load_data(path="/data/milatmp1/carriepl/datasets/1000_Genome_project/data_files/900k_noimputation/")
    print("Load '900k no imputation' done")

    load_data(path="/data/milatmp1/carriepl/datasets/1000_Genome_project/data_files/900k_imputation/")
    print("Load '900k imputation' done")

    # Process the data split by chromosome (with imputation)
    for i in range(1, 23):
        load_data(path="/data/milatmp1/carriepl/datasets/1000_Genome_project/data_files/per_chromosome_imputation/chr%i/" % i)
        print("Load chromosome (imputation) %i done" % i)

    # Process the data split by chromosome (with imputation)
    for i in range(1, 23):
        load_data(path="/data/milatmp1/carriepl/datasets/1000_Genome_project/data_files/per_chromosome_noimputation/chr%i/" % i)
        print("Load chromosome (no_imputation) %i done" % i)

    # Process datasets with different LD and MAF filtering parameters
    load_data(path="/data/milatmp1/carriepl/datasets/1000_Genome_project/data_files/maf005_prunned05/")
    print("Load 'MAF0.05 Prunned0.5' done")

    load_data(path="/data/milatmp1/carriepl/datasets/1000_Genome_project/data_files/prunned035/")
    print("Load 'Prunned0.35' done")

    load_data(path="/data/milatmp1/carriepl/datasets/1000_Genome_project/data_files/prunned05/")
    print("Load 'MPrunned0.5' done")