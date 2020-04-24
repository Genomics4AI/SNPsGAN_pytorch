import os
import argparse
from os import path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable
import socket
import datetime
import random
import re
import requests
import gzip
import logomaker # https://github.com/jbkinney/logomaker/tree/master/logomaker/tutorials
import easydict
from pytz import timezone
import matplotlib
import matplotlib.pyplot as plt
from IPython.display import Image
import lib

tz = timezone('US/Eastern')  # To monitor training time (showing start & end points of a fixed timezone when the code runs on a remote server)

#checkpoint = "/mnt/wd_4tb/shared_disk_wd4tb/ahmad/Generating-and-designing-DNA/scripts/log/gan_test/2020.02.10-23h52m43s_kepler/checkpoints/checkpoint_1000"
checkpoint = None

args = easydict.EasyDict({
        "generic": True,             # Generate generic data on the fly (ignores data_loc and data_start args)
        "pseudo_data": "pattern1",    # If 'generic=True', how to simulate sequence of nucllitides ('random', 'pattern1' or 'pattern2')
        "motif": False,                # If 'generic=True', if we want to have a motif also occuring in the simulated sequence
        "motif_match": "GTATT[AC]A",  # If 'motif=True', what would be the motif pattern accurning in each nth position (motif_nth_position) of a simulated sequence
        "motif_nth_position": 10,     # If 'motif=True', motif pattern tend to occur in each nth position (motif_nth_position) of a simulated sequence
        "motif_max": 2,               # If 'motif=True', maximum number of motifs in a simulated sequence
        "data_loc": "dataset/",       # Data location (create if it does not exist)
        "dataset_name": "chr1.fa.gz", # Name of the dataset in 'data_loc' to be loaded and trained/evaluated on
        "data_start": 0,              # Line number to start when parsing data (useful for ignoring header)
        "log_dir": "log/",            # Base log folder (create if it does not exist)
        "log_name": "gan_test",       # Name to use when logging this model
        "checkpoint": checkpoint,     # Foldername of checkpoint to load (full directory)
        "model_type": "resnet1d",          # Which type of model architecture to use ('mlp', 'resnet1d', 'resnet2d', etc)
        "train_iters": 4000,          # Number of iterations to train GAN for
        "disc_iters": 1,              # Number of iterations to train discriminator for at each training step
        "checkpoint_iters": 250,      # Number of iterations before saving checkpoint
        "latent_dim": 32,             # Size of latent space
        "gen_dim": 64,                # Size of each hidden layer in Generator
        "disc_dim": 64,               # Size of each hidden layer in Discriminator
        "gen_layers": 1,              # How many (middle or hidden) layers in Generator (ie. 'mlp': w/o 1st & last; 'resnet's: num. resudual blocks)
        "disc_layers": 1,             # How many (middle or hidden) layers in Discriminator (ie. 'mlp':  w/o 1st & last; 'resnet's: num. resudual blocks)
        "batch_size": 64,             # Batch size
        "max_seq_len": 36,            # Maximum sequence length of data
        "vocab": "dna_nt_only",               # Which vocabulary to use. Options are 'dna', 'rna', 'dna_nt_only', and 'rna_nt_only'.
        "vocab_order": None,          # Specific order for the one-hot encodings of vocab characters
        "annotate": False,            # Include annotation as part of training/generation process?
        "validate": True,             # Whether to use validation set
        "learning_rate": 0.004,       # Learning rate for the optimizer
        "beta1": 0.5,                 # 'beta1' for the optimizer
        "lmbda": 10,                  # 'lmbda' for Lipschitz gradient penalty hyperparameter
        "reproducible": True,         # Do we want to have a reproducible result?
        "seed": 1234,                 # Seed used for Random Number Generation if 'reproducible=True'
})

# sets device for model and PyTorch tensors
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#os.environ["CUDA_VISIBLE_DEVICES"]="1"

# set RNG
if args.reproducible:
    seed = args.seed
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device.type=='cuda': torch.cuda.manual_seed_all(seed)

# fix vocabulary of model
charmap, rev_charmap = lib.get_vocab(args.vocab, args.vocab_order)
vocab_size = len(charmap)

I = np.eye(vocab_size)

# organize model logs/checkpoints
logdir, checkpoint_baseline = lib.log(args, samples_dir=True)

if args.annotate:
    data_enc_dim = vocab_size + 1
else:
    data_enc_dim = vocab_size
data_size = args.max_seq_len * data_enc_dim

# build GAN
if args.model_type=="mlp":
    netG = lib.models.mlp_generator(args.gen_dim, args.latent_dim, args.max_seq_len*vocab_size, args.gen_layers).to(device)
    netD = lib.models.mlp_discriminator(args.disc_dim, args.max_seq_len*vocab_size, args.disc_layers).to(device)
elif args.model_type=="resnet1d":
    netG = lib.models.resnet_generator_1d(args.gen_dim, args.latent_dim, vocab_size, args.max_seq_len, args.gen_layers).to(device)
    netD = lib.models.resnet_discriminator_1d(args.disc_dim, vocab_size, args.max_seq_len, args.disc_layers).to(device)
elif args.model_type=="resnet2d":
    netG = lib.models.resnet_generator_2d(args.gen_dim, args.latent_dim, vocab_size, args.max_seq_len, args.gen_layers).to(device)
    netD = lib.models.resnet_discriminator_2d(args.disc_dim, vocab_size, args.max_seq_len, args.disc_layers).to(device)

print("{}\nGenerator Network Architecture:\n{}\n{}\n".format("="*31, "~"*31, netG))
print("{}\nDiscriminator Network Architecture:\n{}\n{}\n".format("="*35, "~"*35, netD))

# load checkpoint
if args.checkpoint:
    ckpt_dir = os.path.join(logdir, "checkpoints", "checkpoint_{}".format(checkpoint_baseline))
    netG.load_state_dict(torch.load('{}/netG'.format(ckpt_dir)))
    netD.load_state_dict(torch.load('{}/netD'.format(ckpt_dir)))
    
    for param in netG.parameters():
        param.requires_grad = True
                                      
    for i in range(checkpoint_baseline):
        schedulerG.step()
        schedulerD.step()
    print(schedulerG.get_lr())
        
# cost function (for discriminator prediction)
criterion = nn.BCEWithLogitsLoss()

# setup optimizer
optimizerG = optim.Adam(list(netG.parameters()), lr=args.learning_rate, betas=(args.beta1, 0.999))
optimizerD = optim.Adam(list(netD.parameters()), lr=args.learning_rate, betas=(args.beta1, 0.999))

# use an exponentially decaying learning rate
schedulerG = optim.lr_scheduler.ExponentialLR(optimizerG, gamma=0.99)
schedulerD= optim.lr_scheduler.ExponentialLR(optimizerD, gamma=0.99)

# load dataset
if args.generic:
    if args.annotate: raise Exception("args `annotate` and `generic` are incompatible.")

    def pseudo_sequence(batch_size, seq_len, vocab_size, pseudo_data="random", model_type="resnet2d", data_len=None):
        """Genertor to generating random sequences for toy test of gnerative model"""
        if data_len:
            assert batch_size <= data_len, "Make sure 'data_len' (num of samples) is not smaller than 'batch_size'"
            feed_ctr = 0
        I = np.eye(vocab_size)
        while True:
            samples = np.random.choice(vocab_size, [batch_size, seq_len])
            if pseudo_data=="pattern1" or pseudo_data=="pattern2":
                temp = np.random.choice(vocab_size, batch_size)+1
                for i, j in enumerate(temp):
                    for j in range(seq_len):
                        samples[i, j]= temp[i]
                        if pseudo_data=="pattern1":
                            temp[i] = temp[i]%vocab_size+1
                        elif pseudo_data=="pattern2":
                            if np.random.uniform()<temp[i]/vocab_size:
                                temp[i] = temp[i]%vocab_size+1
                    samples[i] = samples[i]-1
            
            
            if args.motif:
                # First, unwrapping possible motifs
                assert set(args.motif_match)-set(["[","]"]) <= set(charmap), "Make sure all the characters of 'motif_match' is allowed in 'vocab'"                   
                list_mtfs = []
                list_temp = []
                if "[" in args.motif_match:
                    list_temp.append(args.motif_match)
                else:
                    list_mtfs.append(args.motif_match)
                while 0<len(list_temp): # while list_temp has elements in which '[]' exist unwrap them
                    st = list_temp[0]
                    ntss = re.findall("\[(.*?)\]", st) # find elements in []
                    for nts in ntss:
                        for nt in nts:
                            ss = st.replace("["+nts+"]", nt)   # replace '[xyz]' with 'x', 'y', or, 'z' 
                            if "[" in ss and ss not in list_temp:  # not completely done done 
                                list_temp.append(ss)
                            else:        # completely done done 
                                if ss not in list_mtfs: list_mtfs.append(ss)
                    list_temp.remove(st)
                motifs = list_mtfs # list of possible motifs
                
                # Now, inserting motifs (randomly) in the sequence at 'motif_nth_position'
                for _ in range(args.motif_max):
                    motifchar = np.random.choice(motifs) # randomly chosing a possible motif
                    motif = [charmap[char] for char in motifchar]
                    for i in range(len(samples)):
                        rand = (np.random.choice(args.max_seq_len)//args.motif_nth_position)*args.motif_nth_position
                        if args.max_seq_len<rand+len(motif):
                            indx = rand+len(motif)-args.max_seq_len
                            samples[i][rand:rand+len(motif)-indx] = motif[0:len(motif)-indx]
                        else:
                            samples[i][rand:rand+len(motif)] = motif
                            
            data = np.vstack([np.expand_dims(I[vec],0) for vec in samples])
            
            if model_type=="mlp":
                reshaped_data = np.reshape(data, [batch_size, -1])
            elif model_type=="resnet1d" or model_type=="resnet2d":
                reshaped_data = np.transpose(data, (0, 2, 1))#np.reshape(data, [batch_size, vocab_size, args.max_seq_len])
            if data_len:
                feed_ctr += batch_size
                if feed_ctr > data_len:
                    feed_ctr = 0
                    yield None
                else:
                    yield reshaped_data
            else:
                yield reshaped_data

    train_seqs = pseudo_sequence(batch_size=args.batch_size, seq_len=args.max_seq_len, vocab_size=vocab_size, 
                                 pseudo_data=args.pseudo_data, model_type=args.model_type, data_len=None)
    
    if args.validate: 
        valid_seqs = pseudo_sequence(batch_size=args.batch_size, seq_len=args.max_seq_len, vocab_size=vocab_size, 
                                     pseudo_data=args.pseudo_data, model_type=args.model_type, data_len=10000)

else:
    data = load(args.data_loc, args.vocab, data_start_line=1, filenames=args.dataset_name, file_format='gz',
                valid=args.validate, annotate=args.annotate, model_type=args.model_type) # if nothing goes wrong (4609492, 50, 5)
    np.random.shuffle(data)
    if args.validate:
        split = len(data) // 2
        train_data = data[:split]
        valid_data = data[split:]
        if len(train_data) == 1: train_data = train_data[0]
        if len(valid_data) == 1: valid_data = valid_data[0]
    else:
        train_data = data
    if args.annotate:
        if args.validate: valid_data = np.concatenate(valid_data, 2)
        train_data = np.concatenate(train_data, 2)

    def feed(data, model_type, batch_size=args.batch_size, reuse=True):
        num_batches = len(data) // batch_size
        if model_type=="mlp":
            reshaped_data = np.reshape(data, [data.shape[0], -1])
        elif model_type=="resnet1d" or model_type=="resnet2d":
            reshaped_data = data
        while True:
            for ctr in range(num_batches):
                yield reshaped_data[ctr * batch_size : (ctr + 1) * batch_size]
            if not reuse and ctr == num_batches - 1:
                yield None
    train_seqs = feed(train_data, args.model_type)
    valid_seqs = feed(valid_data, args.model_type, reuse=False)


# train GAN
start_time = datetime.datetime.now(tz)
former_iteration_endpint = start_time
print("~~ TIME ~~")
print("Time started: {}".format(start_time.strftime("%Y-%m-%d %H:%M:%S")))
print("======================================================================")
print("Training GAN")
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
train_counts = []
valid_counts = []
disc_error_train = []
disc_error_valid = []
gen_error_train = []
for idx in range(args.train_iters):
    true_count = idx + 1 + checkpoint_baseline
    train_counts.append(true_count)
    
    ############################
    # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z))) + GP(disc_grad_penalty) wanting D(x)=1 & D(G(z))=0
    ############################
    optimizerG.zero_grad()
    optimizerD.zero_grad()
    
    errD_iters = 0
    for d in range(args.disc_iters):
        # 'Real' data
        real_seqs = next(train_seqs)
        real_seqs = Variable(torch.tensor(real_seqs).float().to(device))
        errD_real, score_real = lib.backprop(netD, real_seqs, disc_exp="real")
        
        # 'Fake' data
        noise = lib.z_sampler(args.batch_size, args.latent_dim)
        fake_seqs = netG(noise)
        errD_fake, score_fake = lib.backprop(netD, fake_seqs, disc_exp="fake")
        
        # gradient penalty error
        errD_GP = lib.disc_grad_penalty(netD, real_seqs, penalty_amount=args.lmbda)
        
        # Total error
        train_score = score_real + score_fake
        errD = errD_real + errD_fake + errD_GP
        
        optimizerD.step()
        
    disc_error_train.append(errD)
        
    ############################
    # (2) Update G network: maximize log(D(G(z))) wanting D(G(z))=1
    ############################
    optimizerG.zero_grad()
    optimizerD.zero_grad()
    
    noise = lib.z_sampler(args.batch_size, args.latent_dim)
    fake_seqs = netG(noise)
    errG, _ = lib.backprop(netD, fake_seqs, disc_exp="real")
    gen_error_train.append(errG)
    
    optimizerG.step()
    
    schedulerG.step()
    schedulerD.step()
    
    if true_count % args.checkpoint_iters == 0:
        # validation
        diff_vals = []
        err_vals = []
        data = next(valid_seqs)
        while data is not None:
            # How discriminator works on the real samples?
            real_seqs = Variable(torch.tensor(data).float().to(device))
            errD_fake, score_real = lib.backprop(netD, real_seqs, disc_exp="fake", backward=False)
            
            # How discriminator works on the fake samples?
            noise = lib.z_sampler(args.batch_size, args.latent_dim)
            fake_seqs = netG(noise)
            errD_real, score_fake = lib.backprop(netD, fake_seqs, disc_exp="real", backward=False)
            
            # How close the guesses are?
            valid_score = score_real - score_fake
            diff_vals.append((score_real+score_fake).cpu().item())
            err_vals.append(errD_real+errD_fake)
            data = next(valid_seqs)

        disc_error_valid.append(np.mean(err_vals))
        valid_counts.append(true_count)
        
        # log results and figures
        name = "valid_disc_cost"
        if checkpoint_baseline > 0: name += "_{}".format(checkpoint_baseline)
        lib.plot(valid_counts, disc_error_valid, logdir, name, xlabel="Iteration", ylabel="Discriminator cost")
        
        # Calculating 'Computation Time' for this round of iteration
        current_iteration_endpint = datetime.datetime.now(tz)
        current_iteration_elapsed = str(current_iteration_endpint - former_iteration_endpint).split(".")[0]
        temp = current_iteration_elapsed.split(":")
        if int(temp[0])==0 and int(temp[1])==0: current_iteration_elapsed = temp[2]
        elif int(temp[0])==0: current_iteration_elapsed = temp[1]+":"+temp[2]
        former_iteration_endpint = current_iteration_endpint
        
        print("Iteration {}/{}: train_diff={:.5f}, valid_diff={:.5f} ({}sec)".format(true_count, args.train_iters, train_score, valid_score, current_iteration_elapsed))
        fake_seqs = fake_seqs.reshape([-1, args.max_seq_len, data_enc_dim])
        lib.save_samples(logdir, fake_seqs.cpu().clone().detach().numpy(), true_count, rev_charmap, annotated=args.annotate)
        
        name = "train_disc_cost"
        if checkpoint_baseline > 0: name += "_{}".format(checkpoint_baseline)
        lib.plot(train_counts, disc_error_train, logdir, name, xlabel="Iteration", ylabel="Discriminator cost")
        
        name = "train_gen_cost"
        if checkpoint_baseline > 0: name += "_{}".format(checkpoint_baseline)
        lib.plot(train_counts, gen_error_train, logdir, name, xlabel="Iteration", ylabel="Generator cost")

    # save checkpoint
    if args.checkpoint_iters and true_count % args.checkpoint_iters == 0:
        lib.save_models(logdir, netG, netD, true_count)

print("======================================================================")
current_time = datetime.datetime.now(tz)
time_elapsed = current_time - start_time
print("Time current: {}".format(current_time.strftime("%Y-%m-%d %H:%M:%S")))
print("Time elapsed: {}".format(str(time_elapsed).split(".")[0]))