#!/usr/bin/env python
import argparse

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os.path
import glob
import itertools as it
import scaffold.utils.chem as uc
from utils.data_structs_scaffold import MolData, Vocabulary
from models.model_scaffold import transformer_RL
from torch.optim import Adam
from utils.Optim import ScheduledOptim
from utils.early_stop.pytorchtools import EarlyStopping





def train_prior(train_data,valid_data,save_prior_path,training_set_path,validate_set_path):

    """Trains the Prior decodertf"""

    # Read vocabulary from a file
    voc = Vocabulary(init_from_file="data/Voc_RE1")

    # Create a Dataset from a SMILES file
    # moldata = MolData(train_data, voc)
    # valid = MolData(valid_data, voc)

    training_sets = load_sets(training_set_path)
    validate_sets = load_sets(validate_set_path)

    # train_data = DataLoader(moldata, batch_size=batch_size, shuffle=True, drop_last=True,
    #                   collate_fn=MolData.collate_fn)
    #
    # valid_data = DataLoader(valid, batch_size=batch_size, shuffle=True, drop_last=True,
    #                   collate_fn=MolData.collate_fn)

    Prior = transformer_RL(voc, d_model, nhead, num_decoder_layers,
                           dim_feedforward, max_seq_length,
                           pos_dropout, trans_dropout)


    optim = ScheduledOptim(
        Adam(Prior.decodertf.parameters(), betas=(0.9, 0.98), eps=1e-09),
        d_model * 8,n_warmup_steps)

    # train_losses, val_losses = train(Prior, optim, num_epochs,save_prior_path,training_sets)
    train_losses = train(voc, Prior, optim, num_epochs, save_prior_path, training_sets,validate_sets)

    # torch.cuda.empty_cache()

def load_sets(set_path):
    file_paths = [set_path]
    if os.path.isdir(set_path):
        file_paths = sorted(glob.glob("{}/*.csv".format(set_path)))
    return file_paths

    # for path in it.cycle(file_paths):  # stores the path instead of the set
    #     yield list(uc.read_csv_file(path, num_fields=2))

def initialize_dataloader(training_set,voc):
        dataset = MolData(training_set, voc)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True,
                              collate_fn=MolData.collate_fn, drop_last=True)

def train(voc,model, optim, num_epochs,save_prior_path,training_sets,validate_sets):

    model.decodertf.to(device)

    model.decodertf.train()
    lowest_val = 1e9
    train_losses = []
    val_losses = []
    total_step = 0

    early_stopping = EarlyStopping(patience=5, verbose=False)
    valid_data = initialize_dataloader(validate_sets[0], voc)
    for epoc in range(num_epochs):
        for epoch, training_set in zip(range(50), training_sets):
            train_data = initialize_dataloader(training_set,voc)
            # When training on a few million compounds, this model converges
            # in a few of epochs or even faster. If model sized is increased
            # its probably a good idea to check loss against an external set of
            # validation SMILES to make sure we dont overfit too much.

            total_loss = 0
            for step, batch in tqdm(enumerate(train_data), total=len(train_data)):

                # Sample from DataLoader
                scaffold_seqs = batch[0].long()
                decorator_seqs = batch[1].long()

                # graph = batch[1]

                # Calculate loss, each_molecule_loss is the loss of  each molecule

                loss, each_molecule_loss = model.likelihood(scaffold_seqs,decorator_seqs)
                # loss = - log_p.mean()

                # Calculate gradients and take a step
                optim.zero_grad()
                loss.backward()
                optim.step_and_update_lr()


                total_loss += loss.item()
                # train_losses.append((step, loss.item()))

                # if step % print_every == print_every - 1:


                if step % 200 == 0 and step != 0:
                    # decrease_learning_rate(optim, decrease_by=0.03)
                    tqdm.write("*" * 50)
                    tqdm.write("Epoch {:3d}   step {:3d}    loss: {:5.2f}\n".format(epoch, step, loss.data))

            print('average epoch loss:', total_loss / len(train_data))

            val_loss = validate(valid_data, model)
            val_losses.append((total_step, val_loss))

            early_stopping(val_loss, model.decodertf, 'RE1_Prior')

            if early_stopping.early_stop:
                print("Early stopping")
                break

            # Save the Prior
            if val_loss < lowest_val:
                lowest_val = val_loss
                torch.save(model.decodertf.state_dict(), save_prior_path)

            print(f'Val Loss: {val_loss}')

        if early_stopping.early_stop:
            print("Early stopping")
            break

    return  val_losses


def validate(valid_data, model):
    # pbar = tqdm(total=len(iter(valid_loader)), leave=False)
    model.decodertf.to(device)
    model.decodertf.eval()
    total_loss = 0

    for step, batch in tqdm(enumerate(valid_data), total=len(valid_data)):
        with torch.no_grad():
            # Sample from DataLoader
            scaffold_seqs = batch[0].long()
            decorator_seqs = batch[1].long()
            # graph = batch[1]

            # Calculate loss, each_molecule_loss is the loss of  each molecule
            loss, each_molecule_loss = model.likelihood(scaffold_seqs,decorator_seqs)
            # loss = - log_p.mean()

            total_loss += loss.item()
            # train_losses.append((step, loss.item()))
    return total_loss / len(valid_data)



if __name__ == "__main__":
    max_seq_length = 140
    # num_tokens=71
    # vocab_size=71
    d_model = 128
    # num_encoder_layers = 6
    num_decoder_layers = 12
    dim_feedforward = 512
    nhead = 8
    pos_dropout = 0.1
    trans_dropout = 0.1
    n_warmup_steps = 1

    num_epochs = 3
    batch_size = 5 #1024
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print(device)

    parser = argparse.ArgumentParser(description="Main script for running the model")
    parser.add_argument('--train-data', action='store', dest='train_data')
    parser.add_argument('--valid-data', action='store', dest='valid_data')

    parser.add_argument("--training-set-path", "-s", default='./data/scaffold/test',help="Path to a file with (scaffold, decoration) tuples \
        or a directory with many of these files to be used as training set.", type=str, required=False,dest='training_set_path')
    parser.add_argument("--validate-set-path", "-s_v", default='./data/scaffold/test_validate',help="Path to a file with (scaffold, decoration) tuples \
        or a directory with many of these files to be used as training set.", type=str, required=False,dest='validate_set_path')

    parser.add_argument('--save-prior-path', action='store', dest='save_prior_path',
                        default='./data/Prior_decorator.ckpt',
                        help='Path to save an checkpoint.')

    arg_dict = vars(parser.parse_args())

    train_prior(**arg_dict)
