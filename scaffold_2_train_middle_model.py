#!/usr/bin/env python
import argparse

import torch
from torch.utils.data import DataLoader
from rdkit import Chem
from rdkit import rdBase
from tqdm import tqdm
from utils.utils import seq_to_smiles
from utils.data_structs_distill_scaffold import MolData, Vocabulary
from models.model_rnn_scaffold import RNN
from utils.utils import decrease_learning_rate
import utils.scaffold_utils.scaffold as usc
rdBase.DisableLog('rdApp.error')


def train_middle(train_data_scaffold,train_data_decorator, save_model='./DM.ckpt'):
    """Trains the Prior RNN"""

    # Read vocabulary from a file
    voc = Vocabulary(init_from_file="data/Voc_RE3")

    # Create a Dataset from a SMILES file
    moldata = MolData(train_data_scaffold,train_data_decorator, voc)
    data = DataLoader(moldata, batch_size=2, shuffle=True, drop_last=True,
                      collate_fn=MolData.collate_fn)

    # Prior = RNN(voc)
    Prior = RNN(num_layers, d_model, voc, trans_dropout)

    optimizer = torch.optim.Adam(Prior.decorator.parameters(), lr=0.001)

    # Prior.decorator.to(device)
    # Prior.decorator.train()
    for epoch in range(1, 9):

        for step, batch in tqdm(enumerate(data), total=len(data)):

            # Sample from DataLoader
            scaffold_seqs = batch[0].long()
            decorator_seqs = batch[1].long()
            # Calculate loss
            loss, each_molecule_loss = Prior.likelihood(scaffold_seqs, decorator_seqs)
            # loss = - loss.mean()

            # Calculate gradients and take a step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Every 500 steps we decrease learning rate and print some information
            if step % 2 == 0 and step != 0:
                decrease_learning_rate(optimizer, decrease_by=0.03)
                tqdm.write("*" * 50)
                tqdm.write("Epoch {:3d}   step {:3d}    loss: {:5.2f}\n".format(epoch, step, loss.cpu().data))
                seqs = Prior.sample(batch_size,scaffold_seqs, max_length=140)
                valid = 0
                valid_decorator = 0
                scaffold_list=(seq_to_smiles(scaffold_seqs, voc))
                for i, seq in enumerate(seqs.cpu().numpy()):
                    smile = voc.decode(seq)
                    mol = usc.join_joined_attachments(scaffold_list[i], smile)

                    if Chem.MolFromSmiles(smile):
                        valid += 1
                    if mol:
                        valid_decorator += 1
                    if i < 5:
                        tqdm.write(smile)

                tqdm.write("\n{:>4.1f}% valid SMILES".format(100 * valid / len(seqs)))
                tqdm.write("\n{:>4.1f}% valid Scaffold+Decorator".format(100 * valid_decorator / len(seqs)))
                tqdm.write("*" * 50 + "\n")
                torch.save(Prior.decorator.state_dict(), save_model)

        # Save the Prior
        torch.save(Prior.decorator.state_dict(), save_model)


if __name__ == "__main__":
    d_model = 256  # 128
    num_layers = 3
    trans_dropout = 0
    batch_size=128

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser(description="Main script for running the model")
    parser.add_argument('--train-data-scaffold', action='store', dest='train_data_scaffold', default='scaffold_for_distill.csv', )
    parser.add_argument('--train-data-decorator', action='store', dest='train_data_decorator', default='decorator_for_distill.csv', )
    parser.add_argument('--save-middle-path', action='store', dest='save_model', default='./data/Prior_decorator_RNN',
                        help='Path and name of middle model.')

    arg_dict = vars(parser.parse_args())

    train_middle(**arg_dict)