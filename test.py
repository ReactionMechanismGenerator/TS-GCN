from argparse import ArgumentParser
import os
import pymol
from rdkit import Chem, Geometry
import tempfile
import torch
from tqdm import tqdm
import yaml

from features.featurization import construct_loader
from model.G2C import G2C
from model.training import render_pymol


parser = ArgumentParser()

parser.add_argument('--log_dir', type=str)
parser.add_argument('--sdf_dir', type=str)
parser.add_argument('--split_path', type=str)

parser.add_argument('--n_epochs', type=int, default=5)
parser.add_argument('--warmup_epochs', type=int, default=2)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--mini_batch', type=int, default=4)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--num_workers', type=int, default=2)

parser.add_argument('--hidden_dim', type=int, default=100)
parser.add_argument('--depth', type=int, default=3)
parser.add_argument('--n_layers', type=int, default=2)

parser.add_argument('--optimizer', type=str, default='adam')
parser.add_argument('--scheduler', type=str, default=None)
parser.add_argument('--verbose', action='store_true', default=False)

args = parser.parse_args()

# construct loader
train_loader, val_loader = construct_loader(args)

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# define paths
# note that these weights are for a model trained only on the training set 
yaml_file_name = 'best_model/model_paramaters.yml'
state_dict = 'best_model/epoch_95_state_dict'

# create the network with the best architecture from hyperopt and load the corresponding best weights
with open(yaml_file_name, 'r') as f:
    content = yaml.load(stream=f, Loader=yaml.FullLoader)
print(content)
model = G2C(**content).to(device)
model.load_state_dict(torch.load(state_dict, map_location=device))
model.eval()

k = 0
# for now, evaluate model's performance on the validation set only
for i, data in tqdm(enumerate(val_loader)):
	data = data.to(device)
	out, mask = model(data)
	n_check = data.batch.unique().size(0) // 15
	for i in range(n_check):
		target_ts = data.mols[i][1]
		predicted_ts = Chem.Mol(target_ts)

		for j in range(predicted_ts.GetNumAtoms()):
			x = data.coords[i][j].double().cpu().detach().numpy()
			predicted_ts.GetConformer().SetAtomPosition(j, Geometry.Point3D(x[0], x[1], x[2]))

		render_pymol(predicted_ts, os.path.join(os.getcwd(), 'best_model', f'ts{k}_model.png'), width=600, height=400)
		render_pymol(target_ts, os.path.join(os.getcwd(), 'best_model', f'ts{k}_target.png'), width=600, height=400)       
		k+=1
