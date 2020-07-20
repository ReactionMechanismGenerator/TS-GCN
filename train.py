from argparse import ArgumentParser
from datetime import datetime
import math
import os
import torch

from model.G2C import G2C
from model.training import train, test, NoamLR, build_lr_scheduler
from utils import create_logger, dict_to_str, plot_train_val_loss, save_yaml_file, get_optimizer_and_scheduler
from features.featurization import construct_loader


parser = ArgumentParser()

parser.add_argument('--log_dir', type=str)
parser.add_argument('--sdf_dir', type=str)
parser.add_argument('--split_path', type=str)

parser.add_argument('--n_epochs', type=int, default=100)
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

args = parser.parse_args()

log_dir = os.path.join(args.log_dir, datetime.today().isoformat())
log_file_name = 'train'
logger = create_logger(log_file_name, log_dir)
logger.info('Arguments are...')
for arg in vars(args):
    logger.info(f'{arg}: {getattr(args, arg)}')

# construct loader and set device
train_loader, val_loader = construct_loader(args)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# build model
model_parameters = {'node_dim': train_loader.dataset.num_node_features,
                    'edge_dim': train_loader.dataset.num_edge_features,
                    'hidden_dim': args.hidden_dim,
                    'depth': args.depth,
                    'n_layers': args.n_layers}
model = G2C(**model_parameters).to(device)

# multi gpu training
if torch.cuda.device_count() > 1:
    logger.info(f'Using {torch.cuda.device_count()} GPUs for training...')
    model = torch.nn.DataParallel(model)

# get optimizer and scheduler
optimizer, scheduler = get_optimizer_and_scheduler(args, model, len(train_loader.dataset))

# record parameters
logger.info(f'\nModel parameters are:\n{dict_to_str(model_parameters)}\n')
yaml_file_name = os.path.join(log_dir, 'model_paramaters.yml')
save_yaml_file(yaml_file_name, model_parameters)
logger.info(f'Optimizer parameters are:\n{optimizer}\n')
logger.info(f'Scheduler state dict is:')
if scheduler:
    for key, value in scheduler.state_dict().items():
        logger.info(f'{key}: {value}')
    logger.info('')

loss = torch.nn.MSELoss(reduction='sum')
# alternative loss: MAE
torch.nn.L1Loss(reduction='sum')  # MAE

best_val_loss = math.inf
best_epoch = 0

logger.info("Starting training...")
for epoch in range(1, args.n_epochs):
    train_loss = train(model, train_loader, optimizer, loss, device, scheduler)
    logger.info("Epoch {}: Training Loss {}".format(epoch, train_loss))

    val_loss = test(model, val_loader, loss, device, log_dir, epoch)
    logger.info("Epoch {}: Validation Loss {}".format(epoch, val_loss))
    if scheduler and not isinstance(scheduler, NoamLR):
        scheduler.step(val_loss)

    if val_loss <= best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch
        # torch.save(model.state_dict(), os.path.join(log_dir, 'best_model'))

logger.info("Best Validation Loss {} on Epoch {}".format(best_val_loss, best_epoch))

log_file = os.path.join(log_dir, log_file_name + '.log')
plot_train_val_loss(log_file)