from argparse import ArgumentParser
from datetime import datetime
import math
import os
import torch

from model.G2C import G2C
from model.training import train, test, NoamLR, build_lr_scheduler
from utils import create_logger, plot_train_val_loss
from features.featurization import construct_loader


parser = ArgumentParser()

parser.add_argument('--log_dir', type=str)
parser.add_argument('--sdf_dir', type=str)
parser.add_argument('--split_path', type=str)

parser.add_argument('--n_epochs', type=int, default=100)
parser.add_argument('--warmup_epochs', type=int, default=2)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--mini_batch', type=int, default=4)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--num_workers', type=int, default=2)

parser.add_argument('--hidden_dim', type=int, default=300)
parser.add_argument('--depth', type=int, default=3)
parser.add_argument('--n_layers', type=int, default=2)

parser.add_argument('--atom_messages', action='store_true')
parser.add_argument('--use_cistrans_messages', action='store_true')

args = parser.parse_args()
log_file_name = datetime.today().isoformat() + '_train'
logger = create_logger(log_file_name, args.log_dir)

logger.info('Arguments are...')
for arg in vars(args):
    logger.info(f'{arg}: {getattr(args, arg)}')

train_loader, val_loader = construct_loader(args)
train_data_size = len(train_loader.dataset)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = G2C(train_loader.dataset.num_node_features, train_loader.dataset.num_edge_features,
            args.hidden_dim, args.depth, args.n_layers).to(device)
if torch.cuda.device_count() > 1:
    logger.info(f'Using {torch.cuda.device_count()} GPUs for training...')
    model = torch.nn.DataParallel(model)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                       factor=0.7, patience=5,
                                                       min_lr=0.00001)
scheduler = build_lr_scheduler(optimizer=optimizer, args=args, train_data_size=train_data_size)

# record parameters
logger.info(f'\nModel architecture is:\n{model}\n')
logger.info(f'Optimizer parameters are:\n{optimizer}\n')
logger.info(f'Scheduler state dict is:')
for key, value in scheduler.state_dict().items():
    logger.info(f'{key}: {value}')
logger.info('')

# alternative lr scheduler
# scheduler = build_lr_scheduler(optimizer, args, len(train_loader.dataset))
loss = torch.nn.MSELoss(reduction='sum')
# alternative loss: MAE
torch.nn.L1Loss(reduction='sum')  # MAE

best_val_loss = math.inf
best_epoch = 0

logger.info("Starting training...")
for epoch in range(1, args.n_epochs):
    train_loss = train(model, train_loader, optimizer, loss, device, scheduler)
    logger.info("Epoch {}: Training Loss {}".format(epoch, train_loss))

    val_loss = test(model, val_loader, loss, device, args.log_dir, epoch)
    logger.info("Epoch {}: Validation Loss {}".format(epoch, val_loss))
    if not isinstance(scheduler, NoamLR):
        scheduler.step(val_loss)

    if val_loss <= best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch
        # torch.save(model.state_dict(), os.path.join(args.log_dir, 'best_model'))

logger.info("Best Validation Loss {} on Epoch {}".format(best_val_loss, best_epoch))

log_file = os.path.join(args.log_dir, log_file_name + '.log')
plot_train_val_loss(log_file)