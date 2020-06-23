from argparse import ArgumentParser
import math
import torch

from model.G2C import G2C
from model.training import train, test, NoamLR
from utils import create_logger
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

parser.add_argument('--atom_messages', action='store_true')
parser.add_argument('--use_cistrans_messages', action='store_true')

args = parser.parse_args()
logger = create_logger('train', args.log_dir)





train_loader, val_loader = construct_loader(args)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = G2C(train_loader.dataset.num_node_features, train_loader.dataset.num_edge_features, 100, 3, 2).to(device)
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                       factor=0.7, patience=5,
                                                       min_lr=0.00001)
# todo: Do we want to use a lr scheduler?
# scheduler = build_lr_scheduler(optimizer, args, len(train_loader.dataset))
loss = torch.nn.MSELoss(reduction='sum')
# or use MAE loss
# todo access mask from data.batch attribute to reflect the previous implementation
# self.loss_distance_all = self.masks["D"] * tf.abs(D_model - D_target)
torch.nn.L1Loss(reduction='mean')  # MAE

best_val_loss = math.inf
best_epoch = 0

logger.info("Starting training...")
for epoch in range(1, args.n_epochs):
    train_loss = train(model, train_loader, optimizer, loss, device, scheduler)
    logger.info("Epoch {}: Training Loss {}".format(epoch, train_loss))

    val_loss = test(model, val_loader, loss, device)
    logger.info("Epoch {}: Validation Loss {}".format(epoch, val_loss))
    if not isinstance(scheduler, NoamLR):
        scheduler.step()

    if val_loss <= best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch
        # torch.save(model.state_dict(), os.path.join(args.log_dir, 'best_model'))

logger.info("Best Validation Loss {} on Epoch {}".format(best_val_loss, best_epoch))
