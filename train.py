from argparse import ArgumentParser
import math
import torch
from tqdm import tqdm

from model.G2C import CombinedNN
from utils import create_logger, Standardizer


parser = ArgumentParser()

parser.add_argument('--log_dir', type=str)
parser.add_argument('--data_path', type=str)
parser.add_argument('--confs_dir', type=str)
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



def train(model, loader, optimizer, loss, stdzer, device):
    model.train()
    loss_all = 0

    for data in tqdm(loader):
        data = data.to(device)
        optimizer.zero_grad()

        out = model(data)
        result = loss(out, stdzer(data.y))
        result.backward()

        # clip the gradients
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=10, norm_type=2)

        optimizer.step()
        loss_all += loss(stdzer(out, rev=True), data.y)

    return math.sqrt(loss_all / len(loader.dataset))  # rmse

def test(model, loader, loss, stdzer, device):
    model.eval()
    error = 0

    for data in tqdm(loader):
        data = data.to(device)
        out = model(data)
        error += loss(stdzer(out, rev=True), data.y).item()


train_loader, val_loader = construct_loader(args)
mean = train_loader.dataset.mean
std = train_loader.dataset.std
stdzer = Standardizer(mean, std)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CombinedNN()().to(device)
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
#                                                        factor=0.7, patience=5,
#                                                        min_lr=0.00001)
#todo: Do we want to use a lr scheduler?
scheduler = build_lr_scheduler(optimizer, args, len(train_loader.dataset))
loss = torch.nn.MSELoss(reduction='sum')
# or use MAE loss
# todo access mask from data.batch attribute to reflect the previous implementation
# self.loss_distance_all = self.masks["D"] * tf.abs(D_model - D_target)
torch.nn.L1Loss(reduction='mean')  # MAE

best_val_loss = math.inf
best_epoch = 0

logger.info("Starting training...")
for epoch in range(1, args.n_epochs):
    train_loss = train(model, train_loader, optimizer, loss, stdzer, device)
    logger.info("Epoch {}: Training Loss {}".format(epoch, train_loss))

    val_loss = test(model, val_loader, loss, stdzer, device)
    logger.info("Epoch {}: Validation Loss {}".format(epoch, val_loss))
    scheduler.step()

    if val_loss <= best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch
        # torch.save(model.state_dict(), os.path.join(args.log_dir, 'best_model'))

logger.info("Best Validation Loss {} on Epoch {}".format(best_val_loss, best_epoch))
