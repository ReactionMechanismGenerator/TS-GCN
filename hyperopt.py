from argparse import ArgumentParser
import logging
import math
import os
import torch
import optuna

from model.G2C import G2C
from model.training import train, test, NoamLR
from utils import create_logger, dict_to_str, save_yaml_file, get_optimizer_and_scheduler
from features.featurization import construct_loader


def optimize(trial, args):

    setattr(args, 'hidden_dim', int(trial.suggest_categorical('d_model', [128, 256, 512])))
    setattr(args, 'depth', int(trial.suggest_discrete_uniform('n_enc', 2, 6, 1)))
    setattr(args, 'n_layers', int(trial.suggest_discrete_uniform('n_enc', 1, 3, 1)))
    setattr(args, 'lr', trial.suggest_loguniform('lr', 1e-5, 1e-2))
    setattr(args, 'batch_size', int(trial.suggest_categorical('batch_size', [16, 32, 64, 128])))

    setattr(args, 'log_dir', os.path.join(args.hyperopt_dir, str(trial._trial_id)))

    torch.manual_seed(0)
    train_logger = create_logger('train', args.log_dir)

    train_logger.info('Arguments are...')
    for arg in vars(args):
        train_logger.info(f'{arg}: {getattr(args, arg)}')

    # construct loader and set device
    train_loader, val_loader = construct_loader(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
        train_logger.info(f'Using {torch.cuda.device_count()} GPUs for training...')
        model = torch.nn.DataParallel(model)

    # get optimizer and scheduler
    optimizer, scheduler = get_optimizer_and_scheduler(args, model, len(train_loader.dataset))
    loss = torch.nn.MSELoss(reduction='sum')

    # record parameters
    train_logger.info(f'\nModel parameters are:\n{dict_to_str(model_parameters)}\n')
    save_yaml_file(os.path.join(args.log_dir, 'model_parameters.yml'), model_parameters)
    train_logger.info(f'Optimizer parameters are:\n{optimizer}\n')
    train_logger.info(f'Scheduler state dict is:')
    if scheduler:
        for key, value in scheduler.state_dict().items():
            train_logger.info(f'{key}: {value}')
        train_logger.info('')

    best_val_loss = math.inf
    best_epoch = 0

    model.to(device)
    train_logger.info("Starting training...")
    for epoch in range(1, args.n_epochs):
        train_loss = train(model, train_loader, optimizer, loss, device, scheduler, logger if args.verbose else None)
        train_logger.info("Epoch {}: Training Loss {}".format(epoch, train_loss))

        val_loss = test(model, val_loader, loss, device, args.log_dir, epoch)
        train_logger.info("Epoch {}: Validation Loss {}".format(epoch, val_loss))
        if scheduler and not isinstance(scheduler, NoamLR):
            scheduler.step(val_loss)

        if val_loss <= best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(args.log_dir, f'epoch_{epoch}_state_dict.pt'))
    train_logger.info("Best Validation Loss {} on Epoch {}".format(best_val_loss, best_epoch))

    train_logger.handlers = []
    return best_val_loss


if __name__ == '__main__':

    parser = ArgumentParser()

    parser.add_argument('--sdf_dir', type=str)
    parser.add_argument('--split_path', type=str)

    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--warmup_epochs', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=5)

    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--scheduler', type=str, default=None)
    parser.add_argument('--verbose', action='store_true', default=False)

    parser.add_argument('--hyperopt_dir', type=str,
                        help='Directory to save all results')
    parser.add_argument('--n_trials', type=int, default=25,
                        help='Number of hyperparameter choices to try')
    args = parser.parse_args()

    if not os.path.exists(args.hyperopt_dir):
        os.makedirs(args.hyperopt_dir)

    logger = logging.getLogger()

    logger.setLevel(logging.INFO)  # Setup the root logger.
    logger.addHandler(logging.FileHandler(os.path.join(args.hyperopt_dir, "hyperopt.log"), mode="w"))

    optuna.logging.enable_propagation()  # Propagate logs to the root logger.
    optuna.logging.disable_default_handler()  # Stop showing logs in sys.stderr.

    study = optuna.create_study()
    logger.info("Running optimization...")
    study.optimize(lambda trial: optimize(trial, args), n_trials=args.n_trials)
