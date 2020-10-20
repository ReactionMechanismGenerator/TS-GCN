import math
from tqdm import tqdm
from typing import List, Union
from argparse import Namespace
import numpy as np
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import torch.nn as nn
from torch_geometric.utils import to_dense_adj

import os
import pymol
import tempfile
from rdkit import Chem, Geometry


def train(model, loader, optimizer, loss, device, scheduler, logger, writer, epoch):
    model.train()
    loss_all = 0

    for i, data in enumerate(tqdm(loader)):
        data = data.to(device)
        optimizer.zero_grad()

        out, mask = model(data)
        result = loss(out, to_dense_adj(data.edge_index, data.batch, data.y)) / mask.sum()
        result.backward()

        # clip the gradients
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=10, norm_type=2)

        if logger:
            pnorm = compute_pnorm(model)
            gnorm = compute_gnorm(model)
            logger.info(f'Parameter Norm: {pnorm}\t Gradient Norm: {gnorm}\t Loss: {result.item()}')

            # debugging
            writer.add_histogram('Vars/pred', model.pred.weight, epoch*len(loader) + i)
            writer.add_histogram('Grads/pred', model.pred.weight.grad, epoch * len(loader) + i)

            writer.add_histogram('Vars/edge_mlp_l1', model.edge_mlp.layers[0].weight, epoch * len(loader) + i)
            writer.add_histogram('Grads/edge_mlp_l1', model.edge_mlp.layers[0].weight.grad, epoch * len(loader) + i)

            writer.add_image('W', model.W[0].unsqueeze(0), epoch * len(loader) + i)
            writer.flush()

        optimizer.step()
        if isinstance(scheduler, NoamLR):
            scheduler.step()
        loss_all += result.item()

    return math.sqrt(loss_all / len(loader.dataset))  # rmse


def test(model, loader, loss, device):
    model.eval()
    error = 0

    for i, data in tqdm(enumerate(loader)):
        data = data.to(device)
        out, mask = model(data)
        result = loss(out, to_dense_adj(data.edge_index, data.batch, data.y)) / mask.sum()
        error += result.item()

        # if i==0:
        #     if epoch<5:
        #         check_ts(data, log_dir, epoch)

    # divides by number of molecules
    return math.sqrt(error / len(loader.dataset))  # rmse


class NoamLR(_LRScheduler):
    """
    Noam learning rate scheduler with piecewise linear increase and exponential decay.

    The learning rate increases linearly from init_lr to max_lr over the course of
    the first warmup_steps (where warmup_steps = warmup_epochs * steps_per_epoch).
    Then the learning rate decreases exponentially from max_lr to final_lr over the
    course of the remaining total_steps - warmup_steps (where total_steps =
    total_epochs * steps_per_epoch). This is roughly based on the learning rate
    schedule from Attention is All You Need, section 5.3 (https://arxiv.org/abs/1706.03762).
    """
    def __init__(self,
                 optimizer: Optimizer,
                 warmup_epochs: List[Union[float, int]],
                 total_epochs: List[int],
                 steps_per_epoch: int,
                 init_lr: List[float],
                 max_lr: List[float],
                 final_lr: List[float]):
        """
        Initializes the learning rate scheduler.

        :param optimizer: A PyTorch optimizer.
        :param warmup_epochs: The number of epochs during which to linearly increase the learning rate.
        :param total_epochs: The total number of epochs.
        :param steps_per_epoch: The number of steps (batches) per epoch.
        :param init_lr: The initial learning rate.
        :param max_lr: The maximum learning rate (achieved after warmup_epochs).
        :param final_lr: The final learning rate (achieved after total_epochs).
        """
        assert len(optimizer.param_groups) == len(warmup_epochs) == len(total_epochs) == len(init_lr) == \
               len(max_lr) == len(final_lr)

        self.num_lrs = len(optimizer.param_groups)

        self.optimizer = optimizer
        self.warmup_epochs = np.array(warmup_epochs)
        self.total_epochs = np.array(total_epochs)
        self.steps_per_epoch = steps_per_epoch
        self.init_lr = np.array(init_lr)
        self.max_lr = np.array(max_lr)
        self.final_lr = np.array(final_lr)

        self.current_step = 0
        self.lr = init_lr
        self.warmup_steps = (self.warmup_epochs * self.steps_per_epoch).astype(int)
        self.total_steps = self.total_epochs * self.steps_per_epoch
        self.linear_increment = (self.max_lr - self.init_lr) / self.warmup_steps

        self.exponential_gamma = (self.final_lr / self.max_lr) ** (1 / (self.total_steps - self.warmup_steps))

        super(NoamLR, self).__init__(optimizer)

    def get_lr(self) -> List[float]:
        """Gets a list of the current learning rates."""
        return list(self.lr)

    def step(self, current_step: int = None):
        """
        Updates the learning rate by taking a step.

        :param current_step: Optionally specify what step to set the learning rate to.
        If None, current_step = self.current_step + 1.
        """
        if current_step is not None:
            self.current_step = current_step
        else:
            self.current_step += 1

        for i in range(self.num_lrs):
            if self.current_step <= self.warmup_steps[i]:
                self.lr[i] = self.init_lr[i] + self.current_step * self.linear_increment[i]
            elif self.current_step <= self.total_steps[i]:
                self.lr[i] = self.max_lr[i] * (self.exponential_gamma[i] ** (self.current_step - self.warmup_steps[i]))
            else:  # theoretically this case should never be reached since training should stop at total_steps
                self.lr[i] = self.final_lr[i]

            self.optimizer.param_groups[i]['lr'] = self.lr[i]


def build_lr_scheduler(optimizer: Optimizer, args: Namespace, train_data_size: int) -> _LRScheduler:
    """
    Builds a learning rate scheduler.

    :param optimizer: The Optimizer whose learning rate will be scheduled.
    :param args: Arguments.
    :param train_data_size: The size of the training dataset.
    :return: An initialized learning rate scheduler.
    """
    # Learning rate scheduler
    return NoamLR(
        optimizer=optimizer,
        warmup_epochs=[args.warmup_epochs],
        total_epochs=[args.n_epochs],
        steps_per_epoch=train_data_size // args.batch_size,
        init_lr=[args.lr / 10],
        max_lr=[args.lr],
        final_lr=[args.lr / 100]
    )


def compute_pnorm(model: nn.Module) -> float:
    """Computes the norm of the parameters of a model."""
    return math.sqrt(sum([p.norm().item() ** 2 for p in model.parameters()]))


def compute_gnorm(model: nn.Module) -> float:
    """Computes the norm of the gradients of a model."""
    return math.sqrt(sum([p.grad.norm().item() ** 2 for p in model.parameters() if p.grad is not None]))


def render_pymol(mol, img_file, width=300, height=200):
    """ Render rdkit molecule with a pymol ball & stick representation """

    target = "all"
    pymol.cmd.delete(target)
    pymol.cmd.set("max_threads", 4)
    pymol.cmd.viewport(width, height)
    pymol.cmd.bg_color(color="white")

    # Save the file
    fd, temp_path = tempfile.mkstemp(suffix=".pdb")
    Chem.MolToPDBFile(mol, temp_path)
    pymol.cmd.load(temp_path)
    os.close(fd)

    # Representation
    pymol.cmd.color("gray40", target)
    pymol.util.cnc()
    pymol.cmd.show_as("spheres", target)
    pymol.cmd.show("licorice", target)
    pymol.cmd.set("sphere_scale", 0.25)
    pymol.cmd.orient()
    pymol.cmd.zoom(target, complete=1)
    pymol.cmd.png(img_file, ray=1)
    return


def check_ts(data, log_dir, epoch):
    """ Save examples of target and model predictions to file """

    n_check = data.batch.unique().size(0) // 10
    for i in range(n_check):

        target_ts = data.mols[i][1]
        predicted_ts = Chem.Mol(target_ts)

        for j in range(predicted_ts.GetNumAtoms()):
            x = data.coords[i][j].double().cpu().detach().numpy()
            predicted_ts.GetConformer().SetAtomPosition(j, Geometry.Point3D(x[0], x[1], x[2]))

        render_pymol(predicted_ts, os.path.join(log_dir, f'step{epoch}_ts{i}_model.png'), width=600, height=400)
        render_pymol(target_ts, os.path.join(log_dir, f'step{epoch}_ts{i}_target.png'), width=600, height=400)
