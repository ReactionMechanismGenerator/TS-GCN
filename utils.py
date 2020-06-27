from datetime import datetime
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

sns.set_style('whitegrid', {'axes.edgecolor': '.2'})
sns.set('poster', rc={"xtick.bottom" : True, "ytick.left" : True,
                    'axes.edgecolor': '.2',
                    "font.weight" : 'bold',
                    "axes.titleweight": 'bold',
                    'axes.labelweight' : 'bold'})
sns.color_palette('husl')

class Standardizer:
    """Z-score standardization"""
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x, rev=False):
        if rev:
            return (x * self.std) + self.mean
        return (x - self.mean) / self.std


def create_logger(name: str, log_dir: str = None) -> logging.Logger:
    """
    Creates a logger with a stream handler and file handler.

    :param name: The name of the logger.
    :param log_dir: The directory in which to save the logs.
    :return: The logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Set logger
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    fh = logging.FileHandler(os.path.join(log_dir, name + '.log'))
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

    return logger

def plot_train_val_loss(log_file):
    """
    Plots the training and validation loss by parsing the log file.

    :param log_file: The path to the log file created during training.
    """
    train_loss = []
    val_loss = []
    with open(log_file) as f:
        lines = f.readlines()
        for line in lines:
            if ': Training Loss' in line:
                train_loss.append(float(line.split(' ')[-1].rstrip()))
            if ': Validation Loss' in line:
                val_loss.append(float(line.split(' ')[-1].rstrip()))

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.plot(np.arange(len(train_loss)), train_loss, label='Train Loss')
    ax.plot(np.arange(len(val_loss)), val_loss, label='Val Loss')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.legend()

    fig.savefig(os.path.join(os.path.dirname(log_file), datetime.today().isoformat() + '_train_val_loss.pdf'), bbox_inches='tight')
