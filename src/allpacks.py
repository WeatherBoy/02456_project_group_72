import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.functional import to_map_style_dataset
from torchtext import datasets

import GPUtil
import spacy

import copy
import time
import math
import os