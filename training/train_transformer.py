"""Training script for vanilla Transformer.
Trains a decoder-only GPT-style Transformer on byte-level enwik8 data.
Demonstrates the O(n^2) attention complexity vs the SSM's O(n).

Usage:
    python train_transformer.py --d_model 64 --n_layer 2 --n_head 4 --epochs 3
"""
import argparse
import math
import os
import sys

# Add project root to path so we can import models/data
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.vanilla_transformer import VanillaTransformer
from data.enwik8 import Enwik8