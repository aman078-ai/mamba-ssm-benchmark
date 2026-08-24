"""Training script for Simplified SSM (State Space Model).
Trains a Mamba-style SSM on byte-level enwik8 data.
Demonstrates the O(n) scan complexity vs Transformer's O(n^2).

Usage:
    python train_ssm.py --d_model 64 --d_state 16 --epochs 3
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

from models.ssm_block import SSMBlock
from data.enwik8 import Enwik8