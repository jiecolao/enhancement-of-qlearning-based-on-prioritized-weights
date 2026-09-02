import tracemalloc
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

def print_memory_stats(label):
    current, peak = tracemalloc.get_traced_memory()
    print(f"{label} | Current: {current / (1024 * 1024):.2f} MB | Peak: {peak / (1024 * 1024):.2f} MB\n")

