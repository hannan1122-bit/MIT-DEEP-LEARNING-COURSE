import mitdeeplearning as mdl

import comet_ml

import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn

import os
import time
import functools

from IPython import display as ipythondisplay
from tqdm import tqdm
from scipy.io.wavfile import write

print(torch.__version__)      # should show 2.5.1+cu121
print(torch.version.cuda)     # should show 12.1
print(torch.cuda.is_available())  # should be True

# Download the dataset
songs = mdl.lab1.load_training_data()

# Print one of the songs to inspect it in greater detail!
example_song = songs[0]
print("\nExample song: ")
print(example_song)

# mdl.lab1.play_song(example_song)
