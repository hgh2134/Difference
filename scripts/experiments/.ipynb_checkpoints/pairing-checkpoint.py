import sys
import numpy as np
import torch
from sacred import Experiment
from sacred.observers import MongoObserver, FileStorageObserver
from ignite.engine import Events, create_supervised_evaluator, create_supervised_trainer