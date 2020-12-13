import os

_curr_dir = os.path.abspath(os.path.dirname(__file__))

ROOT_DIR = os.path.join(_curr_dir, os.pardir)
DATA_DIR = os.path.join(ROOT_DIR, 'data')
LOG_DIR = os.path.join(ROOT_DIR, 'runs')