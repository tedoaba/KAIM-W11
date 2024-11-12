import os
import sys
import numpy as np
import tensorflow as tf
import warnings

warnings.filterwarnings('ignore')

sys.path.append(os.path.abspath('..'))

from scripts.tasks import task_1, task_2, task_3, task_4

if __name__ == "__main__":
    task_1()
    task_2()
    task_3()
    task_4()