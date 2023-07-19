"""
FileName: script1.py
Description: 
Time: 2020/9/8 9:56
Project: GNN_benchmark
Author: Shurui Gui
"""

import time, os, sys
from cilog import create_logger

create_logger(sub_print=True)

table = [['', 'node/graph classification', '\\# of class', 'synthetic/real'],
         ['BA-LRP', 'graph', '2', 'synthetic'],
         ['Clintox', 'graph', '2', 'real'],
         ['Tox21', 'graph', '2', 'real'],
         ['BA-shape', 'node', '4', 'synthetic']]
print(f'#T#!latex{table}')