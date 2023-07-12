"""
FileName: script2.py
Description: 
Time: 2020/9/8 9:56
Project: GNN_benchmark
Author: Shurui Gui
"""

import time, os
from cilog import create_logger

logger = create_logger(name='log',
                       file=os.path.join('.', 'script2.log'),
                       use_color=True)


time.sleep(100)
logger.info('hello world2')

