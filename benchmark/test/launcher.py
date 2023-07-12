"""
FileName: launcher.py
Description: 
Time: 2020/9/8 9:56
Project: GNN_benchmark
Author: Shurui Gui
"""
import subprocess, shlex, time, os, signal

def signal_process():
    # os.setsid()
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    # signal.signal(signal.SIGINT, signal.SIG_IGN)

conda_env = '/home/citrine/anaconda3/envs/torch_v1.5/bin/python'

args_group = [
    '-m benchmark.test.script1'
]

cmd_args_group = [' '.join([conda_env, args]).split(' ') for args in args_group]

print(cmd_args_group)

for cmd_args in cmd_args_group:
    process = subprocess.Popen(cmd_args, preexec_fn=signal_process, close_fds=True,
                               stdout=open('/dev/null'), stderr=subprocess.STDOUT)
    while 1:
        retruncode = process.poll()
        print(retruncode)
        if retruncode is not None:
            break
        time.sleep(1)



# time.sleep(5)
#
# pid = os.fork()
# if pid != 0:
#     time.sleep(5)
#     # os._exit(0)
#     pass
# else:
#     time.sleep(10)
#     print('I am a child.')
