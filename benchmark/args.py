"""
FileName: args.py
Description: All Hyper Arguments here.
Time: 2020/7/28 13:10
Project: GNN_benchmark
Author: Shurui Gui
"""

from tap import Tap
from typing_extensions import Literal
from typing import List, Tuple, Dict
from definitions import ROOT_DIR
import torch, os


class GeneralArgs(Tap):
    random_seed: int = 123              # fixed random seed for reproducibility
    task: Literal['train', 'test', 'explain', 'table'] = 'test' # running mode
    dataset_split: List[float] = [0.8, 0.1, 0.1]    # train_val_test split
    train_bs: int = 3000                 # batch size for training
    val_bs: int = 3000                   # batch size for validation
    test_bs: int = 3000                  # batch size for test
    x_bs: int = 1                        # batch size for explain
    dataset_name: str = 'Tox21'              # dataset
    model_name: str = 'GCN'  # specify model name
    explainer: str = 'GNNExplainer'
    dataset_type: Literal['nlp', 'mol'] = 'mol'  # dataset type
    model_level: Literal['node', 'line', 'graph'] = 'graph'  # model level
    task_type: Literal['bcs', 'mcs', 'reg-l1'] = 'bcs'      # task type: b/m classification or regression
    target_idx: int = -1  # choose one target from multi-target task
    email: bool = False                 # email you after process down please use mail_setting.json
    explain_idx: int = 0                # default explain_idx 0
    log_file: str = 'pipeline.log'      # log file, root_dir: ROOT_DIR/log
    device_idx: int = 0                 # GPU index

    def process_args(self) -> None:
        super().process_args()
        self.device = torch.device(f'cuda:{self.device_idx}' if torch.cuda.is_available() else 'cpu')





class TrainArgs(GeneralArgs):

    tr_ctn: bool = False                    # flag for training continue
    ctn_epoch: int = 0                      # start Epoch for continue training
    lr: float = 0.01                        # learning rate
    mile_stones: List[int] = [500]
    weight_decay: int = 5e-4                # weight decay
    epoch: int = 100                        # Epochs to stop training
    val_gap: int = 100                      # do validation after val_gap batches training
    ckpt_dir: str = None                    # checkpoint dir for saving ckpt files
    save_gap: int = 10                      # how long to save a epoch

    def process_args(self) -> None:
        super().process_args()
        if self.ckpt_dir == None:
            self.ckpt_dir = os.path.join(ROOT_DIR, 'checkpoints',
                                         self.dataset_name, self.model_name,
                                         str(self.target_idx))


class ValArgs(GeneralArgs):
    pass                      # batch size for validation


class TestArgs(GeneralArgs):
    test_ckpt: str = None                   # path of model checkpoint

    def process_args(self) -> None:
        super().process_args()
        if self.test_ckpt == None:
            self.test_ckpt = \
                os.path.join(ROOT_DIR, 'checkpoints', self.dataset_name,
                             self.model_name, str(self.target_idx),
                             f'{self.model_name}_best.ckpt')


class XArgs(TestArgs):
    vis: bool = False
    sparsity: float = 0.5
    walk: bool = False
    debug: bool = False
    nolabel: bool = False
    list_sample: bool = False
    save_fig: bool = False
    explain_pred_label: bool = True
    force_recalculate: bool = False
    block_table: bool = False
    num_explain: int = 100

    def process_args(self) -> None:
        super().process_args()
        self.table_format = [['GradCAM', 'DeepLIFT', 'GNNExplainer', 'PGExplainer', 'GNN_GI', 'GNN_LRP', 'FlowShap_orig', 'FlowShap_plus', 'FlowMask', 'PGMExplainer', 'SubgraphX', 'Gem'],
                              ['ba_shapes', 'ba_lrp', 'clintox', 'tox21', 'bbbp', 'bace', 'graph_sst2', 'ba_infe'],
                              ['0.1S', '0.2S', '0.3S', '0.4S', '0.5S', '0.6S', '0.7S', '0.8S', '0.9S']]

class FSArgs(XArgs):
    score_lr: float = 2e-5                  # score map's learning rate
    no_mask: bool = False                   # set true for original FlowShap
    fidelity_plus = False                   # set true for Fidelity+ orientation


class DataArgs(GeneralArgs):
    dim_node: int = 0                       # Default: invalid num
    dim_edge: int = 0                       # Default: invalid num
    num_targets: int = 0                        # Default: invalid num
    dim_hidden: int = 300                   # node hidden feature's dimension
    dim_ffn: int = 300                      # final linear layer dim

class GemArgs(XArgs):
    top_k: int = 40                         # num of generated ground-truth nodes
    force_regen: bool = False               # Force to regenerate the ground-truth
    force_retrain: bool = False             # Force to retrain the VAE



common_args = GeneralArgs().parse_args(known_only=True)
data_args = DataArgs().parse_args(known_only=True)
train_args = TrainArgs().parse_args(known_only=True)
val_args = ValArgs().parse_args(known_only=True)
test_args = TestArgs().parse_args(known_only=True)
x_args = XArgs().parse_args(known_only=True)
fs_args = FSArgs().parse_args(known_only=True)
gem_args = GemArgs().parse_args(known_only=True)

