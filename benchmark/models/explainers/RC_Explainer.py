import torch
import torch.nn as nn

from torch_geometric.data import DataLoader, Data

import munch
import torch
from torch import Tensor
from torch_geometric.utils.loop import add_self_loops
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj

from .ExplainerBase import EdgeBase
from benchmark import data_args
from torch_geometric.data import Batch, Data

from torch.nn import Sequential, Linear, ReLU, ModuleList, Softmax, ELU, Sigmoid
import torch.nn.functional as F
from torch_geometric.utils import softmax
from torch_scatter import scatter_max
import os.path as osp
import copy


import torch.nn.functional as F
from torch.distributions import Bernoulli, Categorical
import numpy as np

from tqdm import tqdm
from torch_scatter import scatter_max

EPS = 1e-15

"""
Using the mini-batch training the RC-Explainer, which is much efficient than the one-by-one training.
"""

class RC_Explainer_Batch(EdgeBase):
    def __init__(self, model, epochs=30, lr=1e-4, explain_graph=False, molecule=False):
        super().__init__(model=model, epochs=epochs, lr=lr, explain_graph=explain_graph, molecule=molecule)

        self.model = model
        self.model = self.model.to(self.device)
        self.model.eval()


        self.temperature = 0.1

        self.edge_action_rep_generator = Sequential(
            Linear(data_args.dim_hidden * 2, data_args.dim_hidden * 4),
            ELU(),
            Linear(data_args.dim_hidden * 4, data_args.dim_hidden * 2),
            ELU(),
            Linear(data_args.dim_hidden * 2, data_args.dim_hidden)
        ).to(self.device)

        self.edge_action_prob_generator = self.build_edge_action_prob_generator()

    def build_edge_action_prob_generator(self):
        edge_action_prob_generator = Sequential(
            Linear(data_args.dim_hidden, data_args.dim_hidden),
            ELU(),
            Linear(data_args.dim_hidden, data_args.num_classes)
        ).to(self.device)
        return edge_action_prob_generator



    def policy(self, graph, state, train_flag=False):
        ocp_edge_index = graph.edge_index.T[state].T
        ocp_edge_attr = graph.edge_attr[state]

        ava_edge_index = graph.edge_index.T[~state].T
        ava_edge_attr = graph.edge_attr[~state]

        ava_node_reps_0 = self.model.get_node_reps(graph.x, graph.edge_index, graph.edge_attr, graph.batch)
        ava_node_reps_1 = self.model.get_node_reps(graph.x, ocp_edge_index, ocp_edge_attr, graph.batch)
        ava_node_reps = ava_node_reps_0 - ava_node_reps_1

        ava_action_reps = torch.cat([ava_node_reps[ava_edge_index[0]],
                                     ava_node_reps[ava_edge_index[1]]], dim=1).to(self.device)
        ava_action_reps = self.edge_action_rep_generator(ava_action_reps)

        ava_action_batch = graph.batch[ava_edge_index[0]]
        ava_y_batch = graph.y[ava_action_batch]

        if self.use_edge_attr:
            ava_edge_reps = self.model.edge_emb(ava_edge_attr)
            ava_action_reps = torch.cat([ava_action_reps, ava_edge_reps], dim=1)

        ava_action_probs = self.predict(ava_action_reps, ava_y_batch, ava_action_batch)

        added_action_probs, added_actions = scatter_max(ava_action_probs, ava_action_batch)

        if train_flag:
            rand_action_probs = torch.rand(ava_action_probs.size()).to(self.device)
            _, rand_actions = scatter_max(rand_action_probs, ava_action_batch)

            return ava_action_probs, ava_action_probs[rand_actions], rand_actions

        return ava_action_probs, added_action_probs, added_actions

    def predict(self, ava_action_reps, target_y, ava_action_batch):
        action_probs = self.edge_action_prob_generator(ava_action_reps)
        action_probs = action_probs.gather(1, target_y.view(-1,1))
        action_probs = action_probs.reshape(-1)

        action_probs = softmax(action_probs, ava_action_batch)
        return action_probs

    def get_optimizer(self, lr=0.01, weight_decay=1e-5, scope='all'):
        if scope in ['all']:
            params = self.parameters()
        else:
            params = list(self.edge_action_rep_generator.parameters()) + \
                     list(self.edge_action_prob_generator.parameters())

        optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
        return optimizer

    def load_policy_net(self, name='policy.pt', path=None):
        if not path:
            path = osp.join(osp.dirname(__file__), '..', '..', 'params', name)
        self.load_state_dict(torch.load(path, map_location=self.device))

    def save_policy_net(self, name='policy.pt', path=None):
        if not path:
            path = osp.join(osp.dirname(__file__), '..', '..', 'params', name)
        torch.save(self.state_dict(), path)



class RC_Explainer_Batch_star(RC_Explainer_Batch):
    def __init__(self, model, epochs=30, lr=1e-4, explain_graph=False, molecule=False):
        super().__init__(model=model, epochs=epochs, lr=lr, explain_graph=explain_graph, molecule=molecule)

    def build_edge_action_prob_generator(self):
        edge_action_prob_generator = nn.ModuleList()
        for i in range(data_args.num_classes):
            i_explainer = Sequential(
                Linear(data_args.dim_hidden * (2), data_args.dim_hidden * 2),
                ELU(),
                Linear(data_args.dim_hidden * 2, data_args.dim_hidden),
                ELU(),
                Linear(data_args.dim_hidden, 1)
            ).to(self.device)
            edge_action_prob_generator.append(i_explainer)

        return edge_action_prob_generator

    def forward(self, x: Tensor, edge_index: Tensor, **kwargs):
        """
        Given a sample, this function will return its predicted masks and corresponding predictions
        for evaluation
        :param x: Tensor - Hiden features of all vertexes
        :param edge_index: Tensor - All connected edge between vertexes/nodes
        :param kwargs:
        :return:
        """
        self.model.eval()
        super().forward(x, edge_index)

        labels = tuple(i for i in range(data_args.num_classes))
        ex_labels = tuple(torch.tensor([label]).to(data_args.device) for label in labels)

        self_loop_edge_index, _ = add_self_loops(edge_index, num_nodes=self.num_nodes)

        # --- setting end ---

        print('#D#Mask Calculate...')
        masks = []
        for ex_label in ex_labels:
            att = self.gen_explanation(Data(x=x, edge_index=edge_index, y=ex_label,
                                            batch=torch.zeros(x.shape[0], dtype=int, device=self.device)),
                                       kwargs.get('sparsity'))
            mask = att.float().to(self.device)
            if mask.shape.__len__() == 0:
                mask = mask.unsqueeze(0)
            mask = self.control_sparsity(mask, kwargs.get('sparsity'))
            masks.append(mask.detach())

        # Store related predictions for further evaluation.
        print('#D#Predict...')

        with torch.no_grad():
            with self.connect_mask(self):
                related_preds = self.eval_related_pred(x, edge_index, masks, **kwargs)

        return None, masks, related_preds

    def policy(self, graph, state, train_flag=False):

        graph_rep = self.model.get_graph_rep(data=graph)

        if len(torch.where(state==True)[0]) == 0:
            subgraph_rep = torch.zeros(graph_rep.size()).to(self.device)
        else:
            subgraph = relabel_graph(graph, state)
            subgraph_rep = self.model.get_graph_rep(data=subgraph, batch_size=graph.batch.max().item() + 1)

        # ava: available
        ava_edge_index = graph.edge_index.T[~state].T
        ava_node_reps, _ = self.model.get_emb(data=Data(x=graph.x, edge_index=ava_edge_index, batch=graph.batch))

        ava_action_reps = torch.cat([ava_node_reps[ava_edge_index[0]],
                                     ava_node_reps[ava_edge_index[1]]], dim=1).to(self.device)

        ava_action_reps = self.edge_action_rep_generator(ava_action_reps)

        ava_action_batch = graph.batch[ava_edge_index[0]]
        ava_y_batch = graph.y[ava_action_batch].long()

        # get the unique elements in batch, in cases where some batches are out of actions.
        unique_batch, ava_action_batch = torch.unique(ava_action_batch, return_inverse=True)

        ava_action_probs = self.predict_star(graph_rep, subgraph_rep, ava_action_reps, ava_y_batch, ava_action_batch)

        # assert len(ava_action_probs) == sum(~state)

        added_action_probs, added_actions = scatter_max(ava_action_probs, ava_action_batch)

        if train_flag:
            rand_action_probs = torch.rand(ava_action_probs.size()).to(self.device)
            _, rand_actions = scatter_max(rand_action_probs, ava_action_batch)

            return ava_action_probs, ava_action_probs[rand_actions], rand_actions, unique_batch

        return ava_action_probs, added_action_probs, added_actions, unique_batch

    def predict_star(self, graph_rep, subgraph_rep, ava_action_reps, target_y, ava_action_batch):
        action_graph_reps = graph_rep - subgraph_rep
        # get the corresponding graph's representation for each edge. Each edge is an action.
        action_graph_reps = action_graph_reps[ava_action_batch]
        action_graph_reps = torch.cat([ava_action_reps, action_graph_reps], dim=1)

        action_probs = []
        for i_explainer in self.edge_action_prob_generator:
            i_action_probs = i_explainer(action_graph_reps)
            action_probs.append(i_action_probs)
        action_probs = torch.cat(action_probs, dim=1)

        action_probs = action_probs.gather(1, target_y.view(-1,1))
        action_probs = action_probs.reshape(-1)

        # action_probs = softmax(action_probs, ava_action_batch)
        # action_probs = F.sigmoid(action_probs)
        return action_probs

    def train_policy(self, train_loader, test_loader,
                     topK_ratio=0.1, debias_flag=False, topN=None, reward_mode='mutual_info',
                     save_model_path=None):
        optimizer = self.get_optimizer(lr=self.lr, weight_decay=1e-4, scope='part')
        num_episodes = self.epochs

        # best_acc_auc, best_acc_curve, best_pre, best_rec = test_policy_all_with_gnd(rc_explainer, model, test_loader, topN)
        best_acc_auc, best_acc_curve, best_pre, best_rec = 0, 0, 0, 0
        ep = 0

        # baseline_reward_list = []

        previous_baseline_list = []
        current_baseline_list = []
        while ep < num_episodes:
            self.train()
            self.model.eval()

            loss = 0.
            avg_reward = []

            # if topK_ratio < 1. and ep != 0 and ep % 5 == 0:
            #     topK_ratio = min(0.5, topK_ratio * 1.25)

            for graph in tqdm(iter(train_loader), total=len(train_loader)):
                graph = graph.to(self.device)
                graph = Batch.from_data_list([graph[i] for i in range(graph.batch[-1] + 1) if graph[i].y == graph[i].y])

                if topK_ratio < 1:
                    valid_budget = max(int(topK_ratio * graph.num_edges / (graph.batch[-1] + 1)), 1)
                else:
                    valid_budget = topK_ratio

                batch_loss = 0.

                full_subgraph_pred = F.softmax(self.model(data=graph)).detach()

                current_state = torch.zeros(graph.num_edges, dtype=torch.bool)

                if debias_flag:
                    pred_bias_list = bias_detector(self.model, graph, valid_budget)

                pre_reward = torch.zeros(graph.y.size()).to(self.device)
                # pre_reward = 0.
                num_beam = 8
                for budget in range(valid_budget):
                    available_action = current_state[~current_state].clone()
                    new_state = current_state.clone()

                    beam_reward_list = []
                    beam_action_list = []
                    beam_action_probs_list = []

                    for beam in range(num_beam):
                        beam_available_action = current_state[~current_state].clone()
                        beam_new_state = current_state.clone()
                        if beam == 0:
                            _, added_action_probs, added_actions, unique_batch = self.policy(graph, current_state,
                                                                                      train_flag=False)
                        else:
                            _, added_action_probs, added_actions, unique_batch = self.policy(graph, current_state,
                                                                                      train_flag=True)

                        beam_available_action[added_actions] = True

                        beam_new_state[~current_state] = beam_available_action

                        new_subgraph = relabel_graph(graph, beam_new_state)
                        new_subgraph_pred = self.model(data=new_subgraph, batch_size=graph.batch.max().item() + 1)

                        if debias_flag:
                            new_subgraph_pred = F.softmax(new_subgraph_pred - pred_bias_list[budget]).detach()
                        else:
                            new_subgraph_pred = F.softmax(new_subgraph_pred).detach()

                        reward = get_reward(full_subgraph_pred, new_subgraph_pred, graph.y,
                                            pre_reward=pre_reward, mode=reward_mode)
                        reward = reward[unique_batch]

                        # ---------------

                        if len(previous_baseline_list) - 1 < budget:
                            baseline_reward = 0.
                        else:
                            baseline_reward = previous_baseline_list[budget]

                        if len(current_baseline_list) - 1 < budget:
                            current_baseline_list.append([torch.mean(reward)])
                        else:
                            current_baseline_list[budget].append(torch.mean(reward))

                        reward -= baseline_reward

                        # if len(baseline_reward_list) - 1 < budget:
                        #     baseline_reward = 0.
                        #     baseline_reward_list.append(0.)
                        # else:
                        #     baseline_reward = baseline_reward_list[budget]
                        #
                        # reward -= baseline_reward
                        #
                        # update_baseline_reward = (baseline_reward + torch.mean(reward))/2
                        # baseline_reward_list[budget] = update_baseline_reward
                        # ---------------

                        # batch_loss += torch.mean(- torch.log(added_action_probs + EPS) * reward)
                        avg_reward += reward.tolist()

                        beam_reward_list.append(reward)
                        beam_action_list.append(added_actions)
                        beam_action_probs_list.append(added_action_probs)

                    beam_reward_list = torch.stack(beam_reward_list).T
                    beam_action_list = torch.stack(beam_action_list).T
                    beam_action_probs_list = torch.stack(beam_action_probs_list).T

                    beam_action_probs_list = F.softmax(beam_action_probs_list, dim=1)
                    batch_loss += torch.mean(- torch.log(beam_action_probs_list + EPS) * beam_reward_list)

                    max_reward, max_reward_idx = torch.max(beam_reward_list, dim=1)
                    max_actions = beam_action_list[range(beam_action_list.size()[0]), max_reward_idx]

                    available_action[max_actions] = True
                    new_state[~current_state] = available_action

                    current_state = new_state.clone()
                    pre_reward[unique_batch] = max_reward

                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

                loss += batch_loss

            avg_reward = torch.mean(torch.FloatTensor(avg_reward))
            last_ep_reward = avg_reward

            ep += 1
            print('Episode: %d, loss: %.4f, average rewards: %.4f' % (ep, loss.detach(), avg_reward.detach()))

            ep_acc_auc, ep_acc_curve, ep_pre, ep_rec = self.test_policy_all_with_gnd(test_loader, topN)

            if ep_acc_auc >= best_acc_auc:
                best_acc_auc = ep_acc_auc
                best_acc_curve = ep_acc_curve
                best_pre = ep_pre
                best_rec = ep_rec

                self.save_policy_net(path=save_model_path)

            self.train()

            previous_baseline_list = [torch.mean(torch.stack(cur_baseline)) for cur_baseline in current_baseline_list]
            current_baseline_list = []

        return self, best_acc_auc, best_acc_curve, best_pre, best_rec

    def test_policy_all_with_gnd(self, test_loader, topN=None):
        self.eval()

        topK_ratio_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        acc_count_list = np.zeros(len(topK_ratio_list))

        precision_topN_count = 0.
        recall_topN_count = 0.

        len_test_loader = min(len(test_loader), 200)
        valid_len = 0

        with torch.no_grad():
            for i, graph in tqdm(enumerate(test_loader)):
                if i >= len_test_loader:
                    break
                if graph.y == graph.y and graph.num_edges > 0:
                    valid_len += 1
                else:
                    continue
                graph = graph.to(self.device)
                max_budget = graph.num_edges
                state = torch.zeros(max_budget, dtype=torch.bool)

                check_budget_list = [max(int(_topK * max_budget), 1) for _topK in topK_ratio_list]
                valid_budget = max(int(0.9 * max_budget), 1)

                for budget in range(valid_budget):
                    available_actions = state[~state].clone()

                    _, _, make_action_id, _ = self.policy(graph=graph, state=state, train_flag=False)

                    available_actions[make_action_id] = True
                    state[~state] = available_actions.clone()

                    if (budget + 1) in check_budget_list:
                        check_idx = check_budget_list.index(budget + 1)
                        subgraph = relabel_graph(graph, state)
                        subgraph_pred = self.model(data=subgraph, batch_size=graph.batch.max().item() + 1)

                        acc_count_list[check_idx] += sum(graph.y == subgraph_pred.argmax(dim=1))

                    if topN is not None and budget == topN - 1:
                        precision_topN_count += torch.sum(state * graph.ground_truth_mask[0]) / topN
                        recall_topN_count += torch.sum(state * graph.ground_truth_mask[0]) / sum(
                            graph.ground_truth_mask[0])

        acc_count_list[-1] = valid_len
        acc_count_list = np.array(acc_count_list) / valid_len

        precision_topN_count = precision_topN_count / valid_len
        recall_topN_count = recall_topN_count / valid_len

        if topN is not None:
            print('\nACC-AUC: %.4f, Precision@5: %.4f, Recall@5: %.4f' %
                  (acc_count_list.mean(), precision_topN_count, recall_topN_count))
        else:
            print('\nACC-AUC: %.4f' % acc_count_list.mean())
        print(acc_count_list)

        return acc_count_list.mean(), acc_count_list, precision_topN_count, recall_topN_count

    def gen_explanation(self, graph, sparsity):
        self.eval()

        graph = graph.to(self.device)
        max_budget = graph.num_edges
        state = torch.zeros(max_budget, dtype=torch.bool)
        valid_budget = max(int((1 - sparsity) * max_budget), 1)

        for budget in range(valid_budget):
            available_actions = state[~state].clone()

            _, _, make_action_id, _ = self.policy(graph=graph, state=state, train_flag=False)

            available_actions[make_action_id] = True
            state[~state] = available_actions.clone()


        return state






def normalize_reward(reward_pool):
    reward_mean = torch.mean(reward_pool)
    reward_std = torch.std(reward_pool) + EPS
    reward_pool = (reward_pool - reward_mean) / reward_std
    return reward_pool


def bias_detector(model, graph, valid_budget):
    pred_bias_list = []

    for budget in range(valid_budget):
        num_repeat = 2

        i_pred_bias = 0.
        for i in range(num_repeat):
            bias_selection = torch.zeros(graph.num_edges, dtype=torch.bool)

            ava_action_batch = graph.batch[graph.edge_index[0]]
            ava_action_probs = torch.rand(ava_action_batch.size()).to(graph.device)
            _, added_actions = scatter_max(ava_action_probs, ava_action_batch)

            bias_selection[added_actions] = True
            bias_subgraph = relabel_graph(graph, bias_selection)
            bias_subgraph_pred = model(data=bias_subgraph).detach()

            i_pred_bias += bias_subgraph_pred / num_repeat

        pred_bias_list.append(i_pred_bias)

    return pred_bias_list


def get_reward(full_subgraph_pred, new_subgraph_pred, target_y, pre_reward, mode='mutual_info'):
    if mode in ['mutual_info']:
        reward = torch.sum(full_subgraph_pred * torch.log(new_subgraph_pred + EPS), dim=1)
        reward += 2 * (target_y == new_subgraph_pred.argmax(dim=1)).float() - 1.

    elif mode in ['binary']:
        reward = (target_y == new_subgraph_pred.argmax(dim=1)).float()
        reward = 2. * reward - 1.

    elif mode in ['cross_entropy']:
        reward = torch.log(new_subgraph_pred + EPS)[:, target_y]

    # reward += pre_reward
    reward += 0.97 * pre_reward

    return reward

def relabel_graph(graph, selection):
    subgraph = copy.deepcopy(graph)

    # retrieval properties of the explanatory subgraph
    # .... the edge_index.
    subgraph.edge_index = graph.edge_index.T[selection].T
    # .... the edge_attr.
    # subgraph.edge_attr = graph.edge_attr[selection]
    # .... the nodes.
    sub_nodes = torch.unique(subgraph.edge_index)
    # .... the node features.
    subgraph.x = graph.x[sub_nodes]
    subgraph.batch = graph.batch[sub_nodes]

    row, col = graph.edge_index
    # remapping the nodes in the explanatory subgraph to new ids.
    node_idx = row.new_full((graph.num_nodes,), -1)
    node_idx[sub_nodes] = torch.arange(sub_nodes.size(0), device=row.device)
    subgraph.edge_index = node_idx[subgraph.edge_index]

    return subgraph