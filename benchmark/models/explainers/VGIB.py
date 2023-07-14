import munch
import torch
from torch import Tensor
from torch_geometric.utils.loop import add_self_loops
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj

from .ExplainerBase import EdgeBase
from benchmark import data_args
from torch_geometric.data import Batch, Data

EPS = 1e-15

import itertools
import matplotlib.pyplot as plt

plt.switch_backend('agg')
from tqdm import tqdm



class VGIB(EdgeBase):
    """
    Semi-Supervised Graph Classification: A Hierarchical Graph Perspective Cautious Iteration model.
    """

    def __init__(self, model, epochs=100, lr=3e-1, explain_graph=False, molecule=False):
        """
        Creating dataset, doing dataset split, creating target and node index vectors.
        :param args: Arguments object.
        """
        super().__init__(model=model, epochs=epochs, lr=lr, explain_graph=explain_graph, molecule=molecule)
        self.args = munch.Munch(epochs=10, mi_weight=1e-4, con_weight=5, lr=1e-2)
        self.batch_size = 16

    def _setup_model(self):
        """
        Creating a SEAL model.
        """
        self.VGIB_model = Subgraph(self.model, self.args).to(self.device)

    def set_requires_grad(self, net, requires_grad=False):

        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

    def fit_a_single_model(self, dataset, use_pred_label=False):
        """
        Fitting a single SEAL model.

        """
        print("\nTraining started.\n")

        self._setup_model()

        optimizer = torch.optim.Adam(itertools.chain(self.VGIB_model.graph_level_model.fully_connected_1.parameters(),
                                                     self.VGIB_model.graph_level_model.fully_connected_2.parameters()),
                                     lr=self.args.lr,
                                     weight_decay=5e-5)
        # optimizer = torch.optim.Adam(self.VGIB_model.parameters(),
        #                              lr=self.args.lr,
        #                              weight_decay=5e-5)

        self.training_data = [data for data in dataset]
        Data_Length = len(self.training_data)
        Num_split = int(Data_Length / self.batch_size)

        best_mean = 1.0

        save_loss = []
        Iter = 0
        # mi_weight = 0

        for Epoch in tqdm(range(self.args.epochs)):

            for i in range(0, Num_split):
                Iter += 1

                data = Batch.from_data_list(self.training_data[int(i * self.batch_size): min(int((i + 1) * self.batch_size), Data_Length)]).to(self.device)

                if use_pred_label:
                    logits = self.model(data=data)
                    pred_label = logits.argmax(-1).data
                else:
                    pred_label = None

                embeddings, positive, noisy_embedding, mi_loss, cls_loss, positive_penalty, preserve_rate = self.VGIB_model(
                    data, pred_label=pred_label)

                loss = cls_loss + positive_penalty

                optimizer.zero_grad()

                loss = loss + self.args.mi_weight * mi_loss

                loss.backward()

                optimizer.step()

                print("MI_pen:%.2f,CLS:%.2f,Pen:%.2f,Pre:%.2f" % (
                    self.args.mi_weight * mi_loss, cls_loss, positive_penalty, preserve_rate))

                one_save_loss = str(self.args.mi_weight * mi_loss) + ' ' + str(cls_loss) + ' ' + str(
                    positive_penalty / self.args.con_weight) + '\n'

        #         save_loss.append(one_save_loss)
        #
        # save_loss_path = self.args.save + 'loss.txt'
        #
        # with open(save_loss_path, 'w') as F:
        #     F.writelines(save_loss)

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
        att = self.VGIB_model.graph_level_model.return_att(data=Data(x=x, edge_index=edge_index))
        att_positive = att[:, 0]
        mask = att_positive
        if mask.shape.__len__() == 0:
            mask = mask.unsqueeze(0)
        mask = (mask[self_loop_edge_index[0]] + mask[self_loop_edge_index[1]]) / 2
        mask = self.control_sparsity(mask, kwargs.get('sparsity'))
        masks = []
        for ex_label in ex_labels:
            masks.append(mask.detach())

        # Store related predictions for further evaluation.
        print('#D#Predict...')

        with torch.no_grad():
            with self.connect_mask(self):
                related_preds = self.eval_related_pred(x, edge_index, masks, **kwargs)

        return None, masks, related_preds


"""Convolutional layers."""



class SAGE(torch.nn.Module):
    """
    SAGE layer class.
    """

    def __init__(self, model):
        """
        Creating a SAGE layer.
        :param args: Arguments object.
        :param number_of_features: Number of node features.
        """
        super(SAGE, self).__init__()
        self.model = model
        self._setup()
        self.mseloss = torch.nn.MSELoss()
        self.relu = torch.nn.ReLU()

    def _setup(self):
        """
        Setting up upstream and pooling layers.
        """
        first_dense_neurons = 16
        second_dense_neurons = 2

        self.fully_connected_1 = torch.nn.Linear(data_args.dim_hidden,
                                                 first_dense_neurons)

        self.fully_connected_2 = torch.nn.Linear(first_dense_neurons,
                                                 second_dense_neurons)

    def forward(self, data):
        """
        Making a forward pass with the graph level data.
        :param data: Data feed dictionary.
        :return graph_embedding: Graph level embedding.
        :return penalty: Regularization loss.
        """
        # edges = data["edges"]
        edges = data.edge_index
        epsilon = 0.0000001

        # features = data["features"]
        # node_features_1 = torch.nn.functional.relu(self.graph_convolution_1(features, edges))
        # node_features_2 = self.graph_convolution_2(node_features_1, edges)
        node_feature, batch = self.model.get_emb(data=data)
        num_nodes = node_feature.size()[0]

        # this part is used to add noise
        # node_feature = node_features
        static_node_feature = node_feature.clone().detach()
        node_feature_std, node_feature_mean = torch.std_mean(static_node_feature, dim=0)

        # this part is used to generate assignment matrix
        abstract_features_1 = torch.tanh(self.fully_connected_1(node_feature))
        assignment = torch.nn.functional.softmax(self.fully_connected_2(abstract_features_1), dim=1)

        gumbel_assignment = self.gumbel_softmax(assignment)

        # This is the graph embedding
        # graph_feature = torch.sum(node_feature, dim=0, keepdim=True)
        graph_feature = self.model.readout(node_feature, data.batch)[None, :]

        # add noise to the node representation
        node_feature_mean = node_feature_mean.repeat(num_nodes, 1)

        # noisy_graph_representation
        lambda_pos = gumbel_assignment[:, 0].unsqueeze(dim=1)
        lambda_neg = gumbel_assignment[:, 1].unsqueeze(dim=1)

        # print(assignment[:0],lambda_pos)

        # this is subgraph embedding
        # subgraph_representation = torch.sum(lambda_pos * node_feature, dim=0, keepdim=True)
        subgraph_representation = self.model.readout(lambda_pos * node_feature, data.batch)[None, :]

        noisy_node_feature_mean = lambda_pos * node_feature + lambda_neg * node_feature_mean
        noisy_node_feature_std = lambda_neg * node_feature_std

        noisy_node_feature = noisy_node_feature_mean + torch.rand_like(noisy_node_feature_mean) * noisy_node_feature_std
        # noisy_graph_feature = torch.sum(noisy_node_feature, dim=0, keepdim=True)
        noisy_graph_feature = self.model.readout(noisy_node_feature, data.batch)[None, :]

        KL_tensor = 0.5 * ((noisy_node_feature_std ** 2) / (node_feature_std + epsilon) ** 2) + \
                    torch.sum(((noisy_node_feature_mean - node_feature_mean) / (node_feature_std + epsilon)) ** 2,
                              dim=0)  # +\
        #            torch.log(node_feature_std / (noisy_node_feature_std + epsilon) + epsilon)
        KL_Loss = torch.mean(KL_tensor)

        if torch.cuda.is_available():
            EYE = torch.ones(2).cuda()
            Pos_mask = torch.FloatTensor([1, 0]).cuda()
        else:
            EYE = torch.ones(2)
            Pos_mask = torch.FloatTensor([1, 0])

        Adj = to_dense_adj(edges, batch=torch.zeros(data.x.shape[0], dtype=int, device=data.x.device), max_num_nodes=assignment.shape[0])[0]
        Adj.requires_grad = False
        new_adj = torch.mm(torch.t(assignment), Adj)
        new_adj = torch.mm(new_adj, assignment)

        normalize_new_adj = F.normalize(new_adj, p=1, dim=1)

        norm_diag = torch.diag(normalize_new_adj)
        pos_penalty = self.mseloss(norm_diag, EYE)
        # cal preserve rate
        preserve_rate = torch.sum(assignment[:, 0] > 0.5) / assignment.size()[0]

        return graph_feature, noisy_graph_feature, subgraph_representation, pos_penalty, KL_Loss, preserve_rate

    def return_att(self, data):
        node_feature, _ = self.model.get_emb(data=data)

        abstract_features_1 = torch.tanh(self.fully_connected_1(node_feature))
        attention = torch.nn.functional.softmax(self.fully_connected_2(abstract_features_1), dim=1)

        return attention

    def gumbel_softmax(self, prob):

        return F.gumbel_softmax(prob, tau=1, dim=-1)


class Subgraph(torch.nn.Module):

    def __init__(self, model, args):
        super(Subgraph, self).__init__()
        self.model = model
        self.args = args
        self._setup()
        self.mse_criterion = torch.nn.MSELoss(reduction='mean')
        self.bce_criterion = torch.nn.BCELoss(reduction='mean')
        self.criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        self.relu = torch.nn.ReLU()

    def _setup(self):
        self.graph_level_model = SAGE(self.model)

    def forward(self, graphs, pred_label=None):
        batch_size = graphs.batch[-1] + 1
        embeddings = []
        positive = []
        negative = []
        subgraph = []
        noisy_graph = []

        labels = []

        positive_penalty = 0
        preserve_rate = 0
        KL_Loss = 0
        for i in range(batch_size):
            graph = graphs[i]
            if graph.edge_index.shape[1] == 0:
                continue
            embedding, noisy, subgraph_emb, pos_penalty, kl_loss, one_preserve_rate = self.graph_level_model(graph)

            embeddings.append(embedding)
            positive.append(noisy)
            subgraph.append(subgraph_emb)
            noisy_graph.append(noisy)
            positive_penalty += pos_penalty
            KL_Loss += kl_loss
            preserve_rate += one_preserve_rate
            if pred_label is None:
                labels.append(graph.y)
            else:
                labels.append(pred_label[i][None])

        embeddings = torch.cat(tuple(embeddings), dim=0)
        positive = torch.cat(tuple(positive), dim=0)
        subgraph = torch.cat(tuple(subgraph), dim=0)
        noisy_graph = torch.cat(tuple(noisy_graph), dim=0)

        labels = torch.cat(tuple(labels), dim=0).view(-1, 1)

        positive_penalty = positive_penalty / batch_size
        KL_Loss = KL_Loss / batch_size
        preserve_rate = preserve_rate / batch_size

        cls_loss = self.supervise_classify_loss(embeddings, positive, subgraph, labels)

        return embeddings, positive, noisy_graph, KL_Loss, cls_loss, self.args.con_weight * positive_penalty, preserve_rate

    def supervise_classify_loss(self, embeddings, positive, subgraph, labels):
        data = torch.cat((embeddings, positive), dim=0)

        labels = torch.cat((labels, labels), dim=0)
        pred = self.model.ffn(data)
        loss = self.criterion(pred.squeeze(), labels.squeeze().long())

        return loss
