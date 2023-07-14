"""
FileName: explainer.py
Description: 
Time: 2021/8/20 23:03
Project: GNN_benchmark
Author: Shurui Gui
"""

class PGExplainer(ExplainerBase):

    def __init__(self, model, epochs=100, lr=3e-1, explain_graph=False, molecule=False):
        super().__init__(model=model, epochs=epochs, lr=lr, explain_graph=explain_graph, molecule=molecule)
        self.pg = pgex(model, in_channels=600, device=self.device, explain_graph=explain_graph)

    def forward(self, x, edge_index, mask_features=False,
                positive=True, **kwargs):
        r"""Learns and returns a node feature mask and an edge mask that play a
        crucial role to explain the prediction made by the GNN for node
        :attr:`node_idx`.

        Args:
            data (Batch): batch from dataloader
            edge_index (LongTensor): The edge indices.
            pos_neg (Literal['pos', 'neg']) : get positive or negative mask
            **kwargs (optional): Additional arguments passed to the GNN module.

        :rtype: (:class:`Tensor`, :class:`Tensor`)
        """
        self.model.eval()
        super().forward(x, edge_index)

        # --- substitute edge_index with self loops ---
        edge_index, _ = add_self_loops(edge_index, num_nodes=self.num_nodes)

        # Only operate on a k-hop subgraph around `node_idx`.
        # Get subgraph and relabel the node, mapping is the relabeled given node_idx.
        if data_args.model_level == 'node':
            node_idx = kwargs.get('node_idx')
            self.node_idx = node_idx
            assert node_idx is not None
            _, _, _, self.hard_edge_mask = subgraph(
                node_idx, self.__num_hops__, edge_index, relabel_nodes=True,
                num_nodes=None, flow=self.__flow__())

        # Assume the mask we will predict
        labels = tuple(i for i in range(data_args.num_classes))
        ex_labels = tuple(torch.tensor([label]).to(data_args.device) for label in labels)

        # Calculate mask
        print('#D#Masks calculate...')
        edge_masks = []
        for ex_label in ex_labels:

            edge_masks.append(self.control_sparsity(self.pg(x, edge_index, **kwargs)[1][0], sparsity=kwargs.get('sparsity')))
            # edge_masks.append(self.gnn_explainer_alg(x, edge_index, ex_label))


        print('#D#Predict...')
        self.__clear_masks__()
        self.__set_masks__(x, edge_index)
        with torch.no_grad():
            related_preds = self.eval_related_pred(x, edge_index, edge_masks, **kwargs)

        self.__clear_masks__()

        return None, edge_masks, related_preds