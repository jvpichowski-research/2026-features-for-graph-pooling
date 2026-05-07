import csv
import json
import os
import random
from copy import deepcopy
from math import ceil
from typing import Optional

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import torch
import torch.nn.functional as F
import torch_geometric.utils as utils
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.nn import Linear, Node2Vec
from torch_geometric.transforms import AddGPSE, BaseTransform
from torch_geometric.transforms.add_positional_encoding import add_node_attr


class NewDataset(InMemoryDataset):
    def __init__(self, data_list):
        super().__init__(None)
        self.data, self.slices = self.collate(data_list)


def create_cluster(num_nodes, feature_range, start_node_id):

    cluster = nx.complete_graph(num_nodes)
    cluster = nx.relabel_nodes(
        cluster, {i: start_node_id + i for i in range(num_nodes)}
    )

    min_val, max_val = feature_range
    for node in cluster.nodes():
        cluster.nodes[node]["x"] = float(random.uniform(min_val, max_val))

    return cluster


def create_graph_from_clusters(cluster_specs, num_connections, central_cluster):

    main_graph = nx.Graph()
    current_node_id = 0
    cluster_nodes = []

    for num_nodes, feature_range in cluster_specs:
        cluster = create_cluster(num_nodes, feature_range, current_node_id)
        main_graph = nx.disjoint_union(main_graph, cluster)
        cluster_nodes.append(list(range(current_node_id, current_node_id + num_nodes)))
        current_node_id += num_nodes

    central_cluster_nodes = cluster_nodes[central_cluster]

    for i, other_cluster in enumerate(cluster_nodes):
        if i != central_cluster:
            for _ in range(num_connections):
                source_node = random.choice(other_cluster)
                target_node = random.choice(central_cluster_nodes)
                main_graph.add_edge(source_node, target_node)

    return main_graph, len(cluster_specs) - 1, len(cluster_specs)


def create_pyg_data(specs, connections, central_idx):
    graph, num_blue_clusters, total_clusters = create_graph_from_clusters(
        specs, connections, central_idx
    )
    pyg_data = utils.from_networkx(graph, group_node_attrs=["x"])
    pyg_data.x = pyg_data.x.float()
    pyg_data.y = torch.tensor([total_clusters], dtype=torch.float)
    return pyg_data


def generate_dataset(num_graphs, specs_variations):
    dataset = []
    for i in range(num_graphs):
        spec = random.choice(list(specs_variations.values()))
        data = create_pyg_data(specs=spec, connections=1, central_idx=1)

        true_labels = []
        cluster_id = 0
        for num_nodes, _ in spec:
            true_labels.extend([cluster_id] * num_nodes)
            cluster_id += 1

        data.true_labels = torch.tensor(true_labels, dtype=torch.long)
        data.num_clusters = len(torch.unique(data.true_labels))
        dataset.append(data)
    return dataset


def get_graphs_specification():
    graph_spec = {
        "spec_1": [(12, (1, 1)), (12, (1, 1))],
        "spec_2": [(6, (1, 1)), (12, (1, 1)), (6, (1, 1))],
        "spec_3": [(4, (1, 1)), (12, (1, 1)), (4, (1, 1)), (4, (1, 1))],
        "spec_4": [(3, (1, 1)), (12, (1, 1)), (3, (1, 1)), (3, (1, 1)), (3, (1, 1))],
        "spec_5": [
            (8, (1, 1)),
            (48, (1, 1)),
            (8, (1, 1)),
            (8, (1, 1)),
            (8, (1, 1)),
            (8, (1, 1)),
            (8, (1, 1)),
        ],
    }
    return graph_spec


def shuffle_node_ids(data):
    perm = torch.randperm(data.num_nodes)

    shuffled_data = data.clone()
    shuffled_data.x = data.x[perm]
    shuffled_data.true_labels = data.true_labels[perm]

    # Map the edge_index to the new node IDs
    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(data.num_nodes)

    shuffled_data.edge_index = inv_perm[data.edge_index]

    return shuffled_data


def apply_gpse(dataset, transform):

    new_dataset = deepcopy(dataset)

    new_data_list = []
    for data in new_dataset:
        data_gpse = transform(data)
        data.x = torch.cat([data_gpse.x, data_gpse.pestat_GPSE], dim=1)
        new_data_list.append(data)

    return NewDataset(new_data_list)


def apply_laplacian_pe(dataset, transform):

    new_dataset = deepcopy(dataset)

    new_data_list = []
    for data in new_dataset:
        data = transform(data)
        data.x = torch.cat([data.x, data.laplacian_eigenvector_pe], dim=1)
        new_data_list.append(data)

    return NewDataset(new_data_list)


def apply_rw_pe(dataset, transform):

    new_dataset = deepcopy(dataset)

    new_data_list = []
    for data in new_dataset:
        data = transform(data)
        data.x = torch.cat([data.x, data.random_walk_pe], dim=1)
        new_data_list.append(data)

    return NewDataset(new_data_list)


@functional_transform("add_node2vec_pe")
class AddNode2VecPE(BaseTransform):
    def __init__(
        self,
        embedding_dim: int,
        walks_per_node: int,
        n2v_epochs: int,
        attr_name: Optional[str] = "node2vec_pe",
        device: str = "cpu",
    ) -> None:
        self.embedding_dim = embedding_dim
        self.walks_per_node = walks_per_node
        self.n2v_epochs = n2v_epochs
        self.attr_name = attr_name
        self.device = device

    def forward(self, data: Data) -> Data:
        assert data.edge_index is not None

        pe = generate_node2vec_embeddings(
            data, self.embedding_dim, self.walks_per_node, self.n2v_epochs, self.device
        ).to("cpu")

        data = add_node_attr(data, pe, attr_name=self.attr_name)
        return data


class AddGPSEPE:
    def __init__(self, model, proj_dim=8):
        self.model = model
        self.proj_dim = proj_dim
        self.gpse_transform = AddGPSE(model=model)
        self.projector = None

    def __call__(self, data):
        data = self.gpse_transform(data)

        if self.projector is None:
            input_dim = data.pestat_GPSE.size(-1)
            self.projector = Linear(input_dim, self.proj_dim)

        data.pe = self.projector(data.pestat_GPSE)

        return data


def apply_node2vec_pe(dataset, embedding_dim, walks_per_node, n2v_epochs, device="cpu"):

    new_dataset = deepcopy(dataset)
    new_data_list = []

    print("Node2Vec pretraining...")
    for data in new_dataset:
        node2vec_pe = generate_node2vec_embeddings(
            data, embedding_dim, walks_per_node, n2v_epochs, device
        ).to("cpu")
        data = data.to("cpu")
        data.x = torch.cat([data.x, node2vec_pe], dim=1)

        new_data_list.append(data)

    return NewDataset(new_data_list)


def get_cluster_assignments(model, data):
    embeddings, cluster_assignments = model.get_embedding(
        data.x, data.edge_index, data.batch
    )
    s_normalized = F.softmax(cluster_assignments, dim=-1)
    pred_labels_new = torch.argmax(s_normalized, dim=-1).squeeze()
    return pred_labels_new


def visualize_graph(pyg_data):
    G1 = utils.to_networkx(pyg_data, to_undirected=True)
    unique_features = sorted(list(set(x.item() for x in pyg_data.x)))
    cmap = plt.get_cmap("viridis", len(unique_features))
    feature_to_color = {feature: cmap(i) for i, feature in enumerate(unique_features)}
    node_colors = [feature_to_color[x.item()] for x in pyg_data.x]

    pos = nx.spring_layout(G1, seed=42)
    plt.figure()
    nx.draw(G1, pos=pos, with_labels=True, node_color=node_colors)
    plt.show()


def parse_args_from_csv(args, counter, parser):
    loaded_args = {}

    with open(args.args) as f:
        loaded_args_list = [row for row in csv.DictReader(f, skipinitialspace=True)]
        if 0 <= counter < len(loaded_args_list):
            loaded_args = loaded_args_list[counter]
        else:
            print(f"SLURM_ARRAY_TASK_ID ({counter}) out of range.")

    arg_types = {action.dest: action.type for action in parser._actions}

    for key, value in loaded_args.items():
        if key in arg_types and value is not None:
            if key == "wl" and value.strip().lower() == "inf":
                setattr(args, key, float("inf"))
                continue

            converter = arg_types[key]
            try:
                setattr(args, key, converter(value))
            except ValueError:
                print(
                    f"Could not convert CSV value '{value}' for '{key}' using type {converter.__name__}. Skipping."
                )
    return args


def generate_node2vec_embeddings(data, embedding_dim, walks_per_node, epochs, device):

    model = Node2Vec(
        data.edge_index,
        embedding_dim=embedding_dim,
        walk_length=20,
        context_size=10,
        walks_per_node=walks_per_node,
        p=1.0,
        q=1.0,
        sparse=True,
        num_nodes=data.num_nodes,
    ).to(device)

    loader = model.loader(batch_size=128, shuffle=True)
    optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    model.eval()
    # return learned node embeddings for given graph
    return model(torch.arange(data.num_nodes, device=device)).detach()


def process_json_files(results_dir_pwd):
    all_data = []

    for dataset_folder in os.listdir(results_dir_pwd):
        dataset_path = os.path.join(results_dir_pwd, dataset_folder)

        if os.path.isdir(dataset_path):
            dataset_name = dataset_folder

            for file_name in os.listdir(dataset_path):
                if file_name.endswith("_results.json"):
                    file_path = os.path.join(dataset_path, file_name)

                    with open(file_path, "r") as f:
                        data = json.load(f)

                    pe = data.get("args", {}).get("pe", "N/A")
                    hidden_channels = data.get("args", {}).get("hidden_channels", "N/A")
                    num_layers = data.get("args", {}).get("num_layers", "N/A")
                    mean_acc = data.get("overall_results", {}).get(
                        "mean_test_accuracy", float("nan")
                    )
                    std_acc = data.get("overall_results", {}).get(
                        "std_dev_test_accuracy", float("nan")
                    )

                    row = {
                        "Dataset": dataset_name,
                        "PE Type": pe,
                        "Hidden Channels": hidden_channels,
                        "Num Layers": num_layers,
                        "Mean Test Accuracy": mean_acc,
                        "Std Dev Test Accuracy": std_acc,
                    }
                    all_data.append(row)

    df = pd.DataFrame(all_data)
    df = df.sort_values(by=["Dataset", "PE Type", "Hidden Channels", "Num Layers"])

    return df
