import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv


class JobGraphSAGE(nn.Module):
    def __init__(self, metadata, hidden_dim=256, out_dim=256, dropout=0.3):
        super().__init__()

        self.dropout = dropout

        self.conv1 = HeteroConv({
            edge_type: SAGEConv((-1, -1), hidden_dim)
            for edge_type in metadata[1]
        }, aggr="mean")

        self.conv2 = HeteroConv({
            edge_type: SAGEConv((-1, -1), out_dim)
            for edge_type in metadata[1]
        }, aggr="mean")

        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x_dict, edge_index_dict):
        x_dict = self.conv1(x_dict, edge_index_dict)

        x_dict = {
            k: F.relu(v)
            for k, v in x_dict.items()
        }

        x_dict = {
            k: F.dropout(v, p=self.dropout, training=self.training)
            for k, v in x_dict.items()
        }

        x_dict = self.conv2(x_dict, edge_index_dict)

        job_z = x_dict["job"]
        job_z = self.norm(job_z)
        job_z = F.normalize(job_z, p=2, dim=1)

        return job_z


class CVJobLinkPredictor(nn.Module):
    def __init__(self, graph_model, cv_input_dim, hidden_dim=256):
        super().__init__()

        self.graph_model = graph_model

        self.cv_encoder = nn.Sequential(
            nn.Linear(cv_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

    def encode_cv(self, cv_embeddings):
        cv_z = self.cv_encoder(cv_embeddings)
        cv_z = F.normalize(cv_z, p=2, dim=1)
        return cv_z

    def encode_job(self, graph):
        job_z = self.graph_model(
            graph.x_dict,
            graph.edge_index_dict
        )

        job_z = F.normalize(job_z, p=2, dim=1)
        return job_z

    def forward(self, graph, cv_embeddings, cv_idx, job_idx):
        cv_z_all = self.encode_cv(cv_embeddings)
        job_z_all = self.encode_job(graph)

        cv_z = cv_z_all[cv_idx]
        job_z = job_z_all[job_idx]

        logits = torch.sum(cv_z * job_z, dim=1)

        return logits, job_z_all, cv_z_all