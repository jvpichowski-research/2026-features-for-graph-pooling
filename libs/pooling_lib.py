import os
import sys
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import bmm
from torch_geometric.data import Data
from torch_geometric.nn.dense.mincut_pool import _rank3_trace
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import to_dense_adj


class LabelsToInt(BaseTransform):
    def forward(self, data):
        data.y = data.y.to(torch.long)
        return data


def x_log_x_sum(p, full_batch=True):
    if full_batch:
        return (p * torch.nan_to_num(torch.log2(p + 1e-8), nan=0.0)).sum()
    else:
        p = p * torch.nan_to_num(torch.log2(p + 1e-8), nan=0.0)
        while p.dim() > 1:
            p = p.sum(dim=-1)
        return p


def mkSmartTeleportationFlow(A, alpha=0.15, iter=1000):
    # build the transition matrix
    T = torch.nan_to_num(A.T * (torch.sum(A, 1) ** (-1.0)).to_dense(), nan=0.0).T

    # distribution according to nodes' in-degrees
    e_v = (torch.sum(A, dim=0) / torch.sum(A)).to_dense()

    # calculate the flow distribution with a power iteration
    p = e_v
    for _ in range(iter):
        p = alpha * e_v + (1 - alpha) * p @ T

    # make the flow matrix for minimising the map equation
    F = alpha * A / torch.sum(A) + (1 - alpha) * (p * T.T).T

    return F, p


class AddFlow(BaseTransform):
    def __init__(self, iter=1000, alpha=0.15, device=None, progress=None):
        self.iter = iter
        self.alpha = alpha
        self.device = device
        self.progress = progress
        self.count = 0

    # create smart teleportation flow matrix and flow distribution
    def forward(self, data: Data) -> Data:
        if self.progress is not None:
            if self.count % self.progress == 0:
                print(self.count)
            self.count += 1

        dev = None
        if self.device is not None:
            dev = data.edge_index.device
            data = data.to(self.device)

        # todo remove duplicated edges
        adj = to_dense_adj(
            edge_index=data.edge_index,
            max_num_nodes=data.num_nodes,
            edge_attr=data.edge_weight,
        )[0]
        F, p = mkSmartTeleportationFlow(adj, self.alpha, self.iter)

        f = F[data.edge_index[0], data.edge_index[1]]
        data.f = f
        data.p = p

        if dev is not None:
            data = data.to(dev)

        return data


def dmon_pool(
    x: torch.Tensor,
    adj: torch.Tensor,
    s: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    eps: float = 1e-15,
) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:

    s = torch.softmax(s, dim=-1)

    (batch_size, num_nodes, _), C = x.size(), s.size(-1)

    if mask is None:
        mask = torch.ones(batch_size, num_nodes, dtype=torch.bool, device=x.device)

    mask = mask.view(batch_size, num_nodes, 1).to(x.dtype)
    x, s = x * mask, s * mask

    out = F.selu(torch.matmul(s.transpose(1, 2), x))
    out_adj = torch.matmul(torch.matmul(s.transpose(1, 2), adj), s)

    # Spectral loss:
    degrees = torch.einsum("ijk->ij", adj)  # B X N
    degrees = degrees.unsqueeze(-1) * mask  # B x N x 1
    degrees_t = degrees.transpose(1, 2)  # B x 1 x N

    m = torch.einsum("ijk->i", degrees) / 2  # B
    m_expand = m.view(-1, 1, 1).expand(-1, C, C)  # B x C x C

    ca = torch.matmul(s.transpose(1, 2), degrees)  # B x C x 1
    cb = torch.matmul(degrees_t, s)  # B x 1 x C

    normalizer = torch.matmul(ca, cb) / 2 / m_expand
    decompose = out_adj - normalizer
    spectral_loss = -_rank3_trace(decompose) / 2 / m
    spectral_loss = spectral_loss.mean()

    # Orthogonality regularization:
    ss = torch.matmul(s.transpose(1, 2), s)
    i_s = torch.eye(C).type_as(ss)
    ortho_loss = torch.norm(
        ss / torch.norm(ss, dim=(-1, -2), keepdim=True) - i_s / torch.norm(i_s),
        dim=(-1, -2),
    )
    ortho_loss = ortho_loss.mean()

    # Cluster loss:
    cluster_size = torch.einsum("ijk->ik", s)  # B x C
    cluster_loss = torch.norm(input=cluster_size, dim=1)
    cluster_loss = cluster_loss / mask.sum(dim=1) * torch.norm(i_s) - 1
    cluster_loss = cluster_loss.mean()

    # Normalize coarsened adjacency matrix:
    ind = torch.arange(C, device=out_adj.device)
    out_adj[:, ind, ind] = 0
    d = torch.einsum("ijk->ij", out_adj)
    d = torch.sqrt(d)[:, None] + eps
    out_adj = (out_adj / d) / d.transpose(1, 2)

    return out, out_adj, spectral_loss, ortho_loss, cluster_loss


def mapeq_loss2(Sm, Flow, p, mask=None, hard=False, full_batch=True):
    # add artificial batch if no batch is there
    Sm = Sm.unsqueeze(0) if Sm.dim() == 2 else Sm
    Flow = Flow.unsqueeze(0) if Flow.dim() == 2 else Flow
    p = p.unsqueeze(0) if p.dim() == 1 else p
    p = p.clone()
    Flow = Flow.clone()

    batch_size, num_nodes, _ = Sm.size()

    if mask is not None:
        mask = mask.view(batch_size, num_nodes, 1)
        # we only mask first layer to exclude nodes
        Sm = Sm * mask

    # -- Flow, Module Flow, Submodule Flow --
    Cm = bmm(Sm.transpose(-1, -2), bmm(Flow, Sm))

    # -- Node rates --
    p_hard = hard
    if p_hard:
        P = p
    else:
        P = torch.einsum("bi,bij->bij", p, Sm)

    # -- Module rates --
    Qm_out = Cm.sum(dim=-1) - torch.diagonal(Cm, dim1=-2, dim2=-1)
    Qm_in = Cm.sum(dim=-2) - torch.diagonal(Cm, dim1=-2, dim2=-1)
    Qm = Qm_out + bmm(Sm.transpose(-1, -2), p.unsqueeze(-1)).squeeze(-1)

    # -- Top Level Rates --
    q = 1.0 - torch.diagonal(Cm, dim1=-2, dim2=-1).sum(-1)

    # -- Total Code Length --
    codelength = (
        x_log_x_sum(q, full_batch=full_batch)
        - x_log_x_sum(Qm_out, full_batch=full_batch)
        - x_log_x_sum(Qm_in, full_batch=full_batch)
        + x_log_x_sum(Qm, full_batch=full_batch)
        - x_log_x_sum(P, full_batch=full_batch)
    )

    loss = (codelength / batch_size) if full_batch else codelength

    return loss


def mdl_pool(
    x: torch.Tensor,
    adj: torch.Tensor,
    s: torch.Tensor,
    flow,
    p,
    mask: Optional[torch.Tensor] = None,
):
    s = torch.softmax(s, dim=-1)
    (batch_size, num_nodes, _), C = x.size(), s.size(-1)

    if mask is None:
        mask = torch.ones(batch_size, num_nodes, dtype=torch.bool, device=x.device)

    mask = mask.view(batch_size, num_nodes, 1).to(x.dtype)
    x, s = x * mask, s * mask

    x = F.selu(torch.matmul(s.transpose(1, 2), x))
    adj = torch.matmul(torch.matmul(s.transpose(1, 2), adj), s)

    loss = mapeq_loss2(s, flow, p, mask=mask, hard=False)

    return x, adj, loss


def just_balance_pool(x, adj, s, mask=None, normalize=True):
    r"""The Just Balance pooling operator from the `"Simplifying Clustering with
    Graph Neural Networks" <https://arxiv.org/abs/2207.08779>`_ paper

    .. math::
        \mathbf{X}^{\prime} &= {\mathrm{softmax}(\mathbf{S})}^{\top} \cdot
        \mathbf{X}

        \mathbf{A}^{\prime} &= {\mathrm{softmax}(\mathbf{S})}^{\top} \cdot
        \mathbf{A} \cdot \mathrm{softmax}(\mathbf{S})

    based on dense learned assignments :math:`\mathbf{S} \in \mathbb{R}^{B
    \times N \times C}`.
    Returns the pooled node feature matrix, the coarsened and symmetrically
    normalized adjacency matrix and the following auxiliary objective:

    .. math::
        \mathcal{L} = - {\mathrm{Tr}(\sqrt{\mathbf{S}^{\top} \mathbf{S}})}

    Args:
        x (Tensor): Node feature tensor :math:`\mathbf{X} \in \mathbb{R}^{B \times N \times F}`
            with batch-size :math:`B`, (maximum) number of nodes :math:`N`
            for each graph, and feature dimension :math:`F`.
        adj (Tensor): Symmetrically normalized adjacency tensor
            :math:`\mathbf{A} \in \mathbb{R}^{B \times N \times N}`.
        s (Tensor): Assignment tensor :math:`\mathbf{S} \in \mathbb{R}^{B \times N \times C}`
            with number of clusters :math:`C`. The softmax does not have to be
            applied beforehand, since it is executed within this method.
        mask (BoolTensor, optional): Mask matrix
            :math:`\mathbf{M} \in {\{ 0, 1 \}}^{B \times N}` indicating
            the valid nodes for each graph. (default: :obj:`None`)

    :rtype: (:class:`Tensor`, :class:`Tensor`, :class:`Tensor`,
        :class:`Tensor`)
    """

    EPS = 1e-15

    x = x.unsqueeze(0) if x.dim() == 2 else x
    adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
    s = s.unsqueeze(0) if s.dim() == 2 else s

    (batch_size, num_nodes, _), k = x.size(), s.size(-1)

    s = torch.softmax(s, dim=-1)

    if mask is not None:
        mask = mask.view(batch_size, num_nodes, 1).to(x.dtype)
        x, s = x * mask, s * mask

    out = torch.matmul(s.transpose(1, 2), x)
    out_adj = torch.matmul(torch.matmul(s.transpose(1, 2), adj), s)

    # Loss
    ss = torch.matmul(s.transpose(1, 2), s)
    ss_sqrt = torch.sqrt(ss + EPS)
    loss = torch.mean(-_rank3_trace(ss_sqrt))
    if normalize:
        loss = loss / torch.sqrt(torch.tensor(num_nodes * k))

    # Fix and normalize coarsened adjacency matrix.
    ind = torch.arange(k, device=out_adj.device)
    out_adj[:, ind, ind] = 0
    d = torch.einsum("ijk->ij", out_adj)
    d = torch.sqrt(d)[:, None] + EPS
    out_adj = (out_adj / d) / d.transpose(1, 2)

    return out, out_adj, loss


def jb_pool(
    x: torch.Tensor,
    adj: torch.Tensor,
    s: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
):

    out, out_adj, loss = just_balance_pool(x=x, adj=adj, s=s, mask=mask, normalize=True)
    return out, out_adj, loss
