"""Correspondence-safe local topology loss for supervised scene flow."""

import torch
import torch.nn.functional as F


def _square_distance(src, dst):
    """Return pairwise squared distances for ``[B, N, C]`` point tensors."""
    dist = -2 * torch.matmul(src, dst.transpose(1, 2))
    dist += torch.sum(src ** 2, dim=-1, keepdim=True)
    dist += torch.sum(dst ** 2, dim=-1).unsqueeze(1)
    return dist


def topology_edge_loss(
    pcd_src,
    gt_flow,
    pred_flow,
    k=16,
    num_anchors=512,
    query_chunk_size=128,
    beta=0.01,
):
    """Match predicted and ground-truth local deformation edge lengths.

    The lung source and target clouds are independently sampled, so equal source
    and target indices are not correspondences. Both warped clouds below are
    instead constructed from the same source samples, giving every local edge a
    valid point-wise correspondence.

    KNN queries are computed for a deterministic subset of source anchors and in
    chunks. This bounds the temporary search tensor at
    ``[B, query_chunk_size, N]`` instead of allocating ``[B, N, N]`` matrices.
    """
    if pred_flow.ndim != 3 or gt_flow.ndim != 3 or pcd_src.ndim != 3:
        raise ValueError('pcd_src, gt_flow, and pred_flow must all be rank-3 tensors')

    if pred_flow.shape[1] == 3 and pred_flow.shape[2] == pcd_src.shape[1]:
        pred_flow = pred_flow.permute(0, 2, 1)
    if pred_flow.shape != pcd_src.shape or gt_flow.shape != pcd_src.shape:
        raise ValueError(
            'Expected pcd_src, gt_flow, and converted pred_flow to have the same '
            f'shape, got {pcd_src.shape}, {gt_flow.shape}, and {pred_flow.shape}'
        )

    num_points = pcd_src.shape[1]
    if num_points < 2:
        raise ValueError('topology_edge_loss requires at least two source points')
    if k < 1 or num_anchors < 1 or query_chunk_size < 1:
        raise ValueError('k, num_anchors, and query_chunk_size must be positive')

    k_eff = min(int(k), num_points - 1)
    anchor_count = min(int(num_anchors), num_points)
    anchor_idx = torch.linspace(
        0, num_points - 1, steps=anchor_count, device=pcd_src.device
    ).round().long()

    # Neighborhood membership is a fixed geometric relation and should not carry
    # gradients. Float32 distance calculation is also safer under AMP.
    neighbor_chunks = []
    with torch.no_grad():
        source_for_knn = pcd_src.detach().float()
        for start in range(0, anchor_count, int(query_chunk_size)):
            chunk_anchor_idx = anchor_idx[start:start + int(query_chunk_size)]
            query = source_for_knn[:, chunk_anchor_idx, :]
            sqrdists = _square_distance(query, source_for_knn)
            self_idx = chunk_anchor_idx.view(1, -1, 1).expand(
                pcd_src.shape[0], -1, 1
            )
            sqrdists.scatter_(2, self_idx, float('inf'))
            neighbor_chunks.append(
                torch.topk(
                    sqrdists, k_eff, dim=-1, largest=False, sorted=False
                ).indices
            )
        neighbor_idx = torch.cat(neighbor_chunks, dim=1)

    pred_warped = pcd_src + pred_flow
    gt_warped = pcd_src + gt_flow
    batch_idx = torch.arange(pcd_src.shape[0], device=pcd_src.device)[:, None, None]

    pred_anchor = pred_warped[:, anchor_idx, :].unsqueeze(2)
    gt_anchor = gt_warped[:, anchor_idx, :].unsqueeze(2)
    pred_neighbors = pred_warped[batch_idx, neighbor_idx]
    gt_neighbors = gt_warped[batch_idx, neighbor_idx]

    pred_lengths = torch.linalg.vector_norm(pred_neighbors - pred_anchor, dim=-1)
    gt_lengths = torch.linalg.vector_norm(gt_neighbors - gt_anchor, dim=-1)
    return F.smooth_l1_loss(pred_lengths, gt_lengths, beta=float(beta))
