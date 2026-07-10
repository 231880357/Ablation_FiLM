import numpy as np
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.csgraph import dijkstra
from ripser import ripser

def compute_topo_features(pcd):
    """
    Computes topological features (persistent homology) for a point cloud using Manifold distance.
    Returns a 1D numpy array of features.
    Here we compute summaries of 0-dim and 1-dim persistence diagrams:   
    [avg_life_0, max_life_0, entropy_0, avg_life_1, max_life_1, entropy_1]      
    Total 6 features.
    """
    N = pcd.shape[0]
    target_N = 500
    if N > target_N:
        idx = np.random.choice(N, target_N, replace=False)
        sub_pcd = pcd[idx]
    else:
        sub_pcd = pcd

    try:
        k = min(20, len(sub_pcd) - 1)
        if k <= 0:
             return np.zeros(6, dtype=np.float32)

        knn_graph = kneighbors_graph(sub_pcd, n_neighbors=k, mode='distance', include_self=False)
        dist_matrix = dijkstra(csgraph=knn_graph, directed=False)
        
        # Handle disconnected components
        max_dist = np.nanmax(dist_matrix[dist_matrix != np.inf]) if np.any(dist_matrix != np.inf) else 10.0
        dist_matrix[dist_matrix == np.inf] = max_dist * 2.0
        np.fill_diagonal(dist_matrix, 0)
        
        rips_result = ripser(dist_matrix, distance_matrix=True, maxdim=1)
        dgms = rips_result['dgms']
        
        def get_feats(diag):
            if len(diag) == 0:
                return np.array([0, 0, 0], dtype=np.float32)
            
            births = diag[:, 0]
            deaths = diag[:, 1]
            finite_mask = (deaths != np.inf) & (~np.isnan(deaths))
            if np.any(finite_mask):
                max_finite = deaths[finite_mask].max()
            else:
                max_finite = births.max() if len(births) > 0 else 1.0

            deaths_finite = np.copy(deaths)
            deaths_finite[~finite_mask] = max_finite * 1.1 + 1e-4

            lifetimes = deaths_finite - births
            valid_mask = lifetimes > 1e-6
            lifetimes = lifetimes[valid_mask]

            if len(lifetimes) == 0:
                 return np.array([0, 0, 0], dtype=np.float32)

            avg_life = np.mean(lifetimes)
            max_life = np.max(lifetimes)

            sum_life = np.sum(lifetimes)
            if sum_life > 0:
                probs = lifetimes / sum_life
                entropy = -np.sum(probs * np.log(probs + 1e-10))
            else:
                entropy = 0.0

            return np.array([avg_life, max_life, entropy], dtype=np.float32)        

        f0 = get_feats(dgms[0] if len(dgms) > 0 else [])
        f1 = get_feats(dgms[1] if len(dgms) > 1 else [])
        
        return np.concatenate([f0, f1]).astype(np.float32)

    except Exception as e:
        print(f"Topology extraction warning: {e}")
        return np.zeros(6, dtype=np.float32)
