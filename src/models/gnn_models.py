"""
GNN Models - Graph Neural Networks
3D volume'leri graph'a dönüştürerek işleyen modeller
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
from .base_model import BaseGraphModel
try:
    import torch_geometric
    from torch_geometric.nn import GCNConv, GATConv, SAGEConv, global_mean_pool, global_max_pool
    from torch_geometric.data import Data, Batch
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False
    print("⚠️  torch_geometric bulunamadı. GNN modelleri kullanılamayacak.")
    print("   Kurulum: pip install torch-geometric")


class VoxelToGraph:
    """3D volume'ü graph'a dönüştürür"""
    
    @staticmethod
    def volume_to_graph(x: torch.Tensor, threshold: float = 0.5, k_neighbors: int = 6) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        3D volume'den graph oluşturur (sadece non-zero voxel'ler)
        
        Args:
            x: Input tensor (B, C, D, H, W)
            threshold: Voxel threshold (binary mask için)
            k_neighbors: Her node için k-nearest neighbor
            
        Returns:
            (node_features, edge_index)
        """
        batch_size = x.size(0)
        graphs = []
        
        for b in range(batch_size):
            volume = x[b, 0]  # (D, H, W)
            
            # Non-zero voxel'leri bul
            nonzero_indices = torch.nonzero(volume > threshold, as_tuple=False)  # (N, 3)
            
            if len(nonzero_indices) == 0:
                # Boş graph - en azından bir node ekle
                node_features = torch.zeros(1, 4, device=x.device)
                edge_index = torch.empty(2, 0, dtype=torch.long, device=x.device)
            else:
                # Node features: (x, y, z, intensity)
                positions = nonzero_indices.float()
                intensities = volume[nonzero_indices[:, 0], 
                                   nonzero_indices[:, 1], 
                                   nonzero_indices[:, 2]].unsqueeze(1)
                node_features = torch.cat([positions, intensities], dim=1)
                
                # Edge construction (k-nearest neighbors)
                edge_index = VoxelToGraph._build_knn_graph(positions, k_neighbors)
            
            graphs.append((node_features, edge_index))
        
        return graphs
    
    @staticmethod
    def _build_knn_graph(positions: torch.Tensor, k: int = 6) -> torch.Tensor:
        """
        K-nearest neighbor graph oluşturur
        
        Args:
            positions: Node positions (N, 3)
            k: Number of neighbors
            
        Returns:
            edge_index (2, E)
        """
        num_nodes = positions.size(0)
        
        if num_nodes <= k:
            # Tüm node'ları birbirine bağla
            src = torch.arange(num_nodes, device=positions.device).repeat_interleave(num_nodes - 1)
            dst = []
            for i in range(num_nodes):
                dst.extend([j for j in range(num_nodes) if j != i])
            dst = torch.tensor(dst, device=positions.device)
            edge_index = torch.stack([src, dst], dim=0)
        else:
            # KNN kullan
            distances = torch.cdist(positions, positions)  # (N, N)
            
            # Her node için k en yakın komşu
            _, indices = torch.topk(distances, k + 1, largest=False, dim=1)
            indices = indices[:, 1:]  # İlk eleman kendisi
            
            # Edge index oluştur
            src = torch.arange(num_nodes, device=positions.device).repeat_interleave(k)
            dst = indices.flatten()
            edge_index = torch.stack([src, dst], dim=0)
        
        return edge_index


class GCNClassifier(BaseGraphModel):
    """Graph Convolutional Network - Kidney anomaly classification"""
    
    def __init__(self, config: Dict):
        super(GCNClassifier, self).__init__(config)
        
        if not TORCH_GEOMETRIC_AVAILABLE:
            raise ImportError("torch_geometric gerekli! pip install torch-geometric")
        
        self.in_channels = config.get('node_features', 4)  # (x, y, z, intensity)
        hidden_channels = config.get('hidden_channels', 64)
        num_layers = config.get('num_layers', 3)
        
        # GCN layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        # First layer
        self.convs.append(GCNConv(self.in_channels, hidden_channels))
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
        
        self.dropout = nn.Dropout(config.get('dropout', 0.5))
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout.p),
            nn.Linear(hidden_channels // 2, self.num_classes)
        )
        
        self.print_model_info()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input 3D volume (B, C, D, H, W)
            
        Returns:
            Output logits (B, num_classes)
        """
        # Volume'den graph'a dönüştür
        graphs = VoxelToGraph.volume_to_graph(x, threshold=0.5, k_neighbors=6)
        
        outputs = []
        for node_features, edge_index in graphs:
            # GCN layers
            h = node_features
            for conv, bn in zip(self.convs, self.bns):
                h = conv(h, edge_index)
                h = bn(h)
                h = F.relu(h)
                h = self.dropout(h)
            
            # Global pooling (graph-level representation)
            graph_features = torch.mean(h, dim=0, keepdim=True)  # (1, hidden_channels)
            
            # Classification
            output = self.classifier(graph_features)
            outputs.append(output)
        
        return torch.cat(outputs, dim=0)  # (B, num_classes)
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Feature extraction"""
        graphs = VoxelToGraph.volume_to_graph(x, threshold=0.5, k_neighbors=6)
        
        features = []
        for node_features, edge_index in graphs:
            h = node_features
            for conv, bn in zip(self.convs, self.bns):
                h = conv(h, edge_index)
                h = bn(h)
                h = F.relu(h)
                h = self.dropout(h)
            
            graph_features = torch.mean(h, dim=0, keepdim=True)
            features.append(graph_features)
        
        return torch.cat(features, dim=0)
    
    def build_graph(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Graph oluşturur"""
        return VoxelToGraph.volume_to_graph(x, threshold=0.5, k_neighbors=6)


class GATClassifier(BaseGraphModel):
    """Graph Attention Network - Attention mechanism ile"""
    
    def __init__(self, config: Dict):
        super(GATClassifier, self).__init__(config)
        
        if not TORCH_GEOMETRIC_AVAILABLE:
            raise ImportError("torch_geometric gerekli!")
        
        self.in_channels = config.get('node_features', 4)
        hidden_channels = config.get('hidden_channels', 64)
        num_layers = config.get('num_layers', 3)
        num_heads = config.get('num_heads', 4)
        
        # GAT layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        # First layer
        self.convs.append(GATConv(self.in_channels, hidden_channels, heads=num_heads))
        self.bns.append(nn.BatchNorm1d(hidden_channels * num_heads))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels * num_heads, hidden_channels, heads=num_heads))
            self.bns.append(nn.BatchNorm1d(hidden_channels * num_heads))
        
        # Last layer (single head)
        self.convs.append(GATConv(hidden_channels * num_heads, hidden_channels, heads=1))
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        
        self.dropout = nn.Dropout(config.get('dropout', 0.5))
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout.p),
            nn.Linear(hidden_channels // 2, self.num_classes)
        )
        
        self.print_model_info()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        graphs = VoxelToGraph.volume_to_graph(x, threshold=0.5, k_neighbors=6)
        
        outputs = []
        for node_features, edge_index in graphs:
            h = node_features
            for conv, bn in zip(self.convs, self.bns):
                h = conv(h, edge_index)
                h = bn(h)
                h = F.relu(h)
                h = self.dropout(h)
            
            graph_features = torch.mean(h, dim=0, keepdim=True)
            output = self.classifier(graph_features)
            outputs.append(output)
        
        return torch.cat(outputs, dim=0)
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        graphs = VoxelToGraph.volume_to_graph(x, threshold=0.5, k_neighbors=6)
        
        features = []
        for node_features, edge_index in graphs:
            h = node_features
            for conv, bn in zip(self.convs, self.bns):
                h = conv(h, edge_index)
                h = bn(h)
                h = F.relu(h)
                h = self.dropout(h)
            
            graph_features = torch.mean(h, dim=0, keepdim=True)
            features.append(graph_features)
        
        return torch.cat(features, dim=0)
    
    def build_graph(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return VoxelToGraph.volume_to_graph(x, threshold=0.5, k_neighbors=6)


class GraphSAGEClassifier(BaseGraphModel):
    """GraphSAGE - Sampling and aggregation"""
    
    def __init__(self, config: Dict):
        super(GraphSAGEClassifier, self).__init__(config)
        
        if not TORCH_GEOMETRIC_AVAILABLE:
            raise ImportError("torch_geometric gerekli!")
        
        self.in_channels = config.get('node_features', 4)
        hidden_channels = config.get('hidden_channels', 64)
        num_layers = config.get('num_layers', 3)
        
        # SAGE layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        self.convs.append(SAGEConv(self.in_channels, hidden_channels))
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
        
        self.dropout = nn.Dropout(config.get('dropout', 0.5))
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout.p),
            nn.Linear(hidden_channels // 2, self.num_classes)
        )
        
        self.print_model_info()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        graphs = VoxelToGraph.volume_to_graph(x, threshold=0.5, k_neighbors=6)
        
        outputs = []
        for node_features, edge_index in graphs:
            h = node_features
            for conv, bn in zip(self.convs, self.bns):
                h = conv(h, edge_index)
                h = bn(h)
                h = F.relu(h)
                h = self.dropout(h)
            
            graph_features = torch.mean(h, dim=0, keepdim=True)
            output = self.classifier(graph_features)
            outputs.append(output)
        
        return torch.cat(outputs, dim=0)
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        graphs = VoxelToGraph.volume_to_graph(x, threshold=0.5, k_neighbors=6)
        
        features = []
        for node_features, edge_index in graphs:
            h = node_features
            for conv, bn in zip(self.convs, self.bns):
                h = conv(h, edge_index)
                h = bn(h)
                h = F.relu(h)
                h = self.dropout(h)
            
            graph_features = torch.mean(h, dim=0, keepdim=True)
            features.append(graph_features)
        
        return torch.cat(features, dim=0)
    
    def build_graph(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return VoxelToGraph.volume_to_graph(x, threshold=0.5, k_neighbors=6)
