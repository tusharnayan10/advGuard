import argparse
import random
import os

import numpy as np
import pandas as pd
import networkx as nx
import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.utils import from_networkx
from sentence_transformers import SentenceTransformer




def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_prompts(path: str, column: str = "prompt"):
    """
    Loads prompts from either:
      - .csv file with a 'prompt' column (or a provided column name), or
      - .txt file with one prompt per line.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(path)
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in {path}. Available: {df.columns.tolist()}")
        prompts = (
            df[column]
            .dropna()
            .astype(str)
            .tolist()
        )
    else:
        with open(path, "r", encoding="utf-8") as f:
            prompts = [line.strip() for line in f if line.strip()]

    print(f"Loaded {len(prompts)} prompts from {path}")
    return prompts



class GCN(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, num_classes: int):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch):
        # x: [num_nodes, in_channels]
        # edge_index: [2, num_edges]
        # batch: [num_nodes] graph id for each node
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.conv3(x, edge_index).relu()
        x = global_mean_pool(x, batch)  # [num_graphs, hidden_channels]
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)  # [num_graphs, num_classes]
        return x




def build_similarity_graph(texts, encoder, percentile=90):
    """
    Build an undirected graph where:
      - nodes = prompts
      - edges between prompts with cosine similarity >= tau
      - tau = percentile-th percentile of pairwise similarities

    Edge attribute 'label' = similarity (float),
    to match QPA's graph_checker => line graph node features.
    """
    G = nx.Graph()
    n = len(texts)
    for i in range(n):
        G.add_node(i)

    # Encode all prompts and normalize
    embs = encoder.encode(texts, convert_to_numpy=True)  # [n, d]
    norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8
    embs = embs / norms

    sims = embs @ embs.T  # cosine similarity matrix
    np.fill_diagonal(sims, 0.0)

    # Compute threshold from upper triangle (i<j)
    upper = sims[np.triu_indices(n, k=1)]
    if len(upper) == 0:
        # Only one node: no edges, just return isolated node graph
        return G

    tau = np.percentile(upper, percentile)
    # print(f"Graph tau (percentile {percentile}): {tau:.3f}")

    for i in range(n):
        for j in range(i + 1, n):
            if sims[i, j] >= tau:
                G.add_edge(i, j, label=float(sims[i, j]))

    return G



def graph_to_pyg_line_graph(G: nx.Graph, label: int):
    """
    Convert original graph G to a line graph (edges become nodes),
    with node attribute 'label' = original edge similarity,
    then to a torch_geometric Data object with:
      - x: node features (similarity values)
      - edge_index: edges of line graph
      - y: graph label (0 = benign, 1 = attack)
    """
    # If graph has fewer than 1 edge, line graph will have 0 nodes.
    # You can either skip such graphs or give them dummy features.
    if G.number_of_edges() == 0:
        # Create a dummy graph with a single node and zero feature
        x = torch.zeros((1, 1), dtype=torch.float32)
        edge_index = torch.empty((2, 0), dtype=torch.long)
        y = torch.tensor([label], dtype=torch.long)
        return Data(x=x, edge_index=edge_index, y=y)

    LG = nx.line_graph(G)
    edge_attr = nx.get_edge_attributes(G, "label")

    for node in LG.nodes:
        # node is an edge from G, e.g. (u, v)
        sim = edge_attr.get(node, 0.0)
        # Use attribute name 'label' to mirror your QPA.graph_checker
        LG.nodes[node]["label"] = float(sim)

    # group_node_attrs='all' will pack 'label' into x by default.
    pyg_graph = from_networkx(LG, group_node_attrs="all")

    # Ensure x is a float tensor
    if not hasattr(pyg_graph, "x"):
        # from_networkx might store 'label' directly; handle gracefully
        label_attr = getattr(pyg_graph, "label", None)
        if label_attr is None:
            raise ValueError("No node features found in converted PyG graph.")
        pyg_graph.x = label_attr.float()
    else:
        pyg_graph.x = pyg_graph.x.float()

    pyg_graph.y = torch.tensor([label], dtype=torch.long)
    return pyg_graph



def build_graph_dataset(benign_prompts,
                        adv_prompts,
                        encoder,
                        num_graphs=1000,
                        benign_size=30,
                        adv_in_attack=30,
                        percentile=90):
    """
    Creates a list of PyG Data graphs.
    Half benign, half attack (roughly).
    """
    graphs = []

    for i in range(num_graphs):
        is_attack = random.random() < 0.6 

        if is_attack:
            # Mix of benign + adversarial prompts
            benign_sample = random.sample(benign_prompts, min(benign_size, len(benign_prompts)))
            adv_sample = random.sample(adv_prompts, min(adv_in_attack, len(adv_prompts)))
            texts = benign_sample + adv_sample
            label = 1
        else:
            # Only benign prompts
            texts = random.sample(benign_prompts, min(benign_size + adv_in_attack, len(benign_prompts)))
            label = 0

        G = build_similarity_graph(texts, encoder, percentile=percentile)
        pyg_graph = graph_to_pyg_line_graph(G, label=label)
        graphs.append(pyg_graph)

    print(f"Built {len(graphs)} graphs (approx. half benign, half attack).")
    return graphs




def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in loader:
        batch = batch.to(DEVICE)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.batch)  # [batch_size, num_classes]
        loss = criterion(out, batch.y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch.num_graphs
        preds = out.argmax(dim=1)
        correct += (preds == batch.y).sum().item()
        total += batch.num_graphs

    avg_loss = total_loss / total if total > 0 else 0.0
    acc = correct / total if total > 0 else 0.0
    return avg_loss, acc


@torch.no_grad()
def eval_one_epoch(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in loader:
        batch = batch.to(DEVICE)
        out = model(batch.x, batch.edge_index, batch.batch)
        loss = criterion(out, batch.y)

        total_loss += loss.item() * batch.num_graphs
        preds = out.argmax(dim=1)
        correct += (preds == batch.y).sum().item()
        total += batch.num_graphs

    avg_loss = total_loss / total if total > 0 else 0.0
    acc = correct / total if total > 0 else 0.0
    return avg_loss, acc



def main():
    parser = argparse.ArgumentParser(description="Train a GCN graph-level classifier on prompt graphs.")
    parser.add_argument("--benign_path", type=str, required=True, help="Path to benign prompts (.csv or .txt).")
    parser.add_argument("--adv_path", type=str, required=True, help="Path to adversarial prompts (.csv or .txt).")
    parser.add_argument("--benign_column", type=str, default="prompt", help="Column name for benign CSV.")
    parser.add_argument("--adv_column", type=str, default="prompt", help="Column name for adv CSV.")
    parser.add_argument("--num_graphs", type=int, default=400, help="Total # of graphs to generate.")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for loader.")
    parser.add_argument("--hidden_channels", type=int, default=32, help="GCN hidden dimension.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--model_out", type=str, default="graph_gcn_model.pt", help="Output model path.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--similarity_percentile", type=float, default=90.0,
                        help="Percentile used to threshold similarities into edges.")
    args = parser.parse_args()

    set_seed(args.seed)
    print(f"Using device: {DEVICE}")

    # 1) Load prompts
    benign_prompts = load_prompts(args.benign_path, column=args.benign_column)
    adv_prompts = load_prompts(args.adv_path, column=args.adv_column)

    if len(benign_prompts) < 999 or len(adv_prompts) < 999:
        raise ValueError("Need at least ~999 benign and ~999 adversarial prompts for meaningful graphs.")

    # 2) Initialize text encoder
    print("Loading SentenceTransformer encoder (all-MiniLM-L6-v2)...")
    encoder = SentenceTransformer("all-MiniLM-L6-v2")

    graphs = build_graph_dataset(
        benign_prompts=benign_prompts,
        adv_prompts=adv_prompts,
        encoder=encoder,
        num_graphs=args.num_graphs,
        benign_size=30,
        adv_in_attack=30,
        percentile=args.similarity_percentile,
    )

    random.shuffle(graphs)
    split = int(0.85 * len(graphs))
    train_graphs = graphs[:split]
    val_graphs = graphs[split:]

    train_loader = DataLoader(train_graphs, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=args.batch_size, shuffle=False)

    in_channels = train_graphs[0].x.size(-1)
    num_classes = 2

    model = GCN(in_channels=in_channels,
                hidden_channels=args.hidden_channels,
                num_classes=num_classes).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    print(f"Model in_channels={in_channels}, hidden={args.hidden_channels}, num_classes={num_classes}")
    print(f"Training on {len(train_graphs)} graphs, validating on {len(val_graphs)} graphs.")

    best_val_acc = 0.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc = eval_one_epoch(model, val_loader, criterion)

        print(f"Epoch {epoch:02d}: "
              f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
              f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict().copy()

    if best_state is not None:
        model.load_state_dict(best_state)

    torch.save(model, args.model_out)
    print(f"Saved best model (val_acc={best_val_acc:.4f}) to: {args.model_out}")


if __name__ == "__main__":
    main()