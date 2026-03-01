#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "torch>=2.1",
#     "torch-geometric>=2.4",
# ]
# ///
"""Graph Transformer teacher model for PixelFlow compiler cost prediction.

Phase 1 (value head): GATv2 on expression trees for cost prediction.
Phase 2 (actor-critic OPD): Hetero-GNN critic on bipartite e-graphs,
  distilling into a 264-param NNUE actor for rule selection.

Usage:
    uv run graph_teacher.py train --data judge_training.jsonl
    uv run graph_teacher.py eval --checkpoint graph_teacher_v1.pt --data judge_training.jsonl
    uv run graph_teacher.py export-distill --checkpoint graph_teacher_v1.pt --data judge_training.jsonl
    uv run graph_teacher.py train-policy --data guide_training.jsonl
    uv run graph_teacher.py eval-policy --data guide_training.jsonl --teacher-checkpoint ... --student-checkpoint ...
    uv run graph_teacher.py export-guide --student-checkpoint guide_student.bin
    uv run graph_teacher.py online-train --judge-weights ../data/judge.bin --rounds 20
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# Deep paren nesting in expressions (e.g. 228-node trees) requires more stack
sys.setrecursionlimit(10_000)

import copy
import struct
import subprocess

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.data import Batch, Data, HeteroData
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv, global_mean_pool, HeteroConv

# ═══════════════════════════════════════════════════════════════════════════════
# OpKind mapping (mirrors pixelflow-ir/src/kind.rs)
# ═══════════════════════════════════════════════════════════════════════════════

OPKIND: dict[str, int] = {
    "Var": 0, "Const": 1,
    "Add": 2, "Sub": 3, "Mul": 4, "Div": 5, "Neg": 6,
    "Sqrt": 7, "Rsqrt": 8, "Abs": 9,
    "Min": 10, "Max": 11, "MulAdd": 12,
    "Recip": 13, "Floor": 14, "Ceil": 15, "Round": 16, "Fract": 17,
    "Sin": 18, "Cos": 19, "Tan": 20, "Asin": 21, "Acos": 22, "Atan": 23,
    "Exp": 24, "Exp2": 25, "Ln": 26, "Log2": 27, "Log10": 28,
    "Atan2": 29, "Pow": 30, "Hypot": 31,
    "Lt": 32, "Le": 33, "Gt": 34, "Ge": 35, "Eq": 36, "Ne": 37,
    "Select": 38, "Clamp": 39, "Tuple": 40,
}
NUM_OPS = 41
NODE_FEAT_DIM = NUM_OPS + 3  # one-hot(41) + is_leaf + depth_norm + subtree_size_norm

# Lowercase name → discriminant (matches OpKind::name() in kind.rs)
OPKIND_LOWER: dict[str, int] = {
    name.lower().replace("muladd", "mul_add"): idx
    for name, idx in OPKIND.items()
}
# Fix the ones where lowercase differs from OpKind::name()
OPKIND_LOWER["mul_add"] = OPKIND["MulAdd"]

# E-node feature dimensions for bipartite e-graph
ENODE_FEAT_DIM = NUM_OPS + 2  # one-hot(41) + is_leaf(1) + arity_norm(1)
ECLASS_FEAT_DIM = 2  # class_size_norm(1) + depth_norm(1)

# GuideNnue constants (mirror Rust RULE_FEATURE_COUNT / GUIDE_HIDDEN_DIM)
# Features: [rule_idx, classes, nodes, match_rate, epochs_since_match,
#            epoch, max_epochs, budget_frac, node_headroom,
#            eval_budget_frac, eval_headroom]
GUIDE_INPUT_DIM = 11
GUIDE_HIDDEN_DIM = 32
GUIDE_PARAM_COUNT = GUIDE_INPUT_DIM * GUIDE_HIDDEN_DIM + GUIDE_HIDDEN_DIM + GUIDE_HIDDEN_DIM + 1  # 417

# ═══════════════════════════════════════════════════════════════════════════════
# ExprNnue mask architecture constants (mirror factored.rs)
# ═══════════════════════════════════════════════════════════════════════════════

EXPR_K = 32                                       # Embedding dimension per op
EXPR_MAX_ARITY = 3                                # Max children per node
EXPR_MAX_DEPTH = 192                              # Max effective depth for PE
EXPR_INPUT_DIM = 4 * EXPR_K + 2                   # 130 (dual acc + scalars)
EXPR_HIDDEN_DIM = 64                              # Backbone hidden
EXPR_EMBED_DIM = 32                               # Shared projection dim
EXPR_MLP_HIDDEN = 16                              # Private MLP hidden
EXPR_MASK_INPUT_DIM = EXPR_EMBED_DIM               # 24 (expr_embed directly, value_pred removed)
EXPR_RULE_CONCAT_DIM = 4 * EXPR_EMBED_DIM         # 96 (4-way concat)
EXPR_MASK_MAX_RULES = 1024                        # Max rules in mask arch

# Mask-specific param count for export (mask_mlp + interaction + bias)
MASK_PARAM_COUNT = (
    EXPR_MASK_INPUT_DIM * EXPR_MLP_HIDDEN         # mask_mlp_w1: 24*16=384
    + EXPR_MLP_HIDDEN                              # mask_mlp_b1: 16
    + EXPR_MLP_HIDDEN * EXPR_EMBED_DIM            # mask_mlp_w2: 16*24=384
    + EXPR_EMBED_DIM                               # mask_mlp_b2: 24
    + EXPR_EMBED_DIM * EXPR_EMBED_DIM             # interaction: 24*24=576
    + EXPR_MASK_MAX_RULES                          # mask_rule_bias: 1024
)  # = 2424 params = 9696 bytes + 4 magic = 9700


def _build_depth_pe_table() -> torch.Tensor:
    """Build sinusoidal PE table matching factored.rs DEPTH_PE.

    PE[d][2i]   = sin(d / 10000^(2i/K))
    PE[d][2i+1] = cos(d / 10000^(2i/K))
    """
    table = torch.zeros(EXPR_MAX_DEPTH, EXPR_K)
    for depth in range(EXPR_MAX_DEPTH):
        for dim in range(EXPR_K):
            dim_pair = 2 * (dim // 2)
            exponent = dim_pair / EXPR_K
            divisor = 10000.0 ** exponent
            angle = depth / divisor
            if dim % 2 == 0:
                table[depth, dim] = math.sin(angle)
            else:
                table[depth, dim] = math.cos(angle)
    return table


DEPTH_PE_TABLE: torch.Tensor = _build_depth_pe_table()


def build_edge_accumulator_batch(
    edge_tuples_batch: list[list[tuple[int, int, int]]],
    op_embeddings: nn.Embedding,
) -> torch.Tensor:
    """Build EdgeAccumulator for a batch of expressions.

    Mirrors factored.rs EdgeAccumulator::from_expr(). Vectorized over all edges
    in the batch, then scatter-added into per-sample accumulators.

    Args:
        edge_tuples_batch: list of list of (parent_op, child_op, effective_depth)
            where effective_depth = tree_depth * MAX_ARITY + child_index
        op_embeddings: nn.Embedding(NUM_OPS, EXPR_K)
    Returns: [B, 4*K+2] tensor (dual accumulator + edge_count + node_count)
    """
    K = EXPR_K
    B = len(edge_tuples_batch)

    all_parent_ops: list[int] = []
    all_child_ops: list[int] = []
    all_depths: list[int] = []
    all_batch_idx: list[int] = []

    for b, edges in enumerate(edge_tuples_batch):
        for parent_op, child_op, eff_depth in edges:
            all_parent_ops.append(parent_op)
            all_child_ops.append(child_op)
            all_depths.append(min(eff_depth, EXPR_MAX_DEPTH - 1))
            all_batch_idx.append(b)

    if not all_parent_ops:
        acc = torch.zeros(B, 4 * K + 2)
        for b in range(B):
            acc[b, 4 * K + 1] = 1.0  # at least 1 node
        return acc

    parent_ops_t = torch.tensor(all_parent_ops, dtype=torch.long)
    child_ops_t = torch.tensor(all_child_ops, dtype=torch.long)
    depths_t = torch.tensor(all_depths, dtype=torch.long)
    batch_idx_t = torch.tensor(all_batch_idx, dtype=torch.long)

    emb_w = op_embeddings.weight.detach()  # [NUM_OPS, K]

    p_emb = emb_w[parent_ops_t]    # [E_total, K]
    c_emb = emb_w[child_ops_t]     # [E_total, K]
    pe = DEPTH_PE_TABLE[depths_t]   # [E_total, K]

    # Complex multiply for depth-encoded half.
    # Each pair (2f, 2f+1) = (real, imaginary).
    # PE: sin at even, cos at odd.
    # Complex: (emb_re + j*emb_im) * (cos + j*sin)
    #   re_out = emb_re*cos - emb_im*sin
    #   im_out = emb_re*sin + emb_im*cos
    p_re, p_im = p_emb[:, 0::2], p_emb[:, 1::2]
    c_re, c_im = c_emb[:, 0::2], c_emb[:, 1::2]
    sin_d, cos_d = pe[:, 0::2], pe[:, 1::2]

    p_depth = torch.zeros_like(p_emb)
    p_depth[:, 0::2] = p_re * cos_d - p_im * sin_d
    p_depth[:, 1::2] = p_re * sin_d + p_im * cos_d

    c_depth = torch.zeros_like(c_emb)
    c_depth[:, 0::2] = c_re * cos_d - c_im * sin_d
    c_depth[:, 1::2] = c_re * sin_d + c_im * cos_d

    # Scatter-add vector features: [flat_p | flat_c | depth_p | depth_c]
    features = torch.cat([p_emb, c_emb, p_depth, c_depth], dim=1)  # [E_total, 4K]
    acc_vectors = torch.zeros(B, 4 * K)
    acc_vectors.index_add_(0, batch_idx_t, features)

    # Append scalar features: edge_count, node_count
    acc = torch.zeros(B, 4 * K + 2)
    acc[:, :4 * K] = acc_vectors
    edge_counts = torch.zeros(B)
    edge_counts.index_add_(0, batch_idx_t, torch.ones(len(all_parent_ops)))
    acc[:, 4 * K] = edge_counts
    acc[:, 4 * K + 1] = edge_counts + 1  # node_count ≈ edge_count + 1

    return acc


# String tokens → OpKind names for method calls and binary operators
METHOD_TO_OP: dict[str, str] = {
    "abs": "Abs",
    "sqrt": "Sqrt",
    "rsqrt": "Rsqrt",
    "min": "Min",
    "max": "Max",
    "mul_add": "MulAdd",
    "sin": "Sin",
    "cos": "Cos",
    "tan": "Tan",
    "exp": "Exp",
    "exp2": "Exp2",
    "ln": "Ln",
    "log2": "Log2",
    "log10": "Log10",
    "recip": "Recip",
    "floor": "Floor",
    "ceil": "Ceil",
    "round": "Round",
    "fract": "Fract",
    "asin": "Asin",
    "acos": "Acos",
    "atan": "Atan",
    "atan2": "Atan2",
    "pow": "Pow",
    "hypot": "Hypot",
    "clamp": "Clamp",
}
BINOP_TO_OP: dict[str, str] = {
    "+": "Add",
    "-": "Sub",
    "*": "Mul",
    "/": "Div",
}


# ═══════════════════════════════════════════════════════════════════════════════
# Expression AST
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ASTNode:
    """A node in the expression AST."""
    op: str  # OpKind name (e.g., "Add", "Var", "Const")
    children: list[ASTNode] = field(default_factory=list)
    value: float | str | None = None  # float for Const, str for Var (X/Y/Z)


# ═══════════════════════════════════════════════════════════════════════════════
# Tokenizer
# ═══════════════════════════════════════════════════════════════════════════════

# Token types: '(', ')', ',', '.', '+', '-', '*', '/', 'FLOAT', 'IDENT', 'EOF'
Token = tuple[str, str | float | None]


def tokenize(s: str) -> list[Token]:
    """Tokenize an expression string into (type, value) pairs.

    Crashes on unexpected characters — no silent fallthrough.
    """
    tokens: list[Token] = []
    i = 0
    n = len(s)
    while i < n:
        c = s[i]
        if c.isspace():
            i += 1
        elif c in "(),":
            tokens.append((c, c))
            i += 1
        elif c == ".":
            tokens.append((".", "."))
            i += 1
        elif c in "+-*/":
            tokens.append((c, c))
            i += 1
        elif c.isdigit():
            j = i
            while j < n and s[j].isdigit():
                j += 1
            if j < n and s[j] == ".":
                j += 1
                while j < n and s[j].isdigit():
                    j += 1
            tokens.append(("FLOAT", float(s[i:j])))
            i = j
        elif c.isalpha() or c == "_":
            j = i
            while j < n and (s[j].isalnum() or s[j] == "_"):
                j += 1
            tokens.append(("IDENT", s[i:j]))
            i = j
        else:
            raise ValueError(
                f"Unexpected character {c!r} at position {i} "
                f"in: ...{s[max(0, i - 20):i + 20]}..."
            )
    tokens.append(("EOF", None))
    return tokens


# ═══════════════════════════════════════════════════════════════════════════════
# Recursive descent parser
# ═══════════════════════════════════════════════════════════════════════════════

class Parser:
    """Recursive descent parser for PixelFlow expression strings.

    Grammar:
        expr       := unary postfix*
        unary      := '-' unary | atom
        atom       := '(' paren_body ')' | FLOAT | VAR
        paren_body := expr (binop expr)?
        postfix    := '.' IDENT '(' arg_list? ')'
        arg_list   := expr (',' expr)*
        binop      := '+' | '-' | '*' | '/'
        VAR        := 'X' | 'Y' | 'Z'
    """

    def __init__(self, tokens: list[Token]):
        self.tokens = tokens
        self.pos = 0

    def peek(self) -> Token:
        return self.tokens[self.pos]

    def advance(self) -> Token:
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def expect(self, kind: str) -> Token:
        tok = self.advance()
        if tok[0] != kind:
            ctx = self.tokens[max(0, self.pos - 3):self.pos + 2]
            raise ValueError(
                f"Expected {kind!r}, got {tok!r} at pos {self.pos - 1} "
                f"(context: {ctx})"
            )
        return tok

    def parse(self) -> ASTNode:
        result = self.parse_expr()
        if self.peek()[0] != "EOF":
            raise ValueError(
                f"Trailing tokens after expression: {self.peek()!r} "
                f"at pos {self.pos}"
            )
        return result

    def parse_expr(self) -> ASTNode:
        """Parse an expression with optional postfix method calls."""
        node = self.parse_unary()
        # Postfix method calls: .abs(), .sqrt(), .min(...), .mul_add(...), etc.
        while self.peek()[0] == ".":
            self.advance()  # consume '.'
            method_tok = self.expect("IDENT")
            method_name = method_tok[1]
            op = METHOD_TO_OP.get(method_name)
            if op is None:
                raise ValueError(f"Unknown method: .{method_name}()")
            self.expect("(")
            args: list[ASTNode] = []
            if self.peek()[0] != ")":
                args.append(self.parse_expr())
                while self.peek()[0] == ",":
                    self.advance()  # consume ','
                    args.append(self.parse_expr())
            self.expect(")")
            # Receiver becomes first child: x.min(y) → Min(x, y)
            node = ASTNode(op, [node] + args)
        return node

    def parse_unary(self) -> ASTNode:
        """Parse unary negation prefix."""
        if self.peek()[0] == "-":
            self.advance()
            child = self.parse_unary()
            return ASTNode("Neg", [child])
        return self.parse_atom()

    def parse_atom(self) -> ASTNode:
        """Parse an atomic expression: parenthesized, float literal, or variable."""
        tok = self.peek()
        if tok[0] == "(":
            self.advance()
            node = self.parse_paren_body()
            self.expect(")")
            return node
        elif tok[0] == "FLOAT":
            self.advance()
            return ASTNode("Const", value=tok[1])
        elif tok[0] == "IDENT" and tok[1] in ("X", "Y", "Z", "W"):
            self.advance()
            return ASTNode("Var", value=tok[1])
        else:
            raise ValueError(
                f"Unexpected token in atom: {tok!r} at pos {self.pos}"
            )

    def parse_paren_body(self) -> ASTNode:
        """Parse the inside of parentheses: expr or expr binop expr."""
        lhs = self.parse_expr()
        tok = self.peek()
        if tok[0] in BINOP_TO_OP:
            self.advance()
            rhs = self.parse_expr()
            return ASTNode(BINOP_TO_OP[tok[0]], [lhs, rhs])
        return lhs


def parse_expression(s: str) -> ASTNode:
    """Parse an expression string into an AST. Crashes on malformed input."""
    tokens = tokenize(s)
    parser = Parser(tokens)
    return parser.parse()


# ═══════════════════════════════════════════════════════════════════════════════
# AST → PyG Data
# ═══════════════════════════════════════════════════════════════════════════════

def ast_to_data(root: ASTNode) -> Data:
    """Convert an AST to a PyG Data object with node and edge features.

    Node features: [one_hot(op, 42) | is_leaf | depth_norm | subtree_size_norm]
    Edge features: [child_index]  (bidirectional edges)
    """
    # Collect all nodes via DFS
    node_info: list[tuple[int, float, int, int] | None] = []
    src_list: list[int] = []
    dst_list: list[int] = []
    edge_attrs: list[float] = []

    def visit(node: ASTNode, depth: int) -> tuple[int, int]:
        """DFS visit. Returns (node_id, subtree_size)."""
        node_id = len(node_info)
        node_info.append(None)  # placeholder

        op_idx = OPKIND.get(node.op)
        if op_idx is None:
            raise ValueError(f"Unknown op in AST: {node.op!r}")

        subtree_size = 1
        for child_idx, child in enumerate(node.children):
            child_id, child_sub = visit(child, depth + 1)
            # Bidirectional: parent→child and child→parent
            src_list.append(node_id)
            dst_list.append(child_id)
            edge_attrs.append(float(child_idx))
            src_list.append(child_id)
            dst_list.append(node_id)
            edge_attrs.append(float(child_idx))
            subtree_size += child_sub

        is_leaf = 1.0 if not node.children else 0.0
        node_info[node_id] = (op_idx, is_leaf, depth, subtree_size)
        return node_id, subtree_size

    visit(root, 0)

    total_nodes = len(node_info)
    max_depth = max(info[2] for info in node_info) if node_info else 0

    # Build node feature matrix
    x = torch.zeros(total_nodes, NODE_FEAT_DIM)
    for i, info in enumerate(node_info):
        op_idx, is_leaf, depth, subtree_size = info  # type: ignore[misc]
        x[i, op_idx] = 1.0  # one-hot
        x[i, NUM_OPS] = is_leaf
        x[i, NUM_OPS + 1] = depth / max(max_depth, 1)
        x[i, NUM_OPS + 2] = subtree_size / max(total_nodes, 1)

    # Build edge tensors
    if src_list:
        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float32).unsqueeze(1)
    else:
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        edge_attr = torch.zeros(0, 1, dtype=torch.float32)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_dataset(path: Path) -> tuple[list[Data], list[str]]:
    """Load judge_training.jsonl, parse every expression, return (graphs, names).

    Crashes on any parse failure — no silent skipping.
    """
    if not path.exists():
        raise FileNotFoundError(f"Training data not found: {path}")

    graphs: list[Data] = []
    names: list[str] = []

    with open(path) as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            name = record["name"]
            expr_str = record["expression"]
            timing_ns = record["timing_ns"]

            if timing_ns < 0:
                raise ValueError(
                    f"Line {line_no} ({name}): timing_ns must be non-negative, "
                    f"got {timing_ns}"
                )
            # Floor at 0.5ns to avoid log(0) = -inf
            timing_ns = max(timing_ns, 0.5)

            ast = parse_expression(expr_str)
            data = ast_to_data(ast)
            data.y = torch.tensor([math.log(timing_ns)], dtype=torch.float32)
            data.timing_ns = timing_ns

            graphs.append(data)
            names.append(name)

            if line_no % 2000 == 0:
                print(f"  Parsed {line_no} expressions...", file=sys.stderr)

    print(f"Loaded {len(graphs)} graphs from {path}", file=sys.stderr)
    return graphs, names


# ═══════════════════════════════════════════════════════════════════════════════
# Model
# ═══════════════════════════════════════════════════════════════════════════════

class ExprGraphNet(nn.Module):
    """GATv2-based graph neural network for expression cost prediction.

    Architecture:
        input_proj(in_dim → hidden)
        → 4× [GATv2Conv + residual + LayerNorm + dropout]
        → global_mean_pool
        → value_head(hidden → 64 → 1)
    """

    def __init__(
        self,
        in_dim: int = NODE_FEAT_DIM,
        hidden: int = 128,
        heads: int = 4,
        layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        if hidden % heads != 0:
            raise ValueError(
                f"hidden ({hidden}) must be divisible by heads ({heads})"
            )
        self.input_proj = nn.Linear(in_dim, hidden)

        self.convs = nn.ModuleList([
            GATv2Conv(
                hidden,
                hidden // heads,
                heads=heads,
                edge_dim=1,
                add_self_loops=False,
                dropout=dropout,
            )
            for _ in range(layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden) for _ in range(layers)
        ])
        self.drop = nn.Dropout(dropout)

        self.value_head = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, data: Data) -> torch.Tensor:
        x = self.input_proj(data.x)

        for conv, norm in zip(self.convs, self.norms):
            residual = x
            x = conv(x, data.edge_index, data.edge_attr)
            x = self.drop(x)
            x = norm(residual + x)

        graph_embed = global_mean_pool(x, data.batch)
        return self.value_head(graph_embed).squeeze(-1)


# ═══════════════════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════════════════

def spearman_rho(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Spearman rank correlation coefficient (no scipy dependency).

    Uses the 1 - 6*sum(d^2)/(n*(n^2-1)) formula, exact when no ties exist.
    Approximation when ties are present; sufficient for diagnostic display.
    """
    pred = pred.detach().cpu().float()
    target = target.detach().cpu().float()
    n = len(pred)
    if n < 2:
        return 0.0

    def _rank(x: torch.Tensor) -> torch.Tensor:
        order = x.argsort()
        ranks = torch.empty_like(x)
        ranks[order] = torch.arange(n, dtype=x.dtype)
        return ranks

    r_pred = _rank(pred)
    r_target = _rank(target)
    d = r_pred - r_target
    return float(1.0 - 6.0 * (d * d).sum().item() / (n * (n * n - 1)))


# ═══════════════════════════════════════════════════════════════════════════════
# E-Graph → PyG HeteroData (bipartite)
# ═══════════════════════════════════════════════════════════════════════════════

def egraph_to_data(egraph: dict, record_id: str = "?") -> HeteroData:
    """Convert an e-graph dict to a PyG HeteroData object.

    Bipartite graph with two vertex types:
    - enode: one-hot op(42) + is_leaf(1) + arity_norm(1) = 44 dims
    - eclass: class_size_norm(1) + depth_norm(1) = 2 dims

    Three edge types:
    1. (enode, member_of, eclass) — each e-node belongs to one class
    2. (enode, child, eclass) — each e-node's args point to child classes
    3. (eclass, contains, enode) — reverse of membership

    Args:
        egraph: dict with "nodes" key containing the e-graph node list
        record_id: identifier for error messages
    """
    nodes = egraph["nodes"]

    if not nodes:
        raise ValueError(f"Empty e-graph in record {record_id}")

    # Collect canonical class IDs
    class_set: set[int] = set()
    for n in nodes:
        class_set.add(n["class"])
        if n.get("children"):
            for c in n["children"]:
                class_set.add(c)
    class_list = sorted(class_set)
    class_to_idx = {c: i for i, c in enumerate(class_list)}
    num_classes = len(class_list)
    num_nodes = len(nodes)

    # Count nodes per class for class features
    class_sizes: dict[int, int] = {}
    for n in nodes:
        cls = n["class"]
        class_sizes[cls] = class_sizes.get(cls, 0) + 1

    # E-node features
    enode_x = torch.zeros(num_nodes, ENODE_FEAT_DIM)
    # Edge lists
    member_of_src: list[int] = []
    member_of_dst: list[int] = []
    child_src: list[int] = []
    child_dst: list[int] = []
    child_attr: list[float] = []

    for node_idx, n in enumerate(nodes):
        kind = n["kind"]
        op_idx = OPKIND_LOWER.get(kind)
        if op_idx is None:
            raise ValueError(f"Unknown e-node kind: {kind!r}")
        enode_x[node_idx, op_idx] = 1.0
        is_leaf = kind in ("var", "const") or not n.get("children")
        enode_x[node_idx, NUM_OPS] = 1.0 if is_leaf else 0.0
        children = n.get("children", [])
        enode_x[node_idx, NUM_OPS + 1] = len(children) / 3.0  # arity_norm (max arity ~3)

        # member_of edge: enode → eclass
        cls_idx = class_to_idx[n["class"]]
        member_of_src.append(node_idx)
        member_of_dst.append(cls_idx)

        # child edges: enode → child eclass
        for slot, child_class in enumerate(children):
            if child_class in class_to_idx:
                child_src.append(node_idx)
                child_dst.append(class_to_idx[child_class])
                child_attr.append(float(slot))

    # E-class features
    max_class_size = max(class_sizes.values()) if class_sizes else 1
    eclass_x = torch.zeros(num_classes, ECLASS_FEAT_DIM)
    for cls, idx in class_to_idx.items():
        size = class_sizes.get(cls, 0)
        eclass_x[idx, 0] = size / max(max_class_size, 1)
        eclass_x[idx, 1] = 0.0  # depth_norm (not available from snapshot, kept for API)

    # Build HeteroData
    data = HeteroData()
    data["enode"].x = enode_x
    data["eclass"].x = eclass_x

    # member_of edges
    if member_of_src:
        data["enode", "member_of", "eclass"].edge_index = torch.tensor(
            [member_of_src, member_of_dst], dtype=torch.long
        )
    else:
        data["enode", "member_of", "eclass"].edge_index = torch.zeros(2, 0, dtype=torch.long)

    # child edges with slot attribute
    if child_src:
        data["enode", "child", "eclass"].edge_index = torch.tensor(
            [child_src, child_dst], dtype=torch.long
        )
        data["enode", "child", "eclass"].edge_attr = torch.tensor(
            child_attr, dtype=torch.float32
        ).unsqueeze(1)
    else:
        data["enode", "child", "eclass"].edge_index = torch.zeros(2, 0, dtype=torch.long)
        data["enode", "child", "eclass"].edge_attr = torch.zeros(0, 1, dtype=torch.float32)

    # contains edges (reverse of member_of)
    if member_of_src:
        data["eclass", "contains", "enode"].edge_index = torch.tensor(
            [member_of_dst, member_of_src], dtype=torch.long
        )
    else:
        data["eclass", "contains", "enode"].edge_index = torch.zeros(2, 0, dtype=torch.long)

    return data


# ═══════════════════════════════════════════════════════════════════════════════
# Policy Dataset Loading
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class PolicySample:
    """One rule decision from a replay buffer trajectory."""
    graph_idx: int  # Index into shared graphs list
    rule_idx: int
    applied: bool
    label: float  # pre-computed: improvement / waste / neutral
    edge_tuples: list = field(default_factory=list)  # [(parent_op, child_op, eff_depth), ...]
    mask_score: float = 0.0  # raw bilinear score from ExprNnue mask


def _graph_cache_path(data_path: Path) -> Path:
    """Return the disk cache path for a given data file."""
    return data_path.with_suffix(".graph_cache.pt")


def load_policy_dataset(path: Path) -> tuple[list[PolicySample], list[HeteroData]]:
    """Load replay buffer JSONL and convert to PolicySamples + unique graphs.

    Returns (samples, graphs) where each sample's graph_idx indexes into graphs.
    Graphs are cached to disk (alongside the data file, .graph_cache.pt) so
    subsequent runs skip the expensive egraph_to_data() conversion.
    Crashes on malformed data — no silent skipping.
    """
    if not path.exists():
        raise FileNotFoundError(f"Replay buffer not found: {path}")

    cache_path = _graph_cache_path(path)

    # Load existing cache: dict mapping trajectory_id -> HeteroData
    graph_cache: dict[str, HeteroData] = {}
    if cache_path.exists():
        graph_cache = torch.load(cache_path, weights_only=False)
        print(f"Loaded {len(graph_cache)} cached graphs from {cache_path}",
              file=sys.stderr)

    samples: list[PolicySample] = []
    graphs: list[HeteroData] = []
    line_count = 0
    new_graphs_built = 0
    skipped = 0

    with open(path) as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)

            # Build or retrieve graph from FINAL egraph (one per trajectory)
            traj_id = record.get("trajectory_id", f"line_{line_no}")
            if traj_id not in graph_cache:
                graph_cache[traj_id] = egraph_to_data(record["final_egraph"], record_id=traj_id)
                new_graphs_built += 1
            egraph_data = graph_cache[traj_id]
            graph_idx = len(graphs)
            graphs.append(egraph_data)

            initial_cost = record["initial_cost"]
            final_cost = record["final_cost"]
            if initial_cost is None or final_cost is None:
                skipped += 1
                graphs.pop()  # undo the append above
                continue
            eval_budget = record["eval_budget"]
            # Prefer ground-truth JIT timing over judge-predicted cost when available.
            # real_final_ns is set by uncertainty-sampled JIT benchmarking in collect_guide_data.
            real_final_ns = record.get("real_final_ns")
            if real_final_ns is not None:
                traj_improvement = (initial_cost - real_final_ns) / max(initial_cost, 1e-3)
            else:
                traj_improvement = (initial_cost - final_cost) / max(initial_cost, 1e-3)

            for decision in record["decisions"]:
                applied = decision["applied"]
                matched = decision["matched"]
                evals = decision["evals"]

                # Edge tuples for EdgeAccumulator (new format; empty for old data)
                raw_edges = decision.get("epoch_edges", [])
                edge_tuples = [(e[0], e[1], e[2]) for e in raw_edges]

                # Raw bilinear score from ExprNnue mask (new format)
                mask_score = decision.get("mask_score", 0.0)

                if applied and matched:
                    label = traj_improvement
                elif applied and not matched:
                    label = -(evals / max(eval_budget, 1))  # wasted budget
                else:
                    label = 0.0  # not applied: neutral

                samples.append(PolicySample(
                    graph_idx=graph_idx,
                    rule_idx=decision["rule_idx"],
                    applied=applied,
                    label=label,
                    edge_tuples=edge_tuples,
                    mask_score=mask_score,
                ))

            line_count += 1
            if line_count % 1000 == 0:
                print(f"  Loaded {line_count} trajectories ({len(samples)} samples)...",
                      file=sys.stderr)

    # Persist any newly built graphs so subsequent runs skip conversion
    if new_graphs_built > 0:
        torch.save(graph_cache, cache_path)
        print(f"Built {new_graphs_built} new graphs, cache saved to {cache_path}",
              file=sys.stderr)
    else:
        print(f"All {len(graphs)} graphs loaded from cache (0 built fresh)",
              file=sys.stderr)

    skipped_msg = f", {skipped} skipped (null cost)" if skipped else ""
    print(f"Loaded {len(samples)} policy samples from {line_count} trajectories "
          f"({len(graphs)} unique graphs){skipped_msg} in {path}",
          file=sys.stderr)
    return samples, graphs


# ═══════════════════════════════════════════════════════════════════════════════
# EGraphNet (Critic)
# ═══════════════════════════════════════════════════════════════════════════════

class EGraphNet(nn.Module):
    """Heterogeneous GATv2 on bipartite e-graph.

    Message passing alternates between enode→eclass and eclass→enode,
    naturally capturing equivalence and composition.
    """

    def __init__(
        self,
        node_dim: int = ENODE_FEAT_DIM,
        class_dim: int = ECLASS_FEAT_DIM,
        hidden: int = 128,
        heads: int = 4,
        layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        if hidden % heads != 0:
            raise ValueError(f"hidden ({hidden}) must be divisible by heads ({heads})")

        self.node_proj = nn.Linear(node_dim, hidden)
        self.class_proj = nn.Linear(class_dim, hidden)

        self.n2c_convs = nn.ModuleList()
        self.c2n_convs = nn.ModuleList()
        self.node_norms = nn.ModuleList()
        self.class_norms = nn.ModuleList()

        for _ in range(layers):
            # Node → Class: membership + child edges
            self.n2c_convs.append(GATv2Conv(
                hidden, hidden // heads, heads=heads,
                add_self_loops=False, dropout=dropout,
            ))
            # Class → Node: contains edges
            self.c2n_convs.append(GATv2Conv(
                hidden, hidden // heads, heads=heads,
                add_self_loops=False, dropout=dropout,
            ))
            self.node_norms.append(nn.LayerNorm(hidden))
            self.class_norms.append(nn.LayerNorm(hidden))

        self.drop = nn.Dropout(dropout)
        self._hidden = hidden

    def encode(self, data: HeteroData) -> torch.Tensor:
        """Encode e-graph into a fixed-size embedding. Returns [B, hidden]."""
        x_node = self.node_proj(data["enode"].x)
        x_class = self.class_proj(data["eclass"].x)

        for n2c, c2n, nn_, cn_ in zip(
            self.n2c_convs, self.c2n_convs, self.node_norms, self.class_norms
        ):
            # Node → Class via member_of edges
            member_edge = data["enode", "member_of", "eclass"].edge_index
            x_class = cn_(x_class + self.drop(n2c(
                (x_node, x_class), member_edge
            )))
            # Class → Node via contains edges
            contains_edge = data["eclass", "contains", "enode"].edge_index
            x_node = nn_(x_node + self.drop(c2n(
                (x_class, x_node), contains_edge
            )))

        # Pool over e-class embeddings (classes are the semantic units)
        batch = data["eclass"].batch if hasattr(data["eclass"], "batch") else None
        if batch is not None:
            return global_mean_pool(x_class, batch)
        return x_class.mean(dim=0, keepdim=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PolicyHead (bilinear critic scoring)
# ═══════════════════════════════════════════════════════════════════════════════

class PolicyHead(nn.Module):
    """Bilinear scoring: graph_embed × rule_embed → scalar logit."""

    def __init__(self, graph_dim: int = 128, num_rules: int = 24, rule_dim: int = 64):
        super().__init__()
        self.rule_embed = nn.Embedding(num_rules, rule_dim)
        self.graph_proj = nn.Sequential(
            nn.Linear(graph_dim, rule_dim),
            nn.ReLU(),
            nn.Linear(rule_dim, rule_dim),
        )

    def forward(self, graph_embed: torch.Tensor, rule_indices: torch.Tensor) -> torch.Tensor:
        """Score (graph, rule) pairs. Returns [B] logits."""
        projected = self.graph_proj(graph_embed)
        rule_emb = self.rule_embed(rule_indices)
        return (projected * rule_emb).sum(-1)


# ═══════════════════════════════════════════════════════════════════════════════
# NNUEStudent (Actor, mirrors ExprNnue bilinear mask architecture)
# ═══════════════════════════════════════════════════════════════════════════════

class NNUEStudent(nn.Module):
    """Mirrors ExprNnue bilinear mask architecture from factored.rs.

    Backbone (frozen from judge): EdgeAccumulator → hidden → expr_embed
    Value head (frozen from judge): expr_embed → cost
    Mask head (trained): [expr_embed, value] → mask_features
    Bilinear scoring: mask_features @ interaction @ rule_embed + bias
    """

    def __init__(self, num_rules: int = 64):
        super().__init__()
        K = EXPR_K

        # Shared backbone (frozen during mask training)
        self.op_embeddings = nn.Embedding(NUM_OPS, K)
        self.backbone = nn.Linear(EXPR_INPUT_DIM, EXPR_HIDDEN_DIM)  # 130 → 64
        self.expr_proj = nn.Linear(EXPR_HIDDEN_DIM, EXPR_EMBED_DIM)  # 64 → 24

        # Value head (frozen during mask training)
        self.value_mlp = nn.Sequential(
            nn.Linear(EXPR_EMBED_DIM, EXPR_MLP_HIDDEN),  # 24 → 16
            nn.ReLU(),
            nn.Linear(EXPR_MLP_HIDDEN, 1),                # 16 → 1
        )

        # Mask head (THIS is what we train)
        self.mask_mlp = nn.Sequential(
            nn.Linear(EXPR_MASK_INPUT_DIM, EXPR_MLP_HIDDEN),  # 24 → 16
            nn.ReLU(),
            nn.Linear(EXPR_MLP_HIDDEN, EXPR_EMBED_DIM),       # 16 → 24
        )

        # Bilinear interaction matrix and per-rule bias
        self.interaction = nn.Parameter(torch.eye(EXPR_EMBED_DIM))  # 24 × 24
        self.mask_rule_bias = nn.Parameter(torch.zeros(num_rules))

        # Proxy rule embeddings (learned during training;
        # in Rust these come from LHS/RHS templates via encode_rule_from_templates)
        self.rule_embeds = nn.Embedding(num_rules, EXPR_EMBED_DIM)

        # Rule projection (from LHS/RHS templates → 4-way concat → rule_embed)
        self.rule_proj = nn.Linear(EXPR_RULE_CONCAT_DIM, EXPR_EMBED_DIM)  # 96 → 24

        self.num_rules = num_rules

    def freeze_backbone(self) -> None:
        """Freeze shared backbone + value head for mask-only training."""
        for p in self.op_embeddings.parameters():
            p.requires_grad = False
        for p in self.backbone.parameters():
            p.requires_grad = False
        for p in self.expr_proj.parameters():
            p.requires_grad = False
        for p in self.value_mlp.parameters():
            p.requires_grad = False
        for p in self.rule_proj.parameters():
            p.requires_grad = False

    def encode_expr(self, edge_acc: torch.Tensor) -> torch.Tensor:
        """[B, 130] edge accumulator → [B, 24] expr embedding."""
        hidden = F.relu(self.backbone(edge_acc))  # [B, 64]
        return self.expr_proj(hidden)              # [B, 24]

    def encode_rules_from_templates(
        self, lhs_accs: torch.Tensor, rhs_accs: torch.Tensor,
    ) -> torch.Tensor:
        """Encode rules from LHS/RHS edge accumulators (siamese backbone).

        Args:
            lhs_accs: [R, 130] LHS edge accumulators
            rhs_accs: [R, 130] RHS edge accumulators
        Returns: [R, 24] rule embeddings
        """
        z_lhs = self.encode_expr(lhs_accs)  # [R, 24]
        z_rhs = self.encode_expr(rhs_accs)  # [R, 24]
        # 4-way concat: [z_LHS | z_RHS | z_LHS-z_RHS | z_LHS*z_RHS]
        concat = torch.cat([
            z_lhs, z_rhs, z_lhs - z_rhs, z_lhs * z_rhs,
        ], dim=-1)  # [R, 96]
        return self.rule_proj(concat)  # [R, 24]

    def forward(
        self,
        edge_acc: torch.Tensor,
        rule_indices: torch.Tensor,
        rule_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Score (expression, rule) pairs via bilinear mask.

        Args:
            edge_acc: [B, 130] edge accumulators for expressions
            rule_indices: [B] which rule index per sample
            rule_embeds: optional [R, 24] pre-computed rule embeddings.
                         If None, uses proxy rule_embeds (self.rule_embeds).
        Returns: [B] logits (apply sigmoid for probability)
        """
        expr_embed = self.encode_expr(edge_acc)       # [B, 24]

        mask_features = self.mask_mlp(expr_embed)      # [B, 24]

        transformed = mask_features @ self.interaction  # [B, 24]

        if rule_embeds is not None:
            selected_rules = rule_embeds[rule_indices]   # [B, 24]
        else:
            selected_rules = self.rule_embeds(rule_indices)  # [B, 24]

        scores = (transformed * selected_rules).sum(dim=-1)  # [B]
        scores = scores + self.mask_rule_bias[rule_indices]

        return scores

    def export_bytes(self) -> bytes:
        """Export mask weights as little-endian f32 with MSK1 header.

        Layout matches factored.rs mask section ordering:
          MSK1 magic (4 bytes)
          mask_mlp_w1: [24][16] row-major (Rust: [MASK_INPUT_DIM][MLP_HIDDEN])
          mask_mlp_b1: [16]
          mask_mlp_w2: [16][24] row-major (Rust: [MLP_HIDDEN][EMBED_DIM])
          mask_mlp_b2: [24]
          interaction: [24][24] row-major
          mask_rule_bias: [1024] (padded to MASK_MAX_RULES)
        """
        buf = bytearray(b"MSK1")

        def write_tensor(t: torch.Tensor) -> None:
            buf.extend(t.detach().cpu().float().contiguous().numpy().tobytes())

        # mask_mlp weights (PyTorch [out,in] → Rust [in][out] via .T)
        write_tensor(self.mask_mlp[0].weight.T)   # [25, 16]
        write_tensor(self.mask_mlp[0].bias)        # [16]
        write_tensor(self.mask_mlp[2].weight.T)   # [16, 24]
        write_tensor(self.mask_mlp[2].bias)        # [24]

        # interaction matrix [24, 24]
        write_tensor(self.interaction)

        # mask_rule_bias, padded to MASK_MAX_RULES
        padded_bias = torch.zeros(EXPR_MASK_MAX_RULES)
        padded_bias[:self.num_rules] = self.mask_rule_bias.detach().cpu()
        write_tensor(padded_bias)

        expected_size = 4 + MASK_PARAM_COUNT * 4
        if len(buf) != expected_size:
            raise RuntimeError(
                f"Export size mismatch: {len(buf)} bytes, expected {expected_size}"
            )
        return bytes(buf)


# ═══════════════════════════════════════════════════════════════════════════════
# EMA Update
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def ema_update(source: nn.Module, target: nn.Module, decay: float) -> None:
    """Polyak exponential moving average: target = decay * target + (1-decay) * source."""
    for s_param, t_param in zip(source.parameters(), target.parameters()):
        t_param.data.mul_(decay).add_(s_param.data, alpha=1.0 - decay)


# ═══════════════════════════════════════════════════════════════════════════════
# Actor-Critic OPD Training
# ═══════════════════════════════════════════════════════════════════════════════

def _encode_all_graphs(
    model: EGraphNet,
    graphs: list[HeteroData],
    device: torch.device,
    batch_size: int = 256,
) -> torch.Tensor:
    """Batch-encode all unique graphs, return [num_graphs, hidden] embeddings.

    Uses PyG Batch.from_data_list() so the GNN runs once per batch of graphs
    instead of once per graph. 3305 graphs at batch_size=256 = ~13 forward passes.
    """
    was_training = model.training
    model.eval()
    embeds = []
    with torch.no_grad():
        for start in range(0, len(graphs), batch_size):
            chunk = graphs[start:start + batch_size]
            batched = Batch.from_data_list(chunk).to(device)
            embeds.append(model.encode(batched))  # [chunk_size, hidden]
    if was_training:
        model.train()
    return torch.cat(embeds, dim=0)  # [num_graphs, hidden]


def train_opd_epoch(
    critic: EGraphNet,
    critic_ema: EGraphNet,
    policy_head: PolicyHead,
    policy_head_ema: PolicyHead,
    actor: NNUEStudent,
    samples: list[PolicySample],
    graphs: list[HeteroData],
    critic_opt: torch.optim.Optimizer,
    actor_opt: torch.optim.Optimizer,
    device: torch.device,
    batch_size: int = 256,
    ema_decay: float = 0.999,
    # Pre-computed embedding caches. None = compute internally (slow path,
    # used when called without an outer round loop).
    critic_embed_cache: torch.Tensor | None = None,
    ema_embed_cache: torch.Tensor | None = None,
) -> tuple[float, float]:
    """One epoch of actor-critic OPD training.

    Pre-encodes all unique graphs once (3305 GNN passes), then trains on
    132K sample batches using cached embeddings. Returns (critic_loss_avg, actor_loss_avg).

    Callers that run multiple epochs over the same graphs should pre-compute
    critic_embed_cache and ema_embed_cache once per round and pass them in,
    amortizing the encoding cost across all epochs.
    """
    # === Phase 1: Pre-encode all unique graphs with critic (detached snapshot) ===
    # We encode once per epoch and detach. The critic GNN learns through
    # the policy head gradients flowing back to the GNN at the next epoch's
    # re-encoding. This avoids retain_graph / in-place mutation conflicts.
    if critic_embed_cache is None:
        critic_embed_cache = _encode_all_graphs(critic, graphs, device)
    # Embeddings are detached snapshots; policy head trains on these directly.
    # GNN improves across epochs as it re-encodes with updated weights.

    # === Phase 2: Pre-encode with EMA critic (no grad, for actor targets) ===
    if ema_embed_cache is None:
        ema_embed_cache = _encode_all_graphs(critic_ema, graphs, device)

    # === Phase 3: Train on sample batches using cached embeddings ===
    critic.train()
    actor.train()

    indices = torch.randperm(len(samples)).tolist()
    total_critic_loss = 0.0
    total_actor_loss = 0.0
    n_critic = 0
    n_actor = 0

    for start in range(0, len(samples), batch_size):
        batch_indices = indices[start:start + batch_size]
        batch_samples = [samples[i] for i in batch_indices]

        graph_idx_t = torch.tensor([s.graph_idx for s in batch_samples], dtype=torch.long)
        rule_idx_t = torch.tensor([s.rule_idx for s in batch_samples], dtype=torch.long, device=device)
        applied_t = torch.tensor([s.applied for s in batch_samples], dtype=torch.bool, device=device)

        # Labels pre-computed in load_policy_dataset (3-way signal)
        labels_t = torch.tensor([s.label for s in batch_samples], dtype=torch.float32, device=device)

        # Look up cached embeddings for this batch
        batch_critic_embed = critic_embed_cache[graph_idx_t]  # [B, hidden]

        # === Critic training (applied rules only) ===
        if applied_t.any():
            critic_logits = policy_head(batch_critic_embed, rule_idx_t)
            applied_logits = critic_logits[applied_t]
            applied_labels = labels_t[applied_t]
            critic_loss = F.mse_loss(torch.sigmoid(applied_logits), applied_labels)

            critic_opt.zero_grad()
            critic_loss.backward()
            critic_opt.step()

            total_critic_loss += critic_loss.item() * applied_t.sum().item()
            n_critic += applied_t.sum().item()

        # === Polyak EMA ===
        ema_update(critic, critic_ema, ema_decay)
        ema_update(policy_head, policy_head_ema, ema_decay)

        # === Actor training (ALL rules, distill from EMA critic) ===
        with torch.no_grad():
            batch_ema_embed = ema_embed_cache[graph_idx_t]
            soft_targets = torch.sigmoid(policy_head_ema(batch_ema_embed, rule_idx_t))

        # Build edge accumulators for actor input
        edge_acc = build_edge_accumulator_batch(
            [s.edge_tuples for s in batch_samples], actor.op_embeddings,
        ).to(device)

        actor_logits = actor(edge_acc, rule_idx_t)
        actor_loss = F.binary_cross_entropy_with_logits(actor_logits, soft_targets)

        actor_opt.zero_grad()
        actor_loss.backward()
        actor_opt.step()

        total_actor_loss += actor_loss.item() * len(batch_samples)
        n_actor += len(batch_samples)

    avg_critic = total_critic_loss / max(n_critic, 1)
    avg_actor = total_actor_loss / max(n_actor, 1)
    return avg_critic, avg_actor


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 1 Training & evaluation
# ═══════════════════════════════════════════════════════════════════════════════

def train_epoch(
    model: ExprGraphNet,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Run one training epoch. Returns average MSE loss."""
    model.train()
    total_loss = 0.0
    n_graphs = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        pred = model(batch)
        target = batch.y.squeeze(-1)
        loss = nn.functional.mse_loss(pred, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
        n_graphs += batch.num_graphs
    return total_loss / max(n_graphs, 1)


@torch.no_grad()
def eval_epoch(
    model: ExprGraphNet,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate model. Returns (mse_loss, spearman_rho)."""
    model.eval()
    all_pred: list[torch.Tensor] = []
    all_target: list[torch.Tensor] = []
    total_loss = 0.0
    n_graphs = 0
    for batch in loader:
        batch = batch.to(device)
        pred = model(batch)
        target = batch.y.squeeze(-1)
        loss = nn.functional.mse_loss(pred, target)
        total_loss += loss.item() * batch.num_graphs
        n_graphs += batch.num_graphs
        all_pred.append(pred.cpu())
        all_target.append(target.cpu())

    avg_loss = total_loss / max(n_graphs, 1)
    all_pred_t = torch.cat(all_pred)
    all_target_t = torch.cat(all_target)
    rho = spearman_rho(all_pred_t, all_target_t)
    return avg_loss, rho


# ═══════════════════════════════════════════════════════════════════════════════
# Commands
# ═══════════════════════════════════════════════════════════════════════════════

def cmd_train(args: argparse.Namespace) -> None:
    """Train the graph teacher model."""
    data_path = Path(args.data)
    checkpoint_path = Path(args.checkpoint)

    torch.manual_seed(args.seed)

    device = _select_device(args.device)
    print(f"Device: {device}", file=sys.stderr)

    print("Loading dataset...", file=sys.stderr)
    graphs, _names = load_dataset(data_path)

    # Deterministic 80/20 split
    n = len(graphs)
    n_train = int(0.8 * n)
    perm = torch.randperm(
        n, generator=torch.Generator().manual_seed(args.seed)
    ).tolist()
    train_graphs = [graphs[i] for i in perm[:n_train]]
    val_graphs = [graphs[i] for i in perm[n_train:]]
    print(f"Train: {len(train_graphs)}, Val: {len(val_graphs)}", file=sys.stderr)

    train_loader = DataLoader(
        train_graphs, batch_size=args.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        val_graphs, batch_size=args.batch_size, shuffle=False
    )

    model = ExprGraphNet(
        in_dim=NODE_FEAT_DIM,
        hidden=args.hidden,
        heads=args.heads,
        layers=args.layers,
        dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}", file=sys.stderr)

    optimizer = AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_loss = float("inf")
    best_epoch = -1

    header = (
        f"{'Epoch':>5}  {'Train MSE':>10}  {'Val MSE':>10}  "
        f"{'Spearman':>10}  {'LR':>10}"
    )
    print(f"\n{header}", file=sys.stderr)
    print("-" * len(header), file=sys.stderr)

    for epoch in range(1, args.epochs + 1):
        t0 = time.monotonic()
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_rho = eval_epoch(model, val_loader, device)
        scheduler.step()
        dt = time.monotonic() - t0

        lr = optimizer.param_groups[0]["lr"]
        marker = " *" if val_loss < best_val_loss else ""
        print(
            f"{epoch:5d}  {train_loss:10.6f}  {val_loss:10.6f}  "
            f"{val_rho:10.4f}  {lr:10.2e}  ({dt:.1f}s){marker}",
            file=sys.stderr,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_loss": val_loss,
                    "val_rho": val_rho,
                    "config": {
                        "in_dim": NODE_FEAT_DIM,
                        "hidden": args.hidden,
                        "heads": args.heads,
                        "layers": args.layers,
                        "dropout": args.dropout,
                    },
                },
                checkpoint_path,
            )

    print(
        f"\nBest val loss: {best_val_loss:.6f} at epoch {best_epoch}",
        file=sys.stderr,
    )
    print(f"Checkpoint saved to: {checkpoint_path}", file=sys.stderr)


def cmd_eval(args: argparse.Namespace) -> None:
    """Evaluate a trained model on data."""
    checkpoint_path = Path(args.checkpoint)
    data_path = Path(args.data)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = _select_device(args.device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt["config"]

    model = ExprGraphNet(**config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    graphs, _names = load_dataset(data_path)
    loader = DataLoader(graphs, batch_size=args.batch_size, shuffle=False)

    loss, rho = eval_epoch(model, loader, device)
    print(f"MSE Loss:       {loss:.6f}")
    print(f"RMSE (log-ns):  {math.sqrt(loss):.4f}")
    print(f"Spearman rho:   {rho:.4f}")


def cmd_export_distill(args: argparse.Namespace) -> None:
    """Export teacher predictions for distillation into the NNUE student."""
    checkpoint_path = Path(args.checkpoint)
    data_path = Path(args.data)
    output_path = Path(args.output)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = _select_device(args.device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt["config"]

    model = ExprGraphNet(**config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Load and parse graphs (preserves ordering)
    graphs, names = load_dataset(data_path)

    # Re-read raw records to preserve expression strings + timing_ns
    raw_records: dict[str, tuple[str, float]] = {}
    with open(data_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            name = rec["name"]
            if name in raw_records:
                raise ValueError(f"Duplicate name in data: {name!r}")
            raw_records[name] = (rec["expression"], rec["timing_ns"])

    loader = DataLoader(graphs, batch_size=args.batch_size, shuffle=False)
    all_preds: list[float] = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch)
            all_preds.extend(pred.cpu().tolist())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for name, pred_log_cost in zip(names, all_preds):
            expr_str, timing_ns = raw_records[name]
            record = {
                "name": name,
                "expression": expr_str,
                "teacher_log_cost": round(pred_log_cost, 6),
                "timing_ns": timing_ns,
            }
            f.write(json.dumps(record) + "\n")

    print(
        f"Exported {len(all_preds)} predictions to {output_path}",
        file=sys.stderr,
    )


def cmd_train_policy(args: argparse.Namespace) -> None:
    """Train actor-critic OPD: GNN critic + NNUE actor distillation."""
    data_path = Path(args.data)
    teacher_path = Path(args.teacher_checkpoint)
    student_path = Path(args.student_checkpoint)

    torch.manual_seed(args.seed)
    device = _select_device(args.device)
    print(f"Device: {device}", file=sys.stderr)

    print("Loading replay buffer...", file=sys.stderr)
    samples, graphs = load_policy_dataset(data_path)
    if not samples:
        raise ValueError(f"No samples loaded from {data_path}")

    # Determine num_rules from data
    num_rules = max(s.rule_idx for s in samples) + 1
    print(f"Samples: {len(samples)}, Graphs: {len(graphs)}, Rules: {num_rules}",
          file=sys.stderr)

    # Deterministic 80/20 split
    n = len(samples)
    n_train = int(0.8 * n)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(args.seed)).tolist()
    train_samples = [samples[i] for i in perm[:n_train]]
    val_samples = [samples[i] for i in perm[n_train:]]
    print(f"Train: {len(train_samples)}, Val: {len(val_samples)}", file=sys.stderr)

    # Build models
    critic = EGraphNet(
        hidden=args.hidden, heads=args.heads, layers=args.layers, dropout=args.dropout,
    ).to(device)
    critic_ema = copy.deepcopy(critic)
    policy_head = PolicyHead(
        graph_dim=args.hidden, num_rules=num_rules, rule_dim=64,
    ).to(device)
    policy_head_ema = copy.deepcopy(policy_head)
    actor = NNUEStudent(num_rules=num_rules).to(device)
    actor.freeze_backbone()

    critic_params = sum(p.numel() for p in critic.parameters())
    ph_params = sum(p.numel() for p in policy_head.parameters())
    actor_params = sum(p.numel() for p in actor.parameters())
    trainable_actor = sum(p.numel() for p in actor.parameters() if p.requires_grad)
    print(
        f"Critic: {critic_params:,} params, PolicyHead: {ph_params:,} params, "
        f"Actor: {actor_params:,} params ({trainable_actor:,} trainable)",
        file=sys.stderr,
    )

    critic_opt = AdamW(
        list(critic.parameters()) + list(policy_head.parameters()),
        lr=args.lr, weight_decay=args.weight_decay,
    )
    actor_opt = AdamW(
        [p for p in actor.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay,
    )

    best_actor_loss = float("inf")
    best_epoch = -1

    # Encode once before the epoch loop — amortized across all epochs.
    # The critic cache is intentionally not refreshed mid-loop; the critic
    # trains on stale embeddings within the run, which is standard in
    # actor-critic (avoids chasing a moving target).
    print("Pre-encoding graphs...", file=sys.stderr)
    t_enc = time.monotonic()
    critic_embed_cache = _encode_all_graphs(critic, graphs, device)
    ema_embed_cache = _encode_all_graphs(critic_ema, graphs, device)
    print(
        f"Encoded {len(graphs)} graphs in {time.monotonic() - t_enc:.1f}s",
        file=sys.stderr,
    )

    header = f"{'Epoch':>5}  {'Critic MSE':>12}  {'Actor BCE':>12}"
    print(f"\n{header}", file=sys.stderr)
    print("-" * len(header), file=sys.stderr)

    for epoch in range(1, args.epochs + 1):
        t0 = time.monotonic()
        critic_loss, actor_loss = train_opd_epoch(
            critic, critic_ema, policy_head, policy_head_ema, actor,
            train_samples, graphs, critic_opt, actor_opt, device,
            batch_size=args.batch_size, ema_decay=args.ema_decay,
            critic_embed_cache=critic_embed_cache,
            ema_embed_cache=ema_embed_cache,
        )
        dt = time.monotonic() - t0

        # Refresh EMA cache every 5 epochs — EMA drifts slowly, so
        # stale embeddings are fine within a window.
        if epoch % 5 == 0:
            ema_embed_cache = _encode_all_graphs(critic_ema, graphs, device)

        # Validate actor on held-out set using cached EMA embeddings
        actor.eval()
        val_actor_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for start in range(0, len(val_samples), args.batch_size):
                batch_s = val_samples[start:start + args.batch_size]
                rule_idx_t = torch.tensor([s.rule_idx for s in batch_s], dtype=torch.long, device=device)
                graph_idx_t = torch.tensor([s.graph_idx for s in batch_s], dtype=torch.long)

                batch_ema_embed = ema_embed_cache[graph_idx_t]
                soft_targets = torch.sigmoid(policy_head_ema(batch_ema_embed, rule_idx_t))

                edge_acc = build_edge_accumulator_batch(
                    [s.edge_tuples for s in batch_s], actor.op_embeddings,
                ).to(device)
                actor_logits = actor(edge_acc, rule_idx_t)
                loss = F.binary_cross_entropy_with_logits(actor_logits, soft_targets)
                val_actor_loss += loss.item() * len(batch_s)
                n_val += len(batch_s)
        val_actor = val_actor_loss / max(n_val, 1)

        marker = " *" if val_actor < best_actor_loss else ""
        print(
            f"{epoch:5d}  {critic_loss:12.6f}  {actor_loss:12.6f}  "
            f"val_actor={val_actor:.6f}  ({dt:.1f}s){marker}",
            file=sys.stderr,
        )

        if val_actor < best_actor_loss:
            best_actor_loss = val_actor
            best_epoch = epoch
            teacher_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "epoch": epoch,
                "critic_state_dict": critic_ema.state_dict(),
                "policy_head_state_dict": policy_head_ema.state_dict(),
                "actor_state_dict": actor.state_dict(),
                "num_rules": num_rules,
                "config": {
                    "hidden": args.hidden,
                    "heads": args.heads,
                    "layers": args.layers,
                    "dropout": args.dropout,
                },
            }, teacher_path)
            student_path.parent.mkdir(parents=True, exist_ok=True)
            student_path.write_bytes(actor.export_bytes())

    print(
        f"\nBest val actor loss: {best_actor_loss:.6f} at epoch {best_epoch}",
        file=sys.stderr,
    )
    print(f"Teacher checkpoint: {teacher_path}", file=sys.stderr)
    print(f"Student binary: {student_path}", file=sys.stderr)


def cmd_eval_policy(args: argparse.Namespace) -> None:
    """Evaluate actor-critic models on replay buffer data."""
    teacher_path = Path(args.teacher_checkpoint)
    student_path = Path(args.student_checkpoint)
    data_path = Path(args.data)

    if not teacher_path.exists():
        raise FileNotFoundError(f"Teacher checkpoint not found: {teacher_path}")
    if not student_path.exists():
        raise FileNotFoundError(f"Student checkpoint not found: {student_path}")

    device = _select_device(args.device)

    # Load models
    ckpt = torch.load(teacher_path, map_location=device, weights_only=False)
    config = ckpt["config"]
    num_rules = ckpt["num_rules"]

    critic = EGraphNet(**config).to(device)
    critic.load_state_dict(ckpt["critic_state_dict"])
    critic.eval()

    policy_head = PolicyHead(graph_dim=config["hidden"], num_rules=num_rules).to(device)
    policy_head.load_state_dict(ckpt["policy_head_state_dict"])
    policy_head.eval()

    actor = NNUEStudent(num_rules=num_rules).to(device)
    actor.load_state_dict(ckpt["actor_state_dict"])
    actor.eval()

    # Load data
    samples, graphs = load_policy_dataset(data_path)

    # Pre-encode all unique graphs once
    embed_cache = _encode_all_graphs(critic, graphs, device)

    # Evaluate critic and actor
    critic_preds: list[float] = []
    actor_preds: list[float] = []
    labels: list[float] = []
    applied_indices: list[int] = []
    per_rule_correct: dict[int, list[bool]] = {}

    with torch.no_grad():
        for i, s in enumerate(samples):
            rule_idx_t = torch.tensor([s.rule_idx], dtype=torch.long, device=device)
            embed = embed_cache[s.graph_idx].unsqueeze(0)

            critic_logit = policy_head(embed, rule_idx_t).item()

            edge_acc = build_edge_accumulator_batch(
                [s.edge_tuples], actor.op_embeddings,
            ).to(device)
            actor_logit = actor(edge_acc, rule_idx_t).item()
            critic_p = 1.0 / (1.0 + math.exp(-critic_logit))
            actor_p = 1.0 / (1.0 + math.exp(-actor_logit))

            critic_preds.append(critic_p)
            actor_preds.append(actor_p)

            if s.applied:
                labels.append(s.label)
                applied_indices.append(i)

                # Per-rule accuracy (threshold 0.3 for "useful")
                is_useful = s.label > 0.1
                critic_correct = (critic_p > 0.3) == is_useful
                actor_correct = (actor_p > 0.3) == is_useful
                per_rule_correct.setdefault(s.rule_idx, []).append(actor_correct)

    # Overall metrics
    n_applied = len(applied_indices)
    if n_applied > 0:
        applied_critic = [critic_preds[i] for i in applied_indices]
        applied_actor = [actor_preds[i] for i in applied_indices]

        critic_rho = spearman_rho(
            torch.tensor(applied_critic), torch.tensor(labels),
        )
        actor_rho = spearman_rho(
            torch.tensor(applied_actor), torch.tensor(labels),
        )
    else:
        critic_rho = 0.0
        actor_rho = 0.0

    print(f"Total samples:   {len(samples)}")
    print(f"Applied samples: {n_applied}")
    print(f"Critic Spearman: {critic_rho:.4f}")
    print(f"Actor Spearman:  {actor_rho:.4f}")
    print()

    # Per-rule breakdown
    print("Per-rule actor accuracy:")
    for rule_idx in sorted(per_rule_correct.keys()):
        corrects = per_rule_correct[rule_idx]
        acc = sum(corrects) / len(corrects) if corrects else 0.0
        print(f"  Rule {rule_idx:3d}: {acc:.3f} ({sum(corrects)}/{len(corrects)})")


def cmd_export_guide(args: argparse.Namespace) -> None:
    """Export actor mask weights to Rust binary format (MSK1)."""
    student_path = Path(args.student_checkpoint)
    output_path = Path(args.output)

    if not student_path.exists():
        raise FileNotFoundError(f"Student checkpoint not found: {student_path}")

    expected_msk1_size = 4 + MASK_PARAM_COUNT * 4  # MSK1 magic + weights

    # Check if it's already a raw MSK1 binary (from train-policy)
    raw_bytes = student_path.read_bytes()
    if len(raw_bytes) == expected_msk1_size and raw_bytes[:4] == b"MSK1":
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(raw_bytes)
        print(f"Copied MSK1 binary ({len(raw_bytes)} bytes) to {output_path}")
        return

    # Legacy: old GuideNnue raw binary
    if len(raw_bytes) == GUIDE_PARAM_COUNT * 4:
        raise ValueError(
            f"{student_path} is old GuideNnue format ({len(raw_bytes)} bytes). "
            f"Retrain with new NNUEStudent to get MSK1 format."
        )

    # Try loading as torch checkpoint
    device = torch.device("cpu")
    ckpt = torch.load(student_path, map_location=device, weights_only=False)
    num_rules = ckpt.get("num_rules", 64)
    actor = NNUEStudent(num_rules=num_rules)
    if "actor_state_dict" in ckpt:
        actor.load_state_dict(ckpt["actor_state_dict"])
    elif "model_state_dict" in ckpt:
        actor.load_state_dict(ckpt["model_state_dict"])
    else:
        raise ValueError(
            f"Checkpoint {student_path} has no 'actor_state_dict' or 'model_state_dict'. "
            f"Keys: {list(ckpt.keys())}"
        )

    exported = actor.export_bytes()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(exported)
    print(f"Exported {len(exported)} bytes ({MASK_PARAM_COUNT} mask params) to {output_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Online Training Infrastructure
# ═══════════════════════════════════════════════════════════════════════════════


def _find_workspace_root() -> Path:
    """Find the Cargo workspace root by walking up from this script."""
    d = Path(__file__).resolve().parent
    for _ in range(10):
        cargo = d / "Cargo.toml"
        if cargo.exists() and "[workspace]" in cargo.read_text():
            return d
        d = d.parent
    raise FileNotFoundError("Could not find Cargo workspace root")


def _cargo_build_collect() -> None:
    """Build collect_guide_data binary (release mode). Fails loudly."""
    workspace = _find_workspace_root()
    print("Building collect_guide_data (release)...", file=sys.stderr)
    result = subprocess.run(
        [
            "cargo", "build", "-p", "pixelflow-pipeline",
            "--bin", "collect_guide_data", "--release", "--features", "training",
        ],
        cwd=workspace,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"cargo build failed:\n{result.stderr}")
    print("Build complete.", file=sys.stderr)


def _run_collect(
    *,
    guide_weights: Path | None,
    judge_weights: str,
    output: Path,
    count: int,
    perturbations: int,
    sigma: float,
    threshold: float,
    max_epochs: int,
    seed: int,
    timeout: int = 600,
) -> None:
    """Run collect_guide_data binary. Fails loudly on non-zero exit."""
    workspace = _find_workspace_root()
    target_bin = workspace / "target" / "release" / "collect_guide_data"
    if not target_bin.exists():
        raise FileNotFoundError(
            f"Binary not found at {target_bin} — did _cargo_build_collect() run?"
        )

    # Truncate output file so we don't accumulate stale data across rounds
    output_resolved = Path(output).resolve()
    if output_resolved.exists():
        output_resolved.unlink()

    # Resolve all paths to absolute so cwd doesn't matter
    cmd: list[str] = [
        str(target_bin),
        "--count", str(count),
        "--perturbations", str(perturbations),
        "--sigma", str(sigma),
        "--threshold", str(threshold),
        "--max-epochs", str(max_epochs),
        "--seed", str(seed),
        "--output", str(output_resolved),
        "--judge-weights", str(Path(judge_weights).resolve()),
    ]
    # guide_weights (trained student MSK1) not yet fed back into the Rust collector —
    # collect_guide_data perturbs mask weights internally from judge.bin.
    _ = guide_weights  # suppress unused-variable warning

    try:
        result = subprocess.run(
            cmd, cwd=workspace, capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            f"collect_guide_data timed out after {timeout}s"
        )
    if result.returncode != 0:
        raise RuntimeError(
            f"collect_guide_data failed (exit {result.returncode}):\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
    if result.stderr.strip():
        print(result.stderr.strip(), file=sys.stderr)
    if result.stdout.strip():
        print(result.stdout.strip(), file=sys.stderr)


def _run_judge_update(
    model_path: Path,
    corpus: str,
    rounds: int = 5,
    bench_budget: int = 64,
) -> None:
    """Run train_online to re-ground the Judge on hard-mined JIT benchmarks."""
    cmd = [
        "cargo", "run", "--release",
        "-p", "pixelflow-pipeline",
        "--features", "training",
        "--bin", "train_online",
        "--",
        "--model", str(model_path),
        "--corpus", corpus,
        "--rounds", str(rounds),
        "--bench-budget", str(bench_budget),
    ]
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        raise RuntimeError(f"train_online failed with exit code {result.returncode}")


def _validate_actor(
    actor: NNUEStudent,
    critic_ema: EGraphNet,
    policy_head_ema: PolicyHead,
    val_samples: list[PolicySample],
    graphs: list[HeteroData],
    device: torch.device,
    batch_size: int,
    ema_embed_cache: torch.Tensor | None = None,
) -> float:
    """Compute validation actor BCE loss against EMA critic soft targets."""
    if ema_embed_cache is None:
        ema_embed_cache = _encode_all_graphs(critic_ema, graphs, device)
    actor.eval()
    total_loss = 0.0
    n_val = 0
    with torch.no_grad():
        for start in range(0, len(val_samples), batch_size):
            batch_s = val_samples[start:start + batch_size]
            rule_idx_t = torch.tensor(
                [s.rule_idx for s in batch_s], dtype=torch.long, device=device,
            )
            graph_idx_t = torch.tensor(
                [s.graph_idx for s in batch_s], dtype=torch.long,
            )
            batch_ema_embed = ema_embed_cache[graph_idx_t]
            soft_targets = torch.sigmoid(policy_head_ema(batch_ema_embed, rule_idx_t))

            edge_acc = build_edge_accumulator_batch(
                [s.edge_tuples for s in batch_s], actor.op_embeddings,
            ).to(device)
            actor_logits = actor(edge_acc, rule_idx_t)
            loss = F.binary_cross_entropy_with_logits(actor_logits, soft_targets)
            total_loss += loss.item() * len(batch_s)
            n_val += len(batch_s)
    actor.train()
    return total_loss / max(n_val, 1)


def cmd_online_train(args: argparse.Namespace) -> None:
    """Online actor-critic OPD: alternate data collection and training."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    replay_path = output_dir / "replay_buffer.jsonl"
    student_path = output_dir / "guide_student.bin"
    teacher_path = output_dir / "graph_teacher.pt"
    log_path = output_dir / "training_log.jsonl"

    torch.manual_seed(args.seed)
    device = _select_device(args.device)
    print(f"Device: {device}", file=sys.stderr)

    # Build the Rust binary once
    _cargo_build_collect()

    # Initialize models — we don't know num_rules yet, so use a reasonable default.
    # Will be rebuilt after first data collection if needed.
    num_rules = 64
    critic = EGraphNet(
        hidden=args.hidden, heads=args.heads,
        layers=args.layers, dropout=args.dropout,
    ).to(device)
    critic_ema = copy.deepcopy(critic)
    policy_head = PolicyHead(
        graph_dim=args.hidden, num_rules=num_rules, rule_dim=64,
    ).to(device)
    policy_head_ema = copy.deepcopy(policy_head)
    actor = NNUEStudent(num_rules=num_rules).to(device)
    actor.freeze_backbone()

    # Warm-start actor from previous MSK1 checkpoint
    if args.warm_start:
        warm_path = Path(args.warm_start)
        if not warm_path.exists():
            raise FileNotFoundError(f"Warm-start weights not found: {warm_path}")
        raw = warm_path.read_bytes()
        expected_msk1 = 4 + MASK_PARAM_COUNT * 4
        if len(raw) != expected_msk1:
            raise ValueError(
                f"Warm-start file is {len(raw)} bytes, expected {expected_msk1} (MSK1). "
                f"Old GuideNnue format? Retrain required."
            )
        _load_actor_from_bytes(actor, raw)
        print(f"Warm-started actor from {warm_path}", file=sys.stderr)

    critic_opt = AdamW(
        list(critic.parameters()) + list(policy_head.parameters()),
        lr=args.lr, weight_decay=args.weight_decay,
    )
    actor_opt = AdamW(
        [p for p in actor.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay,
    )

    critic_params = sum(p.numel() for p in critic.parameters())
    ph_params = sum(p.numel() for p in policy_head.parameters())
    actor_params = sum(p.numel() for p in actor.parameters())
    trainable_actor = sum(p.numel() for p in actor.parameters() if p.requires_grad)
    print(
        f"Critic: {critic_params:,} params, PolicyHead: {ph_params:,} params, "
        f"Actor: {actor_params:,} params ({trainable_actor:,} trainable)",
        file=sys.stderr,
    )

    best_val_loss = float("inf")
    best_round = -1

    for round_idx in range(args.rounds):
        print(
            f"\n{'=' * 60}\n=== Round {round_idx + 1}/{args.rounds} ===\n{'=' * 60}",
            file=sys.stderr,
        )

        # === COLLECT PHASE ===
        t_collect = time.monotonic()
        # Export current actor weights for the collector to use
        if round_idx > 0 or args.warm_start:
            student_path.write_bytes(actor.export_bytes())
            guide_w = student_path
        else:
            guide_w = None  # First round: random actor

        print(
            f"Collecting replay data (seed={args.seed + round_idx})...",
            file=sys.stderr,
        )
        _run_collect(
            guide_weights=guide_w,
            judge_weights=args.judge_weights,
            output=replay_path,
            count=args.collect_count,
            perturbations=args.collect_perturbations,
            sigma=args.collect_sigma,
            threshold=args.collect_threshold,
            max_epochs=args.collect_max_epochs,
            seed=args.seed + round_idx,
        )
        dt_collect = time.monotonic() - t_collect

        # === LOAD PHASE ===
        samples, graphs = load_policy_dataset(replay_path)
        if not samples:
            raise ValueError(f"No samples collected in round {round_idx + 1}")

        data_num_rules = max(s.rule_idx for s in samples) + 1
        if data_num_rules != num_rules:
            print(
                f"  Resizing PolicyHead + Actor: {num_rules} → {data_num_rules} rules",
                file=sys.stderr,
            )
            num_rules = data_num_rules
            policy_head = PolicyHead(
                graph_dim=args.hidden, num_rules=num_rules, rule_dim=64,
            ).to(device)
            policy_head_ema = copy.deepcopy(policy_head)
            actor = NNUEStudent(num_rules=num_rules).to(device)
            actor.freeze_backbone()
            critic_opt = AdamW(
                list(critic.parameters()) + list(policy_head.parameters()),
                lr=args.lr, weight_decay=args.weight_decay,
            )
            actor_opt = AdamW(
                [p for p in actor.parameters() if p.requires_grad],
                lr=args.lr, weight_decay=args.weight_decay,
            )

        # Train/val split (deterministic per round)
        n = len(samples)
        n_train = int(0.8 * n)
        gen = torch.Generator().manual_seed(args.seed + round_idx)
        perm = torch.randperm(n, generator=gen).tolist()
        train_samples = [samples[i] for i in perm[:n_train]]
        val_samples = [samples[i] for i in perm[n_train:]]

        print(
            f"  {len(samples)} samples, {len(graphs)} graphs, "
            f"{len(train_samples)} train / {len(val_samples)} val  "
            f"(collect: {dt_collect:.1f}s)",
            file=sys.stderr,
        )

        # === TRAIN PHASE ===
        # Encode once per round — amortized across all epochs in this round.
        # The critic cache is intentionally not refreshed within the round;
        # training on stale embeddings avoids chasing a moving target.
        print("  Pre-encoding graphs...", file=sys.stderr)
        t_enc = time.monotonic()
        critic_embed_cache = _encode_all_graphs(critic, graphs, device)
        ema_embed_cache = _encode_all_graphs(critic_ema, graphs, device)
        print(
            f"  Encoded {len(graphs)} graphs in {time.monotonic() - t_enc:.1f}s",
            file=sys.stderr,
        )

        header = f"  {'Ep':>3}  {'Critic':>10}  {'Actor':>10}  {'Val':>10}"
        print(header, file=sys.stderr)
        print(f"  {'-' * (len(header) - 2)}", file=sys.stderr)

        for epoch in range(1, args.train_epochs_per_round + 1):
            t_train = time.monotonic()
            critic_loss, actor_loss = train_opd_epoch(
                critic, critic_ema, policy_head, policy_head_ema, actor,
                train_samples, graphs, critic_opt, actor_opt, device,
                batch_size=args.batch_size, ema_decay=args.ema_decay,
                critic_embed_cache=critic_embed_cache,
                ema_embed_cache=ema_embed_cache,
            )
            # Refresh EMA cache every 5 epochs — EMA drifts slowly, so
            # stale embeddings are fine within a window.
            if epoch % 5 == 0:
                ema_embed_cache = _encode_all_graphs(critic_ema, graphs, device)
            val_actor = _validate_actor(
                actor, critic_ema, policy_head_ema,
                val_samples, graphs, device, args.batch_size,
                ema_embed_cache=ema_embed_cache,
            )
            dt_train = time.monotonic() - t_train

            marker = " *" if val_actor < best_val_loss else ""
            print(
                f"  {epoch:3d}  {critic_loss:10.6f}  {actor_loss:10.6f}  "
                f"{val_actor:10.6f}  ({dt_train:.1f}s){marker}",
                file=sys.stderr,
            )

            # Append to structured log
            log_entry = {
                "round": round_idx,
                "epoch": epoch,
                "critic_loss": round(critic_loss, 6),
                "actor_loss": round(actor_loss, 6),
                "val_actor_loss": round(val_actor, 6),
                "n_samples": len(samples),
                "n_graphs": len(graphs),
                "collect_time_s": round(dt_collect, 1),
                "train_time_s": round(dt_train, 1),
            }
            with open(log_path, "a") as f:
                f.write(json.dumps(log_entry) + "\n")

            if val_actor < best_val_loss:
                best_val_loss = val_actor
                best_round = round_idx

        # === CHECKPOINT (every round) ===
        torch.save(
            {
                "round": round_idx,
                "critic_state_dict": critic_ema.state_dict(),
                "policy_head_state_dict": policy_head_ema.state_dict(),
                "actor_state_dict": actor.state_dict(),
                "num_rules": num_rules,
                "config": {
                    "hidden": args.hidden,
                    "heads": args.heads,
                    "layers": args.layers,
                    "dropout": args.dropout,
                },
            },
            teacher_path,
        )
        student_path.write_bytes(actor.export_bytes())
        print(
            f"  Checkpoint saved. Best val_actor: {best_val_loss:.6f} "
            f"(round {best_round + 1})",
            file=sys.stderr,
        )

        # === JUDGE UPDATE (every N rounds) ===
        if (args.judge_update_every > 0
                and (round_idx + 1) % args.judge_update_every == 0):
            print(
                f"\n  [Judge update: {args.judge_update_rounds} adversarial rounds]",
                file=sys.stderr,
            )
            _run_judge_update(
                model_path=Path(args.judge_weights),
                corpus=args.corpus,
                rounds=args.judge_update_rounds,
                bench_budget=args.judge_update_bench_budget,
            )
            # Reload updated backbone so the actor targets a fresh Value function.
            _reload_actor_backbone(actor, Path(args.judge_weights))
            print("  [Backbone reloaded from updated judge.bin]", file=sys.stderr)

    print(
        f"\n{'=' * 60}\n"
        f"Online training complete. {args.rounds} rounds.\n"
        f"Best val actor loss: {best_val_loss:.6f} at round {best_round + 1}\n"
        f"Student: {student_path}\n"
        f"Teacher: {teacher_path}\n"
        f"Log: {log_path}\n"
        f"{'=' * 60}",
        file=sys.stderr,
    )


def _load_actor_from_bytes(actor: NNUEStudent, raw: bytes) -> None:
    """Load NNUEStudent mask weights from MSK1 raw bytes.

    MSK1 layout:
      magic: b"MSK1" (4 bytes)
      mask_mlp_w1: [25][16] row-major LE f32
      mask_mlp_b1: [16] LE f32
      mask_mlp_w2: [16][24] row-major LE f32
      mask_mlp_b2: [24] LE f32
      interaction: [24][24] row-major LE f32
      mask_rule_bias: [1024] LE f32
    """
    expected_size = 4 + MASK_PARAM_COUNT * 4
    if len(raw) != expected_size:
        raise ValueError(
            f"MSK1 binary is {len(raw)} bytes, expected {expected_size}. "
            f"Old GuideNnue format? Retrain required."
        )
    if raw[:4] != b"MSK1":
        raise ValueError(
            f"Bad magic: {raw[:4]!r}, expected b'MSK1'. "
            f"Old GuideNnue format? Retrain required."
        )

    import numpy as np
    data = np.frombuffer(raw[4:], dtype=np.float32)
    idx = 0

    # mask_mlp_w1: [24][16] row-major → PyTorch [16, 24]
    n = EXPR_MASK_INPUT_DIM * EXPR_MLP_HIDDEN  # 384
    w1 = torch.from_numpy(data[idx:idx + n].copy()).reshape(EXPR_MASK_INPUT_DIM, EXPR_MLP_HIDDEN)
    actor.mask_mlp[0].weight.data.copy_(w1.T)
    idx += n

    # mask_mlp_b1: [16]
    b1 = torch.from_numpy(data[idx:idx + EXPR_MLP_HIDDEN].copy())
    actor.mask_mlp[0].bias.data.copy_(b1)
    idx += EXPR_MLP_HIDDEN

    # mask_mlp_w2: [16][24] row-major → PyTorch [24, 16]
    n = EXPR_MLP_HIDDEN * EXPR_EMBED_DIM  # 384
    w2 = torch.from_numpy(data[idx:idx + n].copy()).reshape(EXPR_MLP_HIDDEN, EXPR_EMBED_DIM)
    actor.mask_mlp[2].weight.data.copy_(w2.T)
    idx += n

    # mask_mlp_b2: [24]
    b2 = torch.from_numpy(data[idx:idx + EXPR_EMBED_DIM].copy())
    actor.mask_mlp[2].bias.data.copy_(b2)
    idx += EXPR_EMBED_DIM

    # interaction: [24][24]
    n = EXPR_EMBED_DIM * EXPR_EMBED_DIM  # 576
    interaction = torch.from_numpy(data[idx:idx + n].copy()).reshape(EXPR_EMBED_DIM, EXPR_EMBED_DIM)
    actor.interaction.data.copy_(interaction)
    idx += n

    # mask_rule_bias: [1024], but actor may have fewer rules
    bias_data = torch.from_numpy(data[idx:idx + EXPR_MASK_MAX_RULES].copy())
    actor.mask_rule_bias.data.copy_(bias_data[:actor.num_rules])


def _reload_actor_backbone(actor: NNUEStudent, judge_path: Path) -> None:
    """Reload backbone weights into actor from an updated judge.bin (TRI7 format).

    Only touches the frozen backbone parameters — mask weights are preserved.
    Re-calls freeze_backbone() to restore requires_grad=False on all backbone params.

    TRI7 backbone fields (in file order, after 4-byte magic):
      embeddings:    [NUM_OPS=41][K=32]     = 1,312 f32s
      w1 (backbone): [INPUT_DIM=130][HIDDEN_DIM=64] = 8,320 f32s
      b1 (backbone): [HIDDEN_DIM=64]        =    64 f32s
      value_w:       [HIDDEN_DIM=64]        =    64 f32s  (legacy value head, skipped)
      value_b:       1 f32                                 (skipped)
      var_w:         [HIDDEN_DIM=64]        =    64 f32s  (TRI7 variance, skipped)
      var_b:         1 f32                                 (skipped)
      rule_embeddings: [MAX_RULES=64][HIDDEN_DIM=64] = 4,096 f32s (skipped)
      rule_bias:     [MAX_RULES=64]         =    64 f32s  (skipped)
      expr_proj_w:   [HIDDEN_DIM=64][EMBED_DIM=24] = 1,536 f32s
      expr_proj_b:   [EMBED_DIM=24]         =    24 f32s
      value_mlp_w1:  [EMBED_DIM=24][MLP_HIDDEN=16] = 384 f32s
      value_mlp_b1:  [MLP_HIDDEN=16]        =    16 f32s
      value_mlp_w2:  [MLP_HIDDEN=16]        =    16 f32s
      value_mlp_b2:  1 f32
      (mask MLP + rule MLP + rule_proj + interaction + mask_rule_bias follow, not read here)

    The rule_proj_{w,b} fields appear after mask/rule MLPs in the file, but those
    are mask-side weights and are already covered by _load_actor_from_bytes.
    """
    import numpy as np

    raw = judge_path.read_bytes()
    magic = raw[:4]
    if magic not in (b"TRI7", b"TRI6", b"TRI5"):
        raise ValueError(
            f"Unexpected judge.bin magic {magic!r}. "
            f"Expected TRI5/TRI6/TRI7. Retrain judge required."
        )
    has_variance = magic == b"TRI7"
    skip_search_head = magic == b"TRI5"

    # Dimensions (must match Rust constants in factored.rs)
    NUM_OPS_RUST = 41
    K_RUST = 32
    INPUT_DIM_RUST = 4 * K_RUST + 2   # 130
    HIDDEN_DIM_RUST = 64
    MAX_RULES_RUST = 64
    EMBED_DIM_RUST = 32
    MLP_HIDDEN_RUST = 16

    data = np.frombuffer(raw[4:], dtype=np.float32)
    idx = 0

    # embeddings: [41][32]
    n_emb = NUM_OPS_RUST * K_RUST  # 1,312
    emb = torch.from_numpy(data[idx:idx + n_emb].copy()).reshape(NUM_OPS_RUST, K_RUST)
    actor.op_embeddings.weight.data.copy_(emb)
    idx += n_emb

    # backbone w1: [130][64] row-major → PyTorch Linear weight [64, 130]
    n_w1 = INPUT_DIM_RUST * HIDDEN_DIM_RUST  # 8,320
    w1 = torch.from_numpy(data[idx:idx + n_w1].copy()).reshape(INPUT_DIM_RUST, HIDDEN_DIM_RUST)
    actor.backbone.weight.data.copy_(w1.T)
    idx += n_w1

    # backbone b1: [64]
    b1 = torch.from_numpy(data[idx:idx + HIDDEN_DIM_RUST].copy())
    actor.backbone.bias.data.copy_(b1)
    idx += HIDDEN_DIM_RUST

    # legacy value_w [64] + value_b [1] — skip, not used in NNUEStudent
    idx += HIDDEN_DIM_RUST + 1

    # TRI7 variance head var_w [64] + var_b [1] — skip
    if has_variance:
        idx += HIDDEN_DIM_RUST + 1

    # TRI5 search head was removed in TRI6 — skip remaining bytes
    if skip_search_head:
        idx += HIDDEN_DIM_RUST + 1

    # rule_embeddings [64][64] + rule_bias [64] — skip
    idx += MAX_RULES_RUST * HIDDEN_DIM_RUST + MAX_RULES_RUST

    # expr_proj_w: [HIDDEN_DIM=64][EMBED_DIM=24] row-major → PyTorch Linear weight [24, 64]
    n_epw = HIDDEN_DIM_RUST * EMBED_DIM_RUST  # 1,536
    epw = torch.from_numpy(data[idx:idx + n_epw].copy()).reshape(HIDDEN_DIM_RUST, EMBED_DIM_RUST)
    actor.expr_proj.weight.data.copy_(epw.T)
    idx += n_epw

    # expr_proj_b: [24]
    epb = torch.from_numpy(data[idx:idx + EMBED_DIM_RUST].copy())
    actor.expr_proj.bias.data.copy_(epb)
    idx += EMBED_DIM_RUST

    # value_mlp_w1: [EMBED_DIM=24][MLP_HIDDEN=16] row-major → PyTorch Linear weight [16, 24]
    n_vm1 = EMBED_DIM_RUST * MLP_HIDDEN_RUST  # 384
    vm1 = torch.from_numpy(data[idx:idx + n_vm1].copy()).reshape(EMBED_DIM_RUST, MLP_HIDDEN_RUST)
    actor.value_mlp[0].weight.data.copy_(vm1.T)
    idx += n_vm1

    # value_mlp_b1: [16]
    vmb1 = torch.from_numpy(data[idx:idx + MLP_HIDDEN_RUST].copy())
    actor.value_mlp[0].bias.data.copy_(vmb1)
    idx += MLP_HIDDEN_RUST

    # value_mlp_w2: [MLP_HIDDEN=16] → PyTorch Linear weight [1, 16]
    vm2 = torch.from_numpy(data[idx:idx + MLP_HIDDEN_RUST].copy()).reshape(1, MLP_HIDDEN_RUST)
    actor.value_mlp[2].weight.data.copy_(vm2)
    idx += MLP_HIDDEN_RUST

    # value_mlp_b2: scalar
    vmb2 = float(data[idx])
    actor.value_mlp[2].bias.data.fill_(vmb2)
    idx += 1  # noqa: F841 (idx kept for clarity if extended later)

    # Freeze the reloaded backbone so mask training cannot corrupt it.
    actor.freeze_backbone()


def _select_device(device_str: str | None) -> torch.device:
    """Select compute device: explicit > MPS > CUDA > CPU."""
    if device_str:
        return torch.device(device_str)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    p = argparse.ArgumentParser(
        description="Graph Transformer teacher for PixelFlow cost prediction",
    )
    sub = p.add_subparsers(dest="command", required=True)

    # ── train ──────────────────────────────────────────────────────────────
    train_p = sub.add_parser("train", help="Train the graph teacher model")
    train_p.add_argument(
        "--data", required=True, help="Path to judge_training.jsonl"
    )
    train_p.add_argument(
        "--checkpoint",
        default="graph_teacher_v1.pt",
        help="Path to save best model checkpoint",
    )
    train_p.add_argument("--epochs", type=int, default=100)
    train_p.add_argument("--batch-size", type=int, default=64)
    train_p.add_argument("--lr", type=float, default=1e-3)
    train_p.add_argument("--weight-decay", type=float, default=1e-4)
    train_p.add_argument("--hidden", type=int, default=128)
    train_p.add_argument("--heads", type=int, default=4)
    train_p.add_argument("--layers", type=int, default=4)
    train_p.add_argument("--dropout", type=float, default=0.1)
    train_p.add_argument("--seed", type=int, default=42)
    train_p.add_argument("--device", default=None, help="Force device (cpu/mps/cuda)")

    # ── eval ──────────────────────────────────────────────────────────────
    eval_p = sub.add_parser("eval", help="Evaluate a trained model")
    eval_p.add_argument(
        "--data", required=True, help="Path to judge_training.jsonl"
    )
    eval_p.add_argument(
        "--checkpoint", required=True, help="Path to model checkpoint"
    )
    eval_p.add_argument("--batch-size", type=int, default=64)
    eval_p.add_argument("--device", default=None)

    # ── export-distill ────────────────────────────────────────────────────
    distill_p = sub.add_parser(
        "export-distill",
        help="Export teacher predictions for NNUE distillation",
    )
    distill_p.add_argument(
        "--data", required=True, help="Path to judge_training.jsonl"
    )
    distill_p.add_argument(
        "--checkpoint", required=True, help="Path to model checkpoint"
    )
    distill_p.add_argument(
        "--output",
        default="teacher_distill.jsonl",
        help="Output JSONL path",
    )
    distill_p.add_argument("--batch-size", type=int, default=64)
    distill_p.add_argument("--device", default=None)

    # ── train-policy ──────────────────────────────────────────────────────
    policy_p = sub.add_parser(
        "train-policy", help="Train actor-critic OPD (GNN critic + NNUE actor)"
    )
    policy_p.add_argument(
        "--data", required=True, help="Path to guide_training.jsonl replay buffer"
    )
    policy_p.add_argument(
        "--teacher-checkpoint", default="graph_teacher_v2.pt",
        help="Path to save teacher (critic EMA) checkpoint",
    )
    policy_p.add_argument(
        "--student-checkpoint", default="guide_student.bin",
        help="Path to save student (actor) binary weights",
    )
    policy_p.add_argument("--epochs", type=int, default=50)
    policy_p.add_argument("--batch-size", type=int, default=64)
    policy_p.add_argument("--lr", type=float, default=5e-4)
    policy_p.add_argument("--weight-decay", type=float, default=1e-4)
    policy_p.add_argument("--ema-decay", type=float, default=0.999)
    policy_p.add_argument("--hidden", type=int, default=128)
    policy_p.add_argument("--heads", type=int, default=4)
    policy_p.add_argument("--layers", type=int, default=4)
    policy_p.add_argument("--dropout", type=float, default=0.1)
    policy_p.add_argument("--seed", type=int, default=42)
    policy_p.add_argument("--device", default=None)

    # ── eval-policy ───────────────────────────────────────────────────────
    eval_policy_p = sub.add_parser(
        "eval-policy", help="Evaluate actor-critic models on replay buffer"
    )
    eval_policy_p.add_argument(
        "--data", required=True, help="Path to guide_training.jsonl replay buffer"
    )
    eval_policy_p.add_argument(
        "--teacher-checkpoint", required=True, help="Path to teacher checkpoint"
    )
    eval_policy_p.add_argument(
        "--student-checkpoint", required=True, help="Path to student binary"
    )
    eval_policy_p.add_argument("--batch-size", type=int, default=64)
    eval_policy_p.add_argument("--device", default=None)

    # ── export-guide ──────────────────────────────────────────────────────
    export_guide_p = sub.add_parser(
        "export-guide", help="Export actor weights to Rust binary"
    )
    export_guide_p.add_argument(
        "--student-checkpoint", required=True,
        help="Path to student checkpoint (.bin or .pt)",
    )
    export_guide_p.add_argument(
        "--output", default="guide_weights.bin",
        help="Output path for raw f32 binary",
    )

    # ── online-train ─────────────────────────────────────────────────────
    online_p = sub.add_parser(
        "online-train",
        help="Online actor-critic OPD: alternate data collection and training",
    )
    online_p.add_argument(
        "--judge-weights", required=True,
        help="Path to trained Judge binary (judge.bin)",
    )
    online_p.add_argument("--rounds", type=int, default=20,
                          help="Number of collect→train cycles")
    online_p.add_argument("--collect-count", type=int, default=100,
                          help="Expressions per collection round")
    online_p.add_argument("--collect-perturbations", type=int, default=5,
                          help="Perturbed trajectories per expression")
    online_p.add_argument("--collect-sigma", type=float, default=0.1,
                          help="Weight perturbation std dev")
    online_p.add_argument("--collect-threshold", type=float, default=0.3,
                          help="Rule application threshold")
    online_p.add_argument("--collect-max-epochs", type=int, default=50,
                          help="Max e-graph epochs per trajectory")
    online_p.add_argument("--train-epochs-per-round", type=int, default=3,
                          help="OPD training epochs per collection round")
    online_p.add_argument("--batch-size", type=int, default=64)
    online_p.add_argument("--lr", type=float, default=5e-4)
    online_p.add_argument("--weight-decay", type=float, default=1e-4)
    online_p.add_argument("--ema-decay", type=float, default=0.999)
    online_p.add_argument("--hidden", type=int, default=128)
    online_p.add_argument("--heads", type=int, default=4)
    online_p.add_argument("--layers", type=int, default=4)
    online_p.add_argument("--dropout", type=float, default=0.1)
    online_p.add_argument("--output-dir", default="guide_online",
                          help="Directory for all artifacts")
    online_p.add_argument("--warm-start", default=None,
                          help="Path to existing student .bin for warm-start")
    online_p.add_argument("--seed", type=int, default=42)
    online_p.add_argument("--device", default=None)
    online_p.add_argument(
        "--corpus",
        default="pixelflow-pipeline/data/bench_corpus.jsonl",
        help="Path to expression corpus JSONL passed to train_online",
    )
    online_p.add_argument(
        "--judge-update-every", type=int, default=5,
        help="Run judge update every N policy rounds (0 = disabled)",
    )
    online_p.add_argument(
        "--judge-update-rounds", type=int, default=5,
        help="Number of train_online adversarial rounds per judge update",
    )
    online_p.add_argument(
        "--judge-update-bench-budget", type=int, default=64,
        help="JIT benchmark budget per train_online round",
    )

    args = p.parse_args()

    commands = {
        "train": cmd_train,
        "eval": cmd_eval,
        "export-distill": cmd_export_distill,
        "train-policy": cmd_train_policy,
        "eval-policy": cmd_eval_policy,
        "export-guide": cmd_export_guide,
        "online-train": cmd_online_train,
    }
    cmd_fn = commands.get(args.command)
    if cmd_fn is None:
        p.print_help()
        sys.exit(1)
    cmd_fn(args)


if __name__ == "__main__":
    main()
