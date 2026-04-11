from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, learning_curve, train_test_split

# Same column order as training/inference rows — must stay aligned with claim_level_df / score_claim.
GRAPH_MODEL_FEATURES: List[str] = [
    "claim_amount",
    "patient_degree",
    "provider_degree",
    "claim_degree",
    "patient_clustering",
    "provider_clustering",
    "claim_clustering",
    "patient_shared_connections",
    "provider_shared_connections",
    "claim_shared_connections",
    "patient_claim_frequency",
    "provider_claim_frequency",
    "claim_frequency",
]


@dataclass
class ModelReport:
    """Test-set metrics from `train_model`; train_* set when a supervised RF fit runs."""

    accuracy: float
    f1: float
    samples: int
    train_accuracy: float | None = None
    train_f1: float | None = None


@dataclass
class CrossValidationReport:
    """Stratified k-fold summary: macro averages and per-class precision/recall/F1."""

    n_splits: int
    n_samples: int
    mean_accuracy: float
    std_accuracy: float
    mean_precision_macro: float
    std_precision_macro: float
    mean_recall_macro: float
    std_recall_macro: float
    mean_f1_macro: float
    std_f1_macro: float
    precision_by_class: Dict[int, float]
    recall_by_class: Dict[int, float]
    f1_by_class: Dict[int, float]
    fold_accuracy: List[float]


@dataclass
class LearningCurveReport:
    """Learning curve at increasing training fractions; use `gap_at_max_training` for overfitting signal."""

    train_sizes: List[int]
    train_scores_mean: List[float]
    train_scores_std: List[float]
    val_scores_mean: List[float]
    val_scores_std: List[float]
    gap_at_max_training: float


class GraphFraudDetector:
    """Build a claim graph, derive graph features, detect patterns, and score fraud."""

    def __init__(
        self,
        claims_csv_path: str | Path,
        synthetic_csv_path: str | Path | None = None,
    ) -> None:
        self.claims_csv_path = Path(claims_csv_path)
        self.synthetic_csv_path = Path(synthetic_csv_path) if synthetic_csv_path else None
        self.graph = nx.Graph()
        self.claims_df = pd.DataFrame()
        self.node_features = pd.DataFrame()
        self.claim_level_df = pd.DataFrame()
        self.model = RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced",
        )
        self.model_report: Optional[ModelReport] = None
        self._is_trained = False
        self._model_kind: str = "rf"
        self._feature_columns: List[str] = list(GRAPH_MODEL_FEATURES)
        self._train_feature_mean: Optional[pd.Series] = None
        self._train_feature_std: Optional[pd.Series] = None
        self._anomaly_scale: float = 1.0

    @staticmethod
    def _as_id(value: Any) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, float) and float(value).is_integer():
            return str(int(value))
        return str(value).strip()

    @staticmethod
    def _require_graph_id(claim_data: Dict[str, Any], key: str) -> str:
        raw = claim_data.get(key)
        if raw is None:
            raise ValueError(f"Graph scoring requires '{key}' (must match API ClaimInput).")
        s = GraphFraudDetector._as_id(raw)
        if not s:
            raise ValueError(f"Graph scoring requires a non-empty '{key}'.")
        return s

    def load_dataset(self, use_synthetic: bool = False, max_rows: int | None = None) -> pd.DataFrame:
        if use_synthetic and self.synthetic_csv_path and self.synthetic_csv_path.exists():
            df = pd.read_csv(self.synthetic_csv_path, nrows=max_rows)
            renamed = df.rename(
                columns={
                    "Claim_ID": "claim_id",
                    "Patient_ID": "patient_id",
                    "Hospital_ID": "provider_id",
                    "Claim_Date": "claim_date",
                    "Claim_Amount": "claim_amount",
                    "Is_Fraudulent": "is_fraudulent",
                }
            )
            cols = ["claim_id", "patient_id", "provider_id", "claim_date", "claim_amount", "is_fraudulent"]
            self.claims_df = renamed[cols].copy()
        else:
            df = pd.read_csv(self.claims_csv_path, nrows=max_rows)
            expected = ["claim_id", "patient_id", "provider_id", "claim_date", "claim_amount", "status"]
            missing = [c for c in expected if c not in df.columns]
            if missing:
                raise ValueError(f"claims.csv is missing columns: {missing}")
            self.claims_df = df[expected].copy()
            # Weak label proxy for model training when true labels are unavailable.
            self.claims_df["is_fraudulent"] = self.claims_df["status"].astype(str).str.lower().eq("rejected")

        self.claims_df["claim_id"] = self.claims_df["claim_id"].map(self._as_id)
        self.claims_df["patient_id"] = self.claims_df["patient_id"].map(self._as_id)
        self.claims_df["provider_id"] = self.claims_df["provider_id"].map(self._as_id)
        for col in ("claim_id", "patient_id", "provider_id"):
            self.claims_df = self.claims_df[self.claims_df[col].astype(str).str.len() > 0]
        self.claims_df["claim_date"] = pd.to_datetime(self.claims_df["claim_date"], errors="coerce")
        self.claims_df["claim_amount"] = pd.to_numeric(self.claims_df["claim_amount"], errors="coerce").fillna(0.0)
        self.claims_df["is_fraudulent"] = self.claims_df["is_fraudulent"].astype(bool)
        self.claims_df = self.claims_df.dropna(subset=["claim_date"]).reset_index(drop=True)
        return self.claims_df

    def build_graph(self) -> nx.Graph:
        if self.claims_df.empty:
            self.load_dataset()

        g = nx.Graph()
        for row in self.claims_df.itertuples(index=False):
            patient_node = f"patient::{row.patient_id}"
            provider_node = f"provider::{row.provider_id}"
            claim_node = f"claim::{row.claim_id}"

            g.add_node(patient_node, node_type="patient", entity_id=row.patient_id)
            g.add_node(provider_node, node_type="provider", entity_id=row.provider_id)
            g.add_node(claim_node, node_type="claim", entity_id=row.claim_id)

            g.add_edge(patient_node, claim_node, edge_type="patient_claim")
            g.add_edge(provider_node, claim_node, edge_type="provider_claim")
            g.add_edge(patient_node, provider_node, edge_type="patient_provider")

        self.graph = g
        return self.graph

    def _shared_connections_graph(self, graph: nx.Graph, node: str) -> int:
        if node not in graph:
            return 0
        neighbors = list(graph.neighbors(node))
        if not neighbors:
            return 0
        second_hop: set[str] = set()
        neighbor_set = set(neighbors)
        for nb in neighbors:
            for two_hop in graph.neighbors(nb):
                if two_hop != node and two_hop not in neighbor_set:
                    second_hop.add(two_hop)
        return len(second_hop)

    def _shared_connections(self, node: str) -> int:
        return self._shared_connections_graph(self.graph, node)

    @staticmethod
    def _claim_frequency_for_node(graph: nx.Graph, node: str) -> int:
        if node not in graph:
            return 0
        attrs = graph.nodes[node]
        ntype = attrs.get("node_type")
        if ntype == "claim":
            return 1
        claim_neighbors = [
            nb for nb in graph.neighbors(node) if graph.nodes[nb].get("node_type") == "claim"
        ]
        return len(claim_neighbors)

    def _claim_frequency_map(self) -> Dict[str, int]:
        mapping: Dict[str, int] = {}
        for node in self.graph.nodes():
            mapping[node] = self._claim_frequency_for_node(self.graph, node)
        return mapping

    def _single_node_features(self, graph: nx.Graph, node: str) -> Dict[str, float]:
        if node not in graph:
            return {
                "degree_centrality": 0.0,
                "clustering_coefficient": 0.0,
                "shared_connections": 0.0,
                "claim_frequency": 0.0,
            }
        n_nodes = graph.number_of_nodes()
        deg_cent = graph.degree(node) / (n_nodes - 1) if n_nodes > 1 else 0.0
        return {
            "degree_centrality": float(deg_cent),
            "clustering_coefficient": float(nx.clustering(graph, node)),
            "shared_connections": float(self._shared_connections_graph(graph, node)),
            "claim_frequency": float(self._claim_frequency_for_node(graph, node)),
        }

    def _augmented_graph_for_claim(self, patient_id: str, provider_id: str, claim_id: str) -> nx.Graph:
        """Graph used at inference: historical graph plus this claim's patient–provider–claim triangle."""
        g = self.graph.copy()
        patient_node = f"patient::{patient_id}"
        provider_node = f"provider::{provider_id}"
        claim_node = f"claim::{claim_id}"
        g.add_node(patient_node, node_type="patient", entity_id=patient_id)
        g.add_node(provider_node, node_type="provider", entity_id=provider_id)
        g.add_node(claim_node, node_type="claim", entity_id=claim_id)
        g.add_edge(patient_node, claim_node, edge_type="patient_claim")
        g.add_edge(provider_node, claim_node, edge_type="provider_claim")
        g.add_edge(patient_node, provider_node, edge_type="patient_provider")
        return g

    def compute_graph_features(self) -> pd.DataFrame:
        if self.graph.number_of_nodes() == 0:
            self.build_graph()

        degree = nx.degree_centrality(self.graph)
        clustering = nx.clustering(self.graph)
        claim_freq = self._claim_frequency_map()

        rows: List[Dict[str, Any]] = []
        for node, attrs in self.graph.nodes(data=True):
            rows.append(
                {
                    "node": node,
                    "node_type": attrs.get("node_type"),
                    "entity_id": attrs.get("entity_id"),
                    "degree_centrality": float(degree.get(node, 0.0)),
                    "clustering_coefficient": float(clustering.get(node, 0.0)),
                    "shared_connections": int(self._shared_connections(node)),
                    "claim_frequency": int(claim_freq.get(node, 0)),
                }
            )
        self.node_features = pd.DataFrame(rows)
        self.claim_level_df = self._build_claim_level_features()
        return self.node_features

    def _node_feature_row(self, node_key: str) -> Dict[str, float]:
        if self.node_features.empty:
            self.compute_graph_features()
        row = self.node_features[self.node_features["node"] == node_key]
        if row.empty:
            return {
                "degree_centrality": 0.0,
                "clustering_coefficient": 0.0,
                "shared_connections": 0.0,
                "claim_frequency": 0.0,
            }
        as_dict = row.iloc[0].to_dict()
        return {
            "degree_centrality": float(as_dict["degree_centrality"]),
            "clustering_coefficient": float(as_dict["clustering_coefficient"]),
            "shared_connections": float(as_dict["shared_connections"]),
            "claim_frequency": float(as_dict["claim_frequency"]),
        }

    def _build_claim_level_features(self) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        for rec in self.claims_df.itertuples(index=False):
            patient_node = f"patient::{rec.patient_id}"
            provider_node = f"provider::{rec.provider_id}"
            claim_node = f"claim::{rec.claim_id}"
            p = self._node_feature_row(patient_node)
            pr = self._node_feature_row(provider_node)
            c = self._node_feature_row(claim_node)
            rows.append(
                {
                    "claim_id": rec.claim_id,
                    "patient_id": rec.patient_id,
                    "provider_id": rec.provider_id,
                    "claim_amount": float(rec.claim_amount),
                    "label": int(bool(rec.is_fraudulent)),
                    "patient_degree": p["degree_centrality"],
                    "provider_degree": pr["degree_centrality"],
                    "claim_degree": c["degree_centrality"],
                    "patient_clustering": p["clustering_coefficient"],
                    "provider_clustering": pr["clustering_coefficient"],
                    "claim_clustering": c["clustering_coefficient"],
                    "patient_shared_connections": p["shared_connections"],
                    "provider_shared_connections": pr["shared_connections"],
                    "claim_shared_connections": c["shared_connections"],
                    "patient_claim_frequency": p["claim_frequency"],
                    "provider_claim_frequency": pr["claim_frequency"],
                    "claim_frequency": c["claim_frequency"],
                }
            )
        return pd.DataFrame(rows)

    def detect_fraud_patterns(
        self,
        provider_degree_quantile: float = 0.99,
        provider_patient_min: int = 8,
        burst_days: int = 7,
        burst_claim_threshold: int = 5,
        repeated_pair_min: int = 3,
    ) -> Dict[str, List[Dict[str, Any]]]:
        if self.node_features.empty:
            self.compute_graph_features()

        patterns: Dict[str, List[Dict[str, Any]]] = {
            "high_degree_providers": [],
            "patient_clusters_same_provider": [],
            "repeated_claim_patterns": [],
            "short_time_bursts": [],
        }

        providers = self.node_features[self.node_features["node_type"] == "provider"].copy()
        if not providers.empty:
            threshold = float(providers["degree_centrality"].quantile(provider_degree_quantile))
            risky = providers[providers["degree_centrality"] >= threshold].sort_values("degree_centrality", ascending=False)
            for r in risky.head(20).itertuples(index=False):
                patterns["high_degree_providers"].append(
                    {
                        "provider_id": r.entity_id,
                        "degree_centrality": round(float(r.degree_centrality), 6),
                        "claim_frequency": int(r.claim_frequency),
                    }
                )

        provider_patient = self.claims_df.groupby("provider_id")["patient_id"].nunique().reset_index(name="unique_patients")
        for r in provider_patient[provider_patient["unique_patients"] >= provider_patient_min].itertuples(index=False):
            patterns["patient_clusters_same_provider"].append(
                {"provider_id": r.provider_id, "unique_patients": int(r.unique_patients)}
            )

        repeated = (
            self.claims_df.assign(amount_bucket=(self.claims_df["claim_amount"] / 100).round().astype(int))
            .groupby(["patient_id", "provider_id", "amount_bucket"])
            .size()
            .reset_index(name="count")
        )
        repeated = repeated[repeated["count"] >= repeated_pair_min]
        for r in repeated.sort_values("count", ascending=False).head(20).itertuples(index=False):
            patterns["repeated_claim_patterns"].append(
                {
                    "patient_id": r.patient_id,
                    "provider_id": r.provider_id,
                    "amount_bucket": int(r.amount_bucket),
                    "count": int(r.count),
                }
            )

        bursts = self._detect_bursts(window_days=burst_days, threshold=burst_claim_threshold)
        patterns["short_time_bursts"].extend(bursts)
        return patterns

    def _detect_bursts(self, window_days: int, threshold: int) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        by_provider = self.claims_df.sort_values("claim_date").groupby("provider_id")
        for provider_id, frame in by_provider:
            dates = frame["claim_date"].tolist()
            left = 0
            max_count = 0
            for right, d in enumerate(dates):
                while d - dates[left] > timedelta(days=window_days):
                    left += 1
                max_count = max(max_count, right - left + 1)
            if max_count >= threshold:
                out.append(
                    {
                        "provider_id": provider_id,
                        "max_claims_in_window": int(max_count),
                        "window_days": int(window_days),
                    }
                )
        return sorted(out, key=lambda x: x["max_claims_in_window"], reverse=True)[:20]

    def train_model(self, test_size: float = 0.25, random_state: int = 42) -> ModelReport:
        if self.claim_level_df.empty:
            self.compute_graph_features()

        df = self.claim_level_df.copy()
        features = self._feature_columns
        X = df[features]
        y = df["label"].astype(int)

        if y.nunique() < 2:
            # No fraud labels to separate: unsupervised anomaly scoring vs training distribution.
            self._model_kind = "anomaly"
            self._train_feature_mean = X.mean()
            self._train_feature_std = X.std().replace(0, 1e-9)
            z = ((X - self._train_feature_mean) / self._train_feature_std).abs()
            mean_z = z.mean(axis=1)
            self._anomaly_scale = float(max(mean_z.quantile(0.95), 1e-6))
            self.model_report = ModelReport(
                accuracy=0.0,
                f1=0.0,
                samples=int(len(df)),
            )
            self._is_trained = True
            return self.model_report

        self._model_kind = "rf"
        self._train_feature_mean = None
        self._train_feature_std = None

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )
        self.model.fit(X_train, y_train)
        train_preds = self.model.predict(X_train)
        preds = self.model.predict(X_test)
        report = ModelReport(
            accuracy=float(accuracy_score(y_test, preds)),
            f1=float(f1_score(y_test, preds, zero_division=0)),
            samples=int(len(df)),
            train_accuracy=float(accuracy_score(y_train, train_preds)),
            train_f1=float(f1_score(y_train, train_preds, zero_division=0)),
        )
        self.model_report = report
        self._is_trained = True
        return report

    def _supervised_xy(self) -> tuple[pd.DataFrame, pd.Series] | None:
        if self.claim_level_df.empty:
            self.compute_graph_features()
        df = self.claim_level_df
        if df.empty:
            return None
        y = df["label"].astype(int)
        if y.nunique() < 2:
            return None
        return df[self._feature_columns], y

    def cross_validate_stratified(
        self,
        n_splits: int = 5,
        random_state: int = 42,
    ) -> CrossValidationReport:
        """Stratified k-fold CV with macro and per-class precision, recall, and F1."""
        prepared = self._supervised_xy()
        if prepared is None:
            raise ValueError(
                "Stratified cross-validation requires non-empty claim features and at least two label classes."
            )
        X, y = prepared
        min_class = int(y.value_counts().min())
        effective_splits = min(n_splits, min_class)
        if effective_splits < 3:
            raise ValueError(
                "Stratified CV needs at least 3 samples in the minority class and "
                f"n_splits ≤ minority count; got minority={min_class}, n_splits={n_splits}."
            )

        skf = StratifiedKFold(n_splits=effective_splits, shuffle=True, random_state=random_state)
        fold_acc: List[float] = []
        fold_prec_m: List[float] = []
        fold_rec_m: List[float] = []
        fold_f1_m: List[float] = []
        prec_c0: List[float] = []
        prec_c1: List[float] = []
        rec_c0: List[float] = []
        rec_c1: List[float] = []
        f1_c0: List[float] = []
        f1_c1: List[float] = []

        for train_idx, test_idx in skf.split(X, y):
            clf = clone(self.model)
            X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
            y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
            clf.fit(X_tr, y_tr)
            pred = clf.predict(X_te)
            fold_acc.append(float(accuracy_score(y_te, pred)))
            fold_prec_m.append(float(precision_score(y_te, pred, average="macro", zero_division=0)))
            fold_rec_m.append(float(recall_score(y_te, pred, average="macro", zero_division=0)))
            fold_f1_m.append(float(f1_score(y_te, pred, average="macro", zero_division=0)))
            p_arr, r_arr, f_arr, _ = precision_recall_fscore_support(
                y_te, pred, labels=[0, 1], zero_division=0
            )
            prec_c0.append(float(p_arr[0]))
            prec_c1.append(float(p_arr[1]))
            rec_c0.append(float(r_arr[0]))
            rec_c1.append(float(r_arr[1]))
            f1_c0.append(float(f_arr[0]))
            f1_c1.append(float(f_arr[1]))

        return CrossValidationReport(
            n_splits=effective_splits,
            n_samples=int(len(y)),
            mean_accuracy=float(np.mean(fold_acc)),
            std_accuracy=float(np.std(fold_acc)),
            mean_precision_macro=float(np.mean(fold_prec_m)),
            std_precision_macro=float(np.std(fold_prec_m)),
            mean_recall_macro=float(np.mean(fold_rec_m)),
            std_recall_macro=float(np.std(fold_rec_m)),
            mean_f1_macro=float(np.mean(fold_f1_m)),
            std_f1_macro=float(np.std(fold_f1_m)),
            precision_by_class={0: float(np.mean(prec_c0)), 1: float(np.mean(prec_c1))},
            recall_by_class={0: float(np.mean(rec_c0)), 1: float(np.mean(rec_c1))},
            f1_by_class={0: float(np.mean(f1_c0)), 1: float(np.mean(f1_c1))},
            fold_accuracy=fold_acc,
        )

    def learning_curve_stratified(
        self,
        n_splits: int = 5,
        random_state: int = 42,
        train_sizes: np.ndarray | None = None,
        scoring: str = "f1_macro",
    ) -> LearningCurveReport:
        """Bias–variance / overfitting diagnostic: training vs validation score vs training set size."""
        prepared = self._supervised_xy()
        if prepared is None:
            raise ValueError(
                "Learning curves require non-empty claim features and at least two label classes."
            )
        X, y = prepared
        min_class = int(y.value_counts().min())
        effective_splits = min(n_splits, min_class)
        if effective_splits < 3:
            raise ValueError(
                "Learning curve CV needs at least 3 samples in the minority class; "
                f"got minority={min_class}, n_splits={n_splits}."
            )

        if train_sizes is None:
            train_sizes = np.linspace(0.2, 1.0, 6)

        cv = StratifiedKFold(n_splits=effective_splits, shuffle=True, random_state=random_state)
        sizes, train_scores, val_scores = learning_curve(
            clone(self.model),
            X,
            y,
            train_sizes=train_sizes,
            cv=cv,
            scoring=scoring,
            n_jobs=1,
            random_state=random_state,
        )
        train_mean = train_scores.mean(axis=1)
        val_mean = val_scores.mean(axis=1)
        train_std = train_scores.std(axis=1)
        val_std = val_scores.std(axis=1)
        gap = float(train_mean[-1] - val_mean[-1])
        return LearningCurveReport(
            train_sizes=[int(s) for s in sizes],
            train_scores_mean=[float(x) for x in train_mean],
            train_scores_std=[float(x) for x in train_std],
            val_scores_mean=[float(x) for x in val_mean],
            val_scores_std=[float(x) for x in val_std],
            gap_at_max_training=gap,
        )

    def _predict_fraud_proba(self, row: pd.DataFrame) -> float:
        X = row[self._feature_columns]
        if self._model_kind == "anomaly":
            assert self._train_feature_mean is not None and self._train_feature_std is not None
            z = ((X - self._train_feature_mean) / self._train_feature_std).abs()
            mean_z = float(z.mean(axis=1).iloc[0])
            scale = max(self._anomaly_scale, 1e-6)
            return float(min(1.0, max(0.0, mean_z / (scale * 1.5))))

        return float(self.model.predict_proba(X)[:, 1][0])

    def _risk_row_from_claim(self, claim_data: Dict[str, Any]) -> pd.DataFrame:
        patient_id = self._require_graph_id(claim_data, "patient_id")
        provider_id = self._require_graph_id(claim_data, "provider_id")
        claim_id = self._require_graph_id(claim_data, "claim_id")
        raw_amt = claim_data.get("amount", claim_data.get("claim_amount", 0.0)) or 0.0
        try:
            amount = float(raw_amt)
        except (TypeError, ValueError):
            amount = 0.0

        g = self._augmented_graph_for_claim(patient_id, provider_id, claim_id)
        patient_node = f"patient::{patient_id}"
        provider_node = f"provider::{provider_id}"
        claim_node = f"claim::{claim_id}"

        p = self._single_node_features(g, patient_node)
        pr = self._single_node_features(g, provider_node)
        c = self._single_node_features(g, claim_node)
        row = {
            "claim_amount": amount,
            "patient_degree": p["degree_centrality"],
            "provider_degree": pr["degree_centrality"],
            "claim_degree": c["degree_centrality"],
            "patient_clustering": p["clustering_coefficient"],
            "provider_clustering": pr["clustering_coefficient"],
            "claim_clustering": c["clustering_coefficient"],
            "patient_shared_connections": p["shared_connections"],
            "provider_shared_connections": pr["shared_connections"],
            "claim_shared_connections": c["shared_connections"],
            "patient_claim_frequency": p["claim_frequency"],
            "provider_claim_frequency": pr["claim_frequency"],
            "claim_frequency": c["claim_frequency"],
        }
        return pd.DataFrame([row])

    def score_claim(self, claim_data: Dict[str, Any]) -> Dict[str, Any]:
        if not self._is_trained:
            self.train_model()
        patterns = self.detect_fraud_patterns()
        row = self._risk_row_from_claim(claim_data)
        proba = self._predict_fraud_proba(row)

        provider_id = self._require_graph_id(claim_data, "provider_id")
        patient_id = self._require_graph_id(claim_data, "patient_id")
        claim_id = self._require_graph_id(claim_data, "claim_id")

        risk_nodes = [
            f"patient::{patient_id}",
            f"provider::{provider_id}",
            f"claim::{claim_id}",
        ]

        high_degree_ids = {str(p["provider_id"]) for p in patterns["high_degree_providers"]}
        burst_ids = {str(x["provider_id"]) for x in patterns["short_time_bursts"]}
        cluster_ids = {str(x["provider_id"]) for x in patterns["patient_clusters_same_provider"]}

        detected = "no_graph_pattern"
        if provider_id in burst_ids:
            detected = "short_time_burst"
        elif provider_id in cluster_ids:
            detected = "patient_cluster_same_provider"
        elif provider_id in high_degree_ids:
            detected = "high_degree_provider"

        return {
            "fraud_probability": round(proba, 4),
            "pattern_detected": detected,
            "risk_nodes": sorted(set(risk_nodes)),
        }

    @staticmethod
    def generate_synthetic_fraud_scenarios(
        normal_claims: int = 180,
        collusion_ring_claims: int = 80,
        burst_claims: int = 80,
        random_state: int = 42,
    ) -> pd.DataFrame:
        rng = np.random.default_rng(random_state)
        rows: List[Dict[str, Any]] = []
        base_date = pd.Timestamp("2025-01-01")

        # Normal traffic
        for i in range(normal_claims):
            rows.append(
                {
                    "claim_id": f"N{i}",
                    "patient_id": f"P{rng.integers(1000, 4000)}",
                    "provider_id": f"PR{rng.integers(100, 350)}",
                    "claim_date": base_date + pd.Timedelta(days=int(rng.integers(0, 120))),
                    "claim_amount": float(rng.normal(1200, 350)),
                    "is_fraudulent": False,
                }
            )

        # Collusion ring: small provider set, repeating patient-provider relationships.
        ring_providers = [f"PRC{i}" for i in range(1, 4)]
        ring_patients = [f"PC{i}" for i in range(1, 21)]
        for i in range(collusion_ring_claims):
            rows.append(
                {
                    "claim_id": f"C{i}",
                    "patient_id": ring_patients[i % len(ring_patients)],
                    "provider_id": ring_providers[i % len(ring_providers)],
                    "claim_date": base_date + pd.Timedelta(days=int(rng.integers(5, 45))),
                    "claim_amount": float(rng.choice([4999, 5001, 5020, 4970])),
                    "is_fraudulent": True,
                }
            )

        # Abnormal provider behavior: one provider generates high-volume bursts.
        burst_provider = "PR_BURST"
        for i in range(burst_claims):
            rows.append(
                {
                    "claim_id": f"B{i}",
                    "patient_id": f"PB{rng.integers(1, 120)}",
                    "provider_id": burst_provider,
                    "claim_date": base_date + pd.Timedelta(days=int(rng.integers(20, 27))),
                    "claim_amount": float(rng.normal(3100, 450)),
                    "is_fraudulent": True,
                }
            )

        out = pd.DataFrame(rows)
        out["claim_amount"] = out["claim_amount"].clip(lower=10.0)
        return out

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> "GraphFraudDetector":
        detector = cls(claims_csv_path="claims.csv")
        detector.claims_df = df.copy()
        detector.claims_df["claim_date"] = pd.to_datetime(detector.claims_df["claim_date"], errors="coerce")
        detector.claims_df["claim_amount"] = pd.to_numeric(detector.claims_df["claim_amount"], errors="coerce").fillna(0.0)
        detector.claims_df["is_fraudulent"] = detector.claims_df["is_fraudulent"].astype(bool)
        detector.claims_df = detector.claims_df.dropna(subset=["claim_date"]).reset_index(drop=True)
        return detector


_detector_singleton: Optional[GraphFraudDetector] = None


def get_graph_detector(base_dir: str | Path) -> GraphFraudDetector:
    global _detector_singleton
    if _detector_singleton is None:
        base = Path(base_dir)
        _detector_singleton = GraphFraudDetector(
            claims_csv_path=base / "claims.csv",
            synthetic_csv_path=base / "synthetic_health_claims.csv",
        )
        _detector_singleton.load_dataset(use_synthetic=True, max_rows=5000)
        _detector_singleton.build_graph()
        _detector_singleton.compute_graph_features()
        _detector_singleton.train_model()
    return _detector_singleton
