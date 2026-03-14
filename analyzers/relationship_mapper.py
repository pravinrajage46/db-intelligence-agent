# analyzers/relationship_mapper.py
from __future__ import annotations

from typing import Dict, List, Tuple

import networkx as nx

from analyzers.schema_analyzer import SchemaInfo


class RelationshipMapper:
    """
    Builds a directed graph of table relationships from foreign key definitions.
    Computes centrality (hub tables), cardinality, and join paths.
    """

    def __init__(self, schema: SchemaInfo):
        self.schema = schema
        self.graph: nx.DiGraph = nx.DiGraph()

    def build(self) -> "RelationshipMapper":
        """Construct the NetworkX graph from schema FK data."""
        # Add nodes (tables)
        for table_name, table in self.schema.tables.items():
            self.graph.add_node(
                table_name,
                row_count=table.row_count,
                column_count=len(table.columns),
                pk=table.primary_keys,
            )

        # Add edges (foreign key relationships)
        for table_name, table in self.schema.tables.items():
            for fk in table.foreign_keys:
                referred = fk.referred_table
                if not referred or referred not in self.schema.tables:
                    continue

                from_col = fk.constrained_columns[0] if fk.constrained_columns else ""
                to_col = fk.referred_columns[0] if fk.referred_columns else ""
                cardinality = self._infer_cardinality(table_name, fk)

                self.graph.add_edge(
                    table_name,
                    referred,
                    from_col=from_col,
                    to_col=to_col,
                    cardinality=cardinality,
                )

        return self

    def _infer_cardinality(self, from_table: str, fk) -> str:
        """Infer relationship cardinality from key structure."""
        table = self.schema.tables[from_table]
        fk_cols = set(fk.constrained_columns)
        pk_cols = set(table.primary_keys)

        if fk_cols == pk_cols:
            return "one-to-one"
        return "many-to-one"

    def get_hub_tables(self) -> List[Tuple[str, float]]:
        """Tables ranked by degree centrality — most connected first."""
        centrality = nx.degree_centrality(self.graph)
        return sorted(centrality.items(), key=lambda x: x[1], reverse=True)

    def get_join_paths(self, table_a: str, table_b: str) -> List[List[str]]:
        """Find all join paths between two tables."""
        try:
            return list(nx.all_simple_paths(self.graph.to_undirected(), table_a, table_b, cutoff=4))
        except Exception:
            return []

    def to_dict(self) -> Dict:
        """Serialize the ER structure for reports and UI."""
        if self.graph.number_of_nodes() == 0:
            self.build()

        return {
            "nodes": [
                {
                    "table": n,
                    "row_count": self.graph.nodes[n].get("row_count", 0),
                    "column_count": self.graph.nodes[n].get("column_count", 0),
                    "is_hub": self.graph.degree(n) >= 3,
                }
                for n in self.graph.nodes
            ],
            "edges": [
                {
                    "from_table": u,
                    "to_table": v,
                    "from_col": self.graph[u][v].get("from_col", ""),
                    "to_col": self.graph[u][v].get("to_col", ""),
                    "cardinality": self.graph[u][v].get("cardinality", "unknown"),
                }
                for u, v in self.graph.edges
            ],
            "hub_tables": [
                {"table": t, "centrality": round(c, 4)}
                for t, c in self.get_hub_tables()
            ],
            "total_relationships": self.graph.number_of_edges(),
            "isolated_tables": [
                n for n in self.graph.nodes
                if self.graph.degree(n) == 0
            ],
        }