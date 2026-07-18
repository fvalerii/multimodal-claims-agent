#!/usr/bin/env python3
"""
Render a LinkedIn-ready architecture diagram for the Multimodal Claims Agent.

Requires (system):
  graphviz   # conda: conda install -c conda-forge graphviz
             # apt:   sudo apt install graphviz
             # brew:  brew install graphviz

Requires (Python):
  pip install diagrams

Usage:
  python scripts/render_architecture.py
  # → claims-agent-architecture.png at the repo root
"""

from __future__ import annotations

from pathlib import Path

from diagrams import Cluster, Diagram, Edge
from diagrams.gcp.ml import AIPlatform
from diagrams.onprem.client import User
from diagrams.onprem.compute import Server
from diagrams.onprem.monitoring import Prometheus
from diagrams.onprem.security import Vault
from diagrams.programming.language import Python


REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT = REPO_ROOT / "claims-agent-architecture"


def main() -> None:
    # High-res PNG for LinkedIn thumbnails (300 DPI).
    graph_attr = {
        "splines": "spline",
        "nodesep": "0.70",
        "ranksep": "1.15",
        "pad": "0.75",
        "bgcolor": "white",
        "fontname": "Helvetica",
        "dpi": "300",
        "label": "Multimodal Claims Agent — Architecture",
        "labelloc": "t",
        "fontsize": "24",
        "fontcolor": "#1a1a2e",
    }
    node_attr = {
        "fontsize": "13",
        "fontname": "Helvetica",
        "height": "1.15",
        "width": "1.7",
    }
    edge_attr = {
        "fontsize": "12",
        "fontname": "Helvetica",
        "color": "#4a5568",
    }

    with Diagram(
        "",
        filename=str(OUTPUT),
        show=False,
        direction="LR",
        outformat="png",
        graph_attr=graph_attr,
        node_attr=node_attr,
        edge_attr=edge_attr,
    ):
        # 1. Ingress — messy multimodal payload
        payload = User("Messy Payload\n(Text + Images)")

        # 2. Semantic safety / scope gate
        guardrail = Vault("Semantic\nGuardrail")

        # Path A — blocked
        rejected = Prometheus("Rejected /\nLogged")

        # 3–5. Isolated LangGraph execution state
        with Cluster("Isolated LangGraph State"):
            router = Server("LangGraph\nVision Router")

            with Cluster("Vision Language Models"):
                claude = AIPlatform("Claude 3.5 Sonnet\n(Vehicles & Tech)")
                gemini = AIPlatform("Gemini 1.5 Flash\n(Packages)")

            validator = Python(
                "Deterministic Fallback\n& Schema Validator\n(Pydantic)"
            )

        # Final clean artefact
        output = Server("Clean Output\n(CSV / JSON)")

        # ---- Edges (exact portfolio flow) --------------------------------
        payload >> Edge(label="ingest", color="#2b6cb0", penwidth="2.0") >> guardrail

        guardrail >> Edge(
            label="Path A — Blocked",
            color="#c53030",
            style="dashed",
            penwidth="1.8",
        ) >> rejected

        guardrail >> Edge(
            label="Path B — Allowed",
            color="#276749",
            penwidth="2.0",
        ) >> router

        router >> Edge(label="car / laptop", color="#553c9a") >> claude
        router >> Edge(label="package", color="#2c7a7b") >> gemini

        claude >> Edge(color="#2d3748", penwidth="1.6") >> validator
        gemini >> Edge(color="#2d3748", penwidth="1.6") >> validator

        validator >> Edge(
            label="schema-valid",
            color="#2b6cb0",
            penwidth="2.0",
        ) >> output

    png = Path(f"{OUTPUT}.png")
    if not png.exists():
        raise SystemExit(f"Expected output missing: {png}")
    print(f"Wrote {png}  ({png.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
