# main.py
from __future__ import annotations

import json
import os
import sys
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.text import Text
from rich import box

from config import config

app = typer.Typer(help="DB Intelligence Agent — Automated database analysis with AI.")
console = Console()


def print_banner():
    console.print(Panel(
        Text.from_markup(
            "\n[bold blue]🧠 DB Intelligence Agent[/bold blue]\n\n"
            "[dim]Schema Extraction · Quality Profiling · Causal Intelligence · AI Narration[/dim]\n\n"
            "[yellow]⭐ Exceptional Feature: Causal Intelligence Engine[/yellow]\n"
            "[dim]Discovers directional column influences · Predictive Health Scores · Auto YAML Contracts[/dim]\n"
        ),
        title="[bold white]Welcome[/bold white]",
        border_style="blue",
        padding=(0, 2),
    ))


@app.command()
def analyze(
    db_url: Optional[str] = typer.Option(None, "--db-url", help="SQLAlchemy DB URL"),
    output_dir: Optional[str] = typer.Option(None, "--output-dir", help="Where to save reports"),
    sample_size: Optional[int] = typer.Option(None, "--sample-size", help="Rows sampled per table"),
    no_ai: bool = typer.Option(False, "--no-ai", help="Skip Claude AI narration"),
    show_contracts: bool = typer.Option(False, "--show-contracts", help="Print YAML contracts to terminal"),
):
    """Run full DB analysis pipeline and save reports."""

    # ── Apply overrides ───────────────────────────────────────────────────────
    if db_url:
        config.DB_URL = db_url
    if output_dir:
        config.OUTPUT_DIR = output_dir
    if sample_size:
        config.SAMPLE_SIZE = sample_size
    if no_ai:
        config.ANTHROPIC_API_KEY = ""

    print_banner()

    # ── Import all modules here (avoids slow startup for --help) ─────────────
    from connectors.db_connector import DBConnector
    from analyzers.schema_analyzer import SchemaAnalyzer
    from analyzers.relationship_mapper import RelationshipMapper
    from analyzers.quality_profiler import QualityProfiler
    from analyzers.causal_engine import CausalIntelligenceEngine
    from generators.ai_narrator import AINarrator
    from generators.report_builder import ReportBuilder

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=False,
    ) as progress:

        # ── STEP 1: Connect ───────────────────────────────────────────────────
        task = progress.add_task("[cyan]Connecting to database...", total=None)
        connector = DBConnector()
        if not connector.test_connection():
            console.print(f"[bold red]❌ Cannot connect to: {config.DB_URL}[/bold red]")
            console.print("[yellow]Tip: Run python create_demo_db.py first to create a demo database.[/yellow]")
            raise typer.Exit(1)

        tables = connector.get_table_names()
        progress.update(task, description=f"[green]✅ Connected — {len(tables)} tables found")

        # ── STEP 2: Schema ────────────────────────────────────────────────────
        progress.update(task, description="[cyan]📋 Extracting schema...")
        schema = SchemaAnalyzer(connector).analyze()

        # ── STEP 3: ER Map ────────────────────────────────────────────────────
        progress.update(task, description="[cyan]🔗 Building ER relationship map...")
        mapper = RelationshipMapper(schema)
        mapper.build()
        er_summary = mapper.to_dict()

        # ── STEP 4: Quality ───────────────────────────────────────────────────
        progress.update(task, description=f"[cyan]💊 Sampling & profiling data quality ({config.SAMPLE_SIZE:,} rows/table)...")
        samples = connector.get_all_samples()
        quality = QualityProfiler(schema, samples).profile_all()

        # ── STEP 5: Causal Intelligence Engine ⭐ ─────────────────────────────
        progress.update(task, description="[yellow]⭐ Running Causal Intelligence Engine (Phase 1-4)...")
        cie = CausalIntelligenceEngine(schema, samples, quality)

        progress.update(task, description="[yellow]  ⭐ Phase 1: Discovering causal relationships...")
        causal_rels = cie.discover_causal_relationships()

        progress.update(task, description="[yellow]  ⭐ Phase 2: Computing predictive health scores...")
        health_scores = cie.compute_health_scores()

        progress.update(task, description="[yellow]  ⭐ Phase 3: Generating YAML data contracts...")
        data_contracts = cie.generate_data_contracts()

        progress.update(task, description="[yellow]  ⭐ Phase 4: Checking schema drift...")
        snapshot_path = os.path.join(config.OUTPUT_DIR, "schema_snapshot_latest.json")
        drift_report = {}
        if os.path.exists(snapshot_path):
            try:
                with open(snapshot_path) as f:
                    baseline = json.load(f)
                drift_report = cie.detect_schema_drift(baseline)
            except Exception:
                drift_report = {"severity": "None", "drift_detected": False}
        else:
            drift_report = {
                "drift_detected": False,
                "severity": "None",
                "message": "No baseline. Snapshot saved — run again to detect drift.",
            }

        # ── STEP 6: AI Narration ──────────────────────────────────────────────
        if config.ANTHROPIC_API_KEY:
            progress.update(task, description="[magenta]🤖 Generating AI narratives (Claude)...")
        else:
            progress.update(task, description="[dim]🤖 AI narration skipped (no API key)...")

        narrator = AINarrator()
        schema_dict = json.loads(schema.model_dump_json())
        ai_summary = narrator.generate_schema_summary(schema_dict, er_summary)
        table_summaries = narrator.generate_table_summaries(schema_dict, quality)
        causal_narrative = narrator.generate_causal_narrative(causal_rels)

        # ── STEP 7: Save Reports ──────────────────────────────────────────────
        progress.update(task, description="[cyan]💾 Writing reports to disk...")
        builder = ReportBuilder()
        output_path = builder.build_all(
            schema, er_summary, quality,
            ai_summary, table_summaries,
            causal_rels, health_scores,
            data_contracts, drift_report,
        )
        progress.update(task, description="[green]✅ Analysis complete!")

    # ═════════════════════════════════════════════════════════════════════════
    # PRINT RESULTS TO TERMINAL
    # ═════════════════════════════════════════════════════════════════════════

    console.print()

    # ── AI Summary ────────────────────────────────────────────────────────────
    console.print(Panel(
        f"[white]{ai_summary}[/white]",
        title="[bold magenta]🤖 AI Schema Summary[/bold magenta]",
        border_style="magenta",
    ))

    # ── Schema Table ──────────────────────────────────────────────────────────
    schema_table = Table(
        title="📋 Schema Overview",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold white on blue",
        show_lines=True,
    )
    schema_table.add_column("Table",       style="cyan bold",   min_width=25)
    schema_table.add_column("Rows",        style="white",       justify="right", min_width=10)
    schema_table.add_column("Columns",     style="white",       justify="right", min_width=8)
    schema_table.add_column("PKs",         style="green",       min_width=15)
    schema_table.add_column("FKs",         style="yellow",      justify="right", min_width=5)
    schema_table.add_column("Completeness",style="white",       justify="right", min_width=13)
    schema_table.add_column("Health",      style="white",       min_width=18)

    for tbl_name, tbl in schema.tables.items():
        q = quality.get(tbl_name, {})
        h = health_scores.get(tbl_name)
        comp = q.get("overall_completeness", 0)
        comp_str = f"{comp:.1%}"
        comp_color = "green" if comp > 0.9 else "yellow" if comp > 0.7 else "red"

        if h:
            score_color = "green" if h.label == "Healthy" else "yellow" if h.label == "At Risk" else "red"
            health_str = f"[{score_color}]{h.icon} {h.label} ({h.score}/100)[/{score_color}]"
        else:
            health_str = "—"

        schema_table.add_row(
            tbl_name,
            f"{tbl.row_count:,}",
            str(len(tbl.columns)),
            ", ".join(tbl.primary_keys) or "—",
            str(len(tbl.foreign_keys)),
            f"[{comp_color}]{comp_str}[/{comp_color}]",
            health_str,
        )

    console.print(schema_table)
    console.print()

    # ── ER Relationships ──────────────────────────────────────────────────────
    edges = er_summary.get("edges", [])
    if edges:
        er_table = Table(
            title=f"🔗 Entity Relationships ({len(edges)} found)",
            box=box.SIMPLE_HEAVY,
            header_style="bold white on dark_green",
        )
        er_table.add_column("From Table",    style="cyan")
        er_table.add_column("Column",        style="yellow")
        er_table.add_column("→",             style="white", justify="center")
        er_table.add_column("To Table",      style="green")
        er_table.add_column("Column",        style="yellow")
        er_table.add_column("Cardinality",   style="white")

        for edge in edges:
            er_table.add_row(
                edge["from_table"],
                edge.get("from_col", ""),
                "→",
                edge["to_table"],
                edge.get("to_col", ""),
                edge.get("cardinality", ""),
            )
        console.print(er_table)
        console.print()

    # ── Health Scores ─────────────────────────────────────────────────────────
    health_table = Table(
        title="💊 Predictive Health Scores",
        box=box.ROUNDED,
        header_style="bold white on purple",
        show_lines=False,
    )
    health_table.add_column("Table",         style="cyan bold",  min_width=25)
    health_table.add_column("Score",         style="white",      justify="right", min_width=7)
    health_table.add_column("Status",        style="white",      min_width=15)
    health_table.add_column("Null Score",    justify="right",    min_width=11)
    health_table.add_column("Freshness",     justify="right",    min_width=10)
    health_table.add_column("Completeness",  justify="right",    min_width=13)
    health_table.add_column("PK Unique",     justify="right",    min_width=10)
    health_table.add_column("Recommendation",style="dim",        min_width=40)

    for tbl_name, h in health_scores.items():
        color = "green" if h.label == "Healthy" else "yellow" if h.label == "At Risk" else "red"
        health_table.add_row(
            tbl_name,
            f"[{color}]{h.score}[/{color}]",
            f"[{color}]{h.icon} {h.label}[/{color}]",
            str(h.breakdown["null_score"]),
            str(h.breakdown["freshness_score"]),
            str(h.breakdown["completeness_score"]),
            str(h.breakdown["uniqueness_score"]),
            h.recommendation[:60] + "..." if len(h.recommendation) > 60 else h.recommendation,
        )

    console.print(health_table)
    console.print()

    # ── Causal Relationships ──────────────────────────────────────────────────
    if causal_rels:
        causal_table = Table(
            title=f"⭐ Causal Intelligence — Top Relationships ({len(causal_rels)} discovered)",
            box=box.SIMPLE_HEAVY,
            header_style="bold white on dark_orange3",
        )
        causal_table.add_column("From Column",   style="cyan",   min_width=30)
        causal_table.add_column("Dir",           style="white",  justify="center", min_width=3)
        causal_table.add_column("To Column",     style="green",  min_width=30)
        causal_table.add_column("Method",        style="yellow", min_width=22)
        causal_table.add_column("Strength",      style="white",  justify="right", min_width=9)
        causal_table.add_column("Business Insight", style="dim", min_width=45)

        for rel in causal_rels[:15]:  # Show top 15
            strength_color = "green" if rel.strength > 0.3 else "yellow" if rel.strength > 0.1 else "white"
            causal_table.add_row(
                f"{rel.from_table}.{rel.from_col}",
                rel.direction,
                f"{rel.to_table}.{rel.to_col}",
                rel.method,
                f"[{strength_color}]{rel.strength:.4f}[/{strength_color}]",
                rel.business_insight[:60] + "..." if len(rel.business_insight) > 60 else rel.business_insight,
            )

        console.print(causal_table)
        console.print()

    # ── Causal AI Narrative ───────────────────────────────────────────────────
    console.print(Panel(
        f"[white]{causal_narrative}[/white]",
        title="[bold yellow]⭐ Causal Narrative[/bold yellow]",
        border_style="yellow",
    ))

    # ── Drift Report ──────────────────────────────────────────────────────────
    if drift_report.get("drift_detected"):
        severity = drift_report.get("severity", "None")
        color = "red" if severity == "Breaking" else "yellow"
        console.print(Panel(
            f"[{color}]Severity: {severity}\n{drift_report.get('summary', '')}[/{color}]",
            title="[bold red]🚨 Schema Drift Detected[/bold red]",
            border_style="red",
        ))
    else:
        console.print(Panel(
            "[green]✅ No schema drift detected.[/green]\n"
            "[dim]Schema snapshot saved — run again to enable drift comparison.[/dim]",
            title="[bold green]Schema Drift[/bold green]",
            border_style="green",
        ))

    # ── Show YAML contracts if requested ──────────────────────────────────────
    if show_contracts:
        console.print("\n[bold yellow]📋 Auto-Generated Data Contracts:[/bold yellow]\n")
        for tbl_name, yaml_str in list(data_contracts.items())[:3]:
            console.print(Panel(yaml_str, title=f"[cyan]{tbl_name}_contract.yaml[/cyan]", border_style="cyan"))

    # ── Output Summary ────────────────────────────────────────────────────────
    output_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    output_table.add_column(style="dim")
    output_table.add_column(style="cyan")

    output_table.add_row("📁 Output directory",   output_path)
    output_table.add_row("📄 Master JSON",        f"db_analysis_*.json")
    output_table.add_row("📖 Data Dictionary",    f"data_dictionary_*.md")
    output_table.add_row("📋 Data Contracts",     f"data_contracts/*.yaml  ({len(data_contracts)} files)")
    output_table.add_row("💊 Health Scores",      f"health_scores_*.json")
    output_table.add_row("🔍 Schema Snapshot",    f"schema_snapshot_latest.json")

    console.print(Panel(
        output_table,
        title="[bold white]📂 Generated Reports[/bold white]",
        border_style="blue",
    ))

    console.print()
    console.print("[bold green]✅ Analysis complete![/bold green]")
    console.print(f"[dim]Open the dashboard to explore visually:[/dim]")
    console.print("[bold cyan]  streamlit run ui/dashboard.py[/bold cyan]\n")


if __name__ == "__main__":
    app()