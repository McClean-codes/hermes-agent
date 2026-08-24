#!/usr/bin/env python3
"""CLI entry point for the workflow engine.

This is a thin wrapper around plugins.workflow.engine.WorkflowEngine.
The engine is the single source of truth — this file just provides a CLI surface.
"""
import json
import sys
import sqlite3
from pathlib import Path

# Import the engine from the plugin — single source of truth.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from plugins.workflow.engine import WorkflowEngine  # noqa: E402


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Workflow Execution Engine")
    sub = parser.add_subparsers(dest="command")

    # start
    start = sub.add_parser("start", help="Start a workflow")
    start.add_argument("workflow", help="Workflow name (YAML file in docs/fleet-pipelines/)")
    start.add_argument("--context", "-c", action="append", help="Key=value context pairs (repeatable)")
    start.add_argument("--board", "-b", help="Board slug to use (overrides YAML and auto-create)")
    start.add_argument("--inputs", "-i", action="append", help="Input key=value pairs (repeatable, available as {inputs.<key>})")
    start.add_argument("--dry-run", action="store_true", help="Print plan without executing")
    start.add_argument("--node", help="Start from a specific node (partial execution)")
    start.add_argument("--resume", action="store_true", help="Resume from saved state")
    start.add_argument("--run-id", help="Specific run ID to resume (used by supervisor subprocess)")

    # validate
    validate = sub.add_parser("validate", help="Validate a workflow without executing")
    validate.add_argument("workflow", help="Workflow name to validate")

    # status
    status = sub.add_parser("status", help="Query workflow state")
    status.add_argument("workflow", nargs="?", help="Workflow name (omit for all)")

    # list
    sub.add_parser("list", help="List available workflow definitions")

    # jobs
    jobs = sub.add_parser("jobs", help="List workflow execution history")
    jobs.add_argument("--status", choices=["running", "completed", "failed", "blocked"],
                      help="Filter by status")
    jobs.add_argument("--limit", type=int, default=10, help="Max results (default: 10)")

    # show
    show = sub.add_parser("show", help="Show pipeline structure (layers + nodes)")
    show.add_argument("workflow", help="Workflow name to display")

    args = parser.parse_args()
    engine = WorkflowEngine()

    if args.command == "start":
        context = None
        if args.context:
            context = {}
            for pair in args.context:
                k, v = pair.split("=", 1)
                context[k] = v
        inputs = None
        if args.inputs:
            inputs = {}
            for pair in args.inputs:
                k, v = pair.split("=", 1)
                inputs[k] = v
        engine.execute(args.workflow, context=context, start_node=args.node,
                      dry_run=args.dry_run, resume=args.resume,
                      board=args.board, inputs=inputs,
                      run_id=args.run_id)

    elif args.command == "validate":
        result = engine.validate(args.workflow)
        if result["valid"]:
            print(f"✓ {args.workflow} — {result['nodes']} nodes, "
                  f"{result['layers']} layers, valid DAG")
        else:
            print(f"✗ {args.workflow} — INVALID")
        if result["issues"]:
            for issue in result["issues"]:
                print(f"  • {issue}")
        sys.exit(0 if result["valid"] else 1)

    elif args.command == "status":
        state = engine.status(args.workflow)
        print(json.dumps(state, indent=2))

    elif args.command == "list":
        for f in sorted(engine.workflows_dir.glob("*.yaml")):
            print(f"  {f.stem}")

    elif args.command == "jobs":
        db_path = Path.home() / ".hermes" / "workflows" / "executions.db"
        if not db_path.exists():
            print("No workflow executions found.")
            return
        query = """
            SELECT w.run_id, w.workflow_name, w.status,
                   w.started_at, w.finished_at,
                   w.current_layer, w.total_layers, w.error,
                   (SELECT COUNT(*) FROM workflow_node_cards WHERE run_id = w.run_id AND status = 'done') as nodes_done,
                   (SELECT COUNT(*) FROM workflow_node_cards WHERE run_id = w.run_id) as nodes_total
            FROM workflow_executions w
        """
        params = []
        if args.status:
            query += " WHERE w.status = ?"
            params.append(args.status)
        query += " ORDER BY w.started_at DESC LIMIT ?"
        params.append(args.limit)
        with sqlite3.connect(str(db_path)) as conn:
            rows = conn.execute(query, params).fetchall()
        if not rows:
            print("No workflow executions found.")
            return
        print(f"{'RUN ID':<30} {'WORKFLOW':<20} {'STATUS':<12} {'STARTED':<22} {'NODES':<10} {'ERROR'}")
        print("-" * 120)
        for row in rows:
            run_id, name, status, started, finished, layer, total, err, nd, nt = row
            nodes_str = f"{nd}/{nt}" if nt else "-"
            err_str = (err[:40] + "...") if err and len(err) > 40 else (err or "")
            print(f"{run_id:<30} {name:<20} {status:<12} {started or '':<22} {nodes_str:<10} {err_str}")

    elif args.command == "show":
        workflow = engine.load_workflow(args.workflow)
        layers = engine.topological_sort(workflow)
        print(f"Pipeline: {workflow.name}")
        print(f"Description: {workflow.description[:80]}...")
        print(f"Layers: {len(layers)} | Nodes: {len(workflow.nodes)}")
        print()
        for i, layer in enumerate(layers):
            print(f"Layer {i}:")
            for nid in layer:
                node = workflow.nodes[nid]
                deps = f" ← {', '.join(node.depends_on)}" if node.depends_on else ""
                agent_label = "synthetic" if node.synthetic else node.agent
                print(f"  [{agent_label}] {nid}{deps}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
