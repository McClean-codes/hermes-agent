---
name: workflow-engine
description: "Use when running multi-agent DAG pipelines via workflow_start — fire-and-forget orchestration across agents."
version: 3.1.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [workflow, pipeline, dag, kanban, orchestration, multi-agent]
    related_skills: [kanban-notification-system, plan]
---

# Workflow Engine

## Overview

The workflow engine runs DAG-based pipelines — multi-step processes where nodes depend on each other and execute in parallel where possible. You call `workflow_start`, it creates kanban cards, and returns immediately. The kanban dispatcher picks up ready cards and spawns workers. When the workflow finishes, you get a notification with a summary of what was done.

Fire-and-forget. No monitoring loop. The engine does not block.

## When to Use

- Multi-step pipeline: research → spec → build → review → deliver
- Parallel work across multiple agents with dependency ordering
- Workflow that pauses for user input and resumes when unblocked
- Sealed-envelope testing (implementer and tester work blind)

**Don't use for:** single tool calls, simple sequential tasks one agent can handle, or real-time mid-workflow interaction.

## Quick Reference

| Action | Call |
|--------|------|
| List available pipelines | `workflow_list()` |
| Show pipeline structure | `workflow_show(workflow="name")` |
| Validate before running | `workflow_validate(workflow="name")` |
| Start a pipeline | `workflow_start(workflow="name", context={...})` |
| Check running status | `workflow_status(workflow="name")` |

## Starting a Workflow

```python
workflow_start(
    workflow="my-pipeline",
    context={"topic": "Should we adopt X?"},
    inputs={"detail_level": "deep"},
    board="my-board",  # optional board override
    attachments={"/path/to/design.png"},
)
```

Returns immediately with `{"status": "dispatched"}`.

The engine:
1. Injects your session info (platform, chat_id, thread_id)
2. Creates kanban cards for every node across all layers
3. Subscribes the final-layer card(s) for notification
4. Returns — the kanban dispatcher handles execution

## User-Feedback Nodes

When a node needs user input, the worker blocks the card. You get a notification:

> Kanban t_abc123 blocked: Needs user feedback — which option?

Then:
1. Surface the options to the user
2. Get their input
3. Call `kanban_unblock(card_id="t_abc123")`

The workflow resumes from where it paused.

## YAML Authoring

### Structure

```yaml
name: my-workflow
description: "Multi-step pipeline"

trigger_events:
  - workflow_dispatch

roles:
  researcher: agent-a
  reviewer: agent-b

nodes:
  setup:
    agent: "{researcher}"
    task: >
      Analyze the topic: "{topic}".
    outputs:
      - findings

  review:
    agent: "{reviewer}"
    task: >
      Read {setup.findings}. Synthesize a recommendation.
    depends_on:
      - setup
    outputs:
      - recommendation
```

### Inputs

Declare expected inputs for progressive disclosure:

```yaml
inputs:
  - name: grill_artifact
    required: true
    description: "Path to the grill artifact file"
  - name: topic
    required: false
    description: "Research topic"
```

`workflow_show` returns the inputs list so agents discover what parameters are needed before triggering.

### Attachment Selection

Workflows declare expected attachments with names. Nodes reference them by name:

```yaml
attachments:
  - name: grill_artifact
    required: true
    description: "The enriched README"
  - name: source_video
    required: true
    description: "Base video for splicing"

nodes:
  enrich-artifact:
    attachment: grill_artifact
  splice-video:
    attachment: source_video
```

The caller passes named file paths:
```
workflow_start(
    workflow="my-workflow",
    attachments={"grill_artifact": "/path/to/file.md", "source_video": "/path/to/video.mp4"}
)
```

If `attachment` is set, only that named file is attached. If omitted, all attachments are attached (first layer) or none (other layers).

### Node Fields

| Field | Required | Description |
|-------|----------|-------------|
| `agent` | Yes | Which agent executes this node. Supports `{role}` template. |
| `task` | Yes | Instruction body. Supports `{upstream.output}` template variables. |
| `depends_on` | No | Node IDs that must complete before this one starts. |
| `outputs` | No | Named outputs — available as `{node-id.output-name}` downstream. |
| `timeout_minutes` | No | Max runtime per node. Default: 10. |
| `fallback_on_timeout` | No | `skip` \| `degraded` \| `fail` (default). |
| `goal_max_turns` | No | Max agent turns. Default: 20. |
| `when` | No | Conditional — node only runs when expression is true. |
| `reviews` | No | Sequential review pipeline (see below). |

### Template Variables

Resolve from:
1. Engine-injected: `{run_id}`, `{date}`
2. Inputs: `{inputs.topic}`, `{inputs.pr_link}`
3. Upstream outputs: `{setup.findings}`

### Roles

Map role names to agent profiles. Swap agents by editing one line:

```yaml
roles:
  architect: agent-a
  executor: agent-b
nodes:
  design:
    agent: "{architect}"
    task: "Analyze the proposal"
```

### DAG Patterns

- **Linear chain:** A → B → C (each depends on the previous)
- **Parallel layer:** Multiple nodes with the same `depends_on` run concurrently
- **Diamond:** Two nodes depend on the same prior node, then converge
- **Failure routing:** `fallback_on_timeout: skip` lets downstream proceed

## Review Pipeline

Nodes can declare a `reviews` attribute — a list of node IDs that review their output.

```yaml
nodes:
  implement:
    agent: "{coder}"
    task: >
      Implement the feature. Commit to your worktree branch.
      Do NOT open a PR. When done, block this card with
      reason "pending review".
    reviews:
      - verify
```

### How it works

1. Creator blocks card with `"pending review"` when done
2. Supervisor detects the block → dispatches the first reviewer
3. Reviewer passes → enriches creator with pass results → sets back to ready
4. Reviewer fails → enriches creator with failure feedback → sets back to ready
5. Creator picks up the enriched card and acts on the feedback

### Block reason convention

- `"pending review"` — triggers the review pipeline
- Any other block reason — treated as an unrelated failure (credential issue, push failure, etc.)

### Sequential reviews

Multiple reviewers run in order:

```yaml
reviews:
  - qa-review
  - security-review
```

When qa-review passes, security-review is dispatched. If any reviewer fails, the creator is enriched with feedback and reset to ready.

### Retry limits

Reviews have configurable retry limits with precedence (highest to lowest):

1. **Per-review**: `{review: "qa-review", max_retries: 5}`
2. **Node-level**: `max_retries: 3` on the node under review
3. **Workflow-level**: `max_retries: 5` in YAML
4. **Env var**: `HERMES_WORKFLOW_MAX_RETRIES`
5. **Engine default**: 3

```yaml
nodes:
  spec-author:
    max_retries: 5
    reviews:
      - qa-review
      - {review: security-review, max_retries: 2}
```

When the limit is hit, the reviewer is not dispatched and the node stays blocked.

## Dry-Run Mode

```python
workflow_start(workflow="name", dry_run=True)
```

Shows the execution plan without creating any cards.

## Resume from Saved State

```python
workflow_start(workflow="name", resume=True, node="specific-node")
```

Reuses saved state. Skips completed nodes.

## Common Pitfalls

| Symptom | Cause | Fix |
|---------|-------|-----|
| "workflow not found" error | Pipeline YAML missing or `HERMES_FLEET_PIPELINES` not set | Check `workflow_list()` shows it |
| No notification arrives | Session info not injected (CLI/cron path) | Call from a gateway session |
| Node stuck "running" | Kanban dispatcher polls wrong board | Verify card is on expected board |
| Template substitution failure | Missing key in context dict | Ensure all `{placeholders}` are in context |
| Card auto-blocked "heartbeat stale" | Handled automatically in fire-and-forget mode | No action needed |
| Review not triggered | Block reason doesn't start with "pending review" | Use exact phrase "pending review" |

## Verification Checklist

- [ ] Pipeline exists: `workflow_list()` shows it
- [ ] Pipeline validates: `workflow_validate(workflow="name")` returns valid
- [ ] All required inputs provided
- [ ] `workflow_start` returns `{"status": "dispatched"}`
- [ ] Final node completion triggers notification in your session
