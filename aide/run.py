"""
AIDE Run Loop with Human-in-the-Loop Support

This module provides the main execution loop for AIDE with support for
human plan review interrupt in the dual-model architecture.
"""

import atexit
import logging
import shutil
import time
from pathlib import Path

from . import backend

from .agent import Agent, AgentState, StepResult
from .interpreter import Interpreter
from .journal import Journal, Node
from .journal2report import journal2report
from omegaconf import OmegaConf
from rich.columns import Columns
from rich.console import Group, Console
from rich.live import Live
from rich.padding import Padding
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
)
from rich.text import Text
from rich.status import Status
from rich.tree import Tree
from rich.markdown import Markdown
from rich.prompt import Prompt, Confirm
from .utils.config import load_task_desc, prep_agent_workspace, save_run, load_cfg

logger = logging.getLogger("aide")
console = Console()


def journal_to_rich_tree(journal: Journal):
    """Convert journal to a Rich tree for visualization."""
    best_node = journal.get_best_node()

    def append_rec(node: Node, tree):
        if node.is_buggy:
            s = "[red]◍ bug"
        else:
            style = "bold " if node is best_node else ""

            if node is best_node:
                s = f"[{style}green]● {node.metric.value:.3f} (best)"
            else:
                s = f"[{style}green]● {node.metric.value:.3f}"
        
        # Add review indicator
        if node.was_human_reviewed:
            s += " [cyan]👤"

        subtree = tree.add(s)
        for child in node.children:
            append_rec(child, subtree)

    tree = Tree("[bold blue]Solution tree")
    for n in journal.draft_nodes:
        append_rec(n, tree)
    return tree


def save_plan_to_file(cfg, plan: str, plan_id: str) -> Path:
    """Save a pending plan to a file for human review."""
    cfg.log_dir.mkdir(parents=True, exist_ok=True)
    plan_file = cfg.log_dir / f"pending_plan_{plan_id}.md"
    with open(plan_file, 'w', encoding='utf-8') as f:
        f.write("# Implementation Plan\n\n")
        f.write("_Review this plan and make any necessary modifications._\n")
        f.write("_Save the file when done, then return to the terminal._\n\n")
        f.write("---\n\n")
        f.write(plan)
    return plan_file


def load_plan_from_file(plan_file: Path) -> str:
    """Load a plan from a file after human review."""
    with open(plan_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Remove the header we added
    if "---\n\n" in content:
        content = content.split("---\n\n", 1)[-1]
    
    return content


def display_plan_review_ui(cfg, plan: str, plan_id: str) -> tuple[str, str | None]:
    """
    Display the plan review UI and wait for human input.
    
    Returns:
        tuple: (approved_plan, reviewer_comments)
    """
    console.print()
    console.print(Panel(
        "[bold yellow]PLAN REVIEW REQUIRED[/bold yellow]",
        title="🔍 Human-in-the-Loop",
        border_style="yellow",
    ))
    console.print()
    
    # Save plan to file
    plan_file = save_plan_to_file(cfg, plan, plan_id)
    
    # Display the plan
    console.print(Panel(
        Markdown(plan[:3000] + ("..." if len(plan) > 3000 else "")),
        title="📋 Generated Plan",
        border_style="blue",
    ))
    console.print()
    
    console.print(f"[dim]Plan saved to:[/dim] [yellow]{plan_file}[/yellow]")
    console.print()
    console.print("[bold]Options:[/bold]")
    console.print("  [green]1[/green] - Approve plan as-is")
    console.print("  [cyan]2[/cyan] - Edit plan in file, then approve")
    console.print("  [yellow]3[/yellow] - Skip this draft")
    console.print("  [red]4[/red] - Abort experiment")
    console.print()
    
    choice = Prompt.ask(
        "Your choice",
        choices=["1", "2", "3", "4"],
        default="1"
    )
    
    if choice == "1":
        # Approve as-is
        return plan, None
    
    elif choice == "2":
        # Edit in file
        console.print()
        console.print(f"[cyan]Edit the plan in:[/cyan] {plan_file}")
        console.print("[dim]Press Enter when done editing...[/dim]")
        input()
        
        # Load modified plan
        approved_plan = load_plan_from_file(plan_file)
        
        # Check if modified
        was_modified = approved_plan.strip() != plan.strip()
        if was_modified:
            console.print("[green]✓ Plan modifications detected[/green]")
        else:
            console.print("[dim]No modifications detected[/dim]")
        
        # Ask for comments
        comments = None
        if Confirm.ask("Add reviewer comments?", default=False):
            comments = Prompt.ask("Comments")
        
        return approved_plan, comments
    
    elif choice == "3":
        # Skip
        return None, None
    
    else:
        # Abort
        console.print("[red]Experiment aborted by user.[/red]")
        raise KeyboardInterrupt("User aborted experiment")


def run():
    """
    Main AIDE execution loop with human-in-the-loop support.
    
    This function handles the complete workflow:
    1. Load configuration and prepare workspace
    2. Execute agent steps in a loop
    3. Handle human plan review interrupts
    4. Save results and generate reports
    """
    cfg = load_cfg()
    logger.info(f'Starting run "{cfg.exp_name}"')

    task_desc = load_task_desc(cfg)
    task_desc_str = backend.compile_prompt_to_md(task_desc)

    with Status("Preparing agent workspace (copying and extracting files) ..."):
        prep_agent_workspace(cfg)

    def cleanup():
        if global_step == 0:
            shutil.rmtree(cfg.workspace_dir)

    atexit.register(cleanup)

    journal = Journal()
    agent = Agent(
        task_desc=task_desc,
        cfg=cfg,
        journal=journal,
    )
    interpreter = Interpreter(
        cfg.workspace_dir,
        **OmegaConf.to_container(cfg.exec),  # type: ignore
    )

    global_step = len(journal)
    prog = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=20),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
    )
    status = Status("[green]Generating code...")
    prog.add_task("Progress:", total=cfg.agent.steps, completed=global_step)

    def exec_callback(*args, **kwargs):
        status.update("[magenta]Executing code...")
        res = interpreter.run(*args, **kwargs)
        status.update("[green]Generating code...")
        return res

    def generate_live():
        tree = journal_to_rich_tree(journal)
        prog.update(prog.task_ids[0], completed=global_step)

        # Show dual-model info
        planner_model = cfg.agent.planner.model if hasattr(cfg.agent, 'planner') else "N/A"
        coder_model = cfg.agent.coder.model if hasattr(cfg.agent, 'coder') else "N/A"
        
        file_paths = [
            f"Result visualization:\n[yellow]▶ {str((cfg.log_dir / 'tree_plot.html'))}",
            f"Agent workspace directory:\n[yellow]▶ {str(cfg.workspace_dir)}",
            f"Experiment log directory:\n[yellow]▶ {str(cfg.log_dir)}",
        ]
        
        model_info = Text()
        model_info.append("Planner: ", style="dim")
        model_info.append(planner_model, style="cyan")
        model_info.append(" | Coder: ", style="dim")
        model_info.append(coder_model, style="green")
        
        left = Group(
            Panel(Text(task_desc_str.strip()), title="Task description"),
            model_info,
            prog,
            status,
        )
        right = tree
        wide = Group(*file_paths)

        title = f'[b]AIDE is working on experiment: [bold green]"{cfg.exp_name}[/b]"'
        subtitle = "Press [b]Ctrl+C[/b] to stop the run"
        
        # Show plan review mode indicator
        review_mode = cfg.agent.get_review_mode()
        if review_mode == "human":
            subtitle += " | [cyan]Review: HUMAN[/cyan]"
        elif review_mode == "critic":
            subtitle += " | [yellow]Review: CRITIC (GPT-4o)[/yellow]"

        return Panel(
            Group(
                Padding(wide, (1, 1, 1, 1)),
                Columns(
                    [Padding(left, (1, 2, 1, 1)), Padding(right, (1, 1, 1, 2))],
                    equal=True,
                ),
            ),
            title=title,
            subtitle=subtitle,
        )

    with Live(
        generate_live(),
        refresh_per_second=16,
        screen=True,
    ) as live:
        while global_step < cfg.agent.steps:
            status.update("[green]Generating code...")
            result = agent.step(exec_callback=exec_callback)
            
            if result.state == AgentState.AWAITING_PLAN_REVIEW:
                # Human review interrupt
                live.stop()  # Stop live display temporarily
                
                approved_plan, comments = display_plan_review_ui(
                    cfg, 
                    result.pending_plan, 
                    result.plan_id
                )
                
                if approved_plan is None:
                    # User chose to skip this draft
                    console.print("[yellow]Skipping this draft...[/yellow]")
                    live.start()
                    continue
                
                # Continue with approved plan
                console.print("[green]Executing approved plan...[/green]")
                live.start()
                
                status.update("[cyan]Executing approved plan...")
                agent.continue_with_approved_plan(
                    approved_plan, 
                    exec_callback,
                    reviewer_comments=comments,
                )
            
            save_run(cfg, journal)
            global_step = len(journal)
            live.update(generate_live())
    
    interpreter.cleanup_session()

    # Generate review report if any human reviews were done
    if journal.human_reviewed_nodes:
        console.print()
        console.print(Panel(
            Markdown(journal.generate_review_report()),
            title="📊 Human Review Report",
            border_style="cyan",
        ))
        
        # Save review report
        review_report_path = cfg.log_dir / "review_report.md"
        with open(review_report_path, 'w', encoding='utf-8') as f:
            f.write(journal.generate_review_report())
        console.print(f"[dim]Review report saved to:[/dim] {review_report_path}")

    if cfg.generate_report:
        print("Generating final report from journal...")
        report = journal2report(journal, task_desc, cfg.report)
        print(report)
        report_file_path = cfg.log_dir / "report.md"
        with open(report_file_path, "w") as f:
            f.write(report)
        print("Report written to file:", report_file_path)


if __name__ == "__main__":
    run()
