"""
AIDE ML - AI-Driven Exploration for Machine Learning

A dual-model agentic system that autonomously writes, evaluates, and improves
ML code using tree-search with human-in-the-loop plan review.

Architecture:
    - Planner (Worker 1): Reasoning model for high-quality planning
    - Coder (Worker 2): Fast model for code execution/iteration
    - Feedback (Critic): Model for evaluating results

Usage:
    # Basic experiment (auto-approve mode)
    import aide
    exp = aide.Experiment(
        data_dir="data/",
        goal="Predict house prices",
        eval="RMSE"
    )
    best = exp.run(steps=20)
    
    # Dual-model with custom configuration
    exp = aide.Experiment(
        data_dir="data/",
        goal="Predict house prices",
        eval="RMSE",
        planner_model="o1-preview",
        coder_model="claude-3-5-sonnet-20241022",
        human_review=False,  # Set True for human-in-the-loop
    )
    
    # Interactive experiment with plan review
    exp = aide.InteractiveExperiment(...)
    while not exp.is_complete:
        result = exp.step()
        if result.needs_review:
            approved_plan = my_review_ui(result.pending_plan)
            exp.continue_with_plan(approved_plan)
"""

from dataclasses import dataclass
from typing import Optional

from .agent import Agent, AgentState, StepResult
from .interpreter import Interpreter
from .journal import Journal, Node, PlanArtifact
from omegaconf import OmegaConf
from rich.status import Status
from .utils.config import (
    load_task_desc,
    prep_agent_workspace,
    save_run,
    _load_cfg,
    prep_cfg,
    PlannerConfig,
    CoderConfig,
    HumanReviewConfig,
)


__version__ = "0.3.0"  # Bumped for dual-model architecture


@dataclass
class Solution:
    """Result of an AIDE experiment."""
    code: str
    valid_metric: float
    plan: Optional[str] = None
    was_human_reviewed: bool = False


class Experiment:
    """
    Standard AIDE experiment runner.
    
    For basic usage where human review is disabled or auto-approved.
    For interactive human-in-the-loop experiments, use InteractiveExperiment.
    """
    
    def __init__(
        self, 
        data_dir: str, 
        goal: str, 
        eval: str | None = None,
        planner_model: str | None = None,
        coder_model: str | None = None,
        plan_review_mode: str = "none",
    ):
        """Initialize a new experiment run.

        Args:
            data_dir (str): Path to the directory containing the data files.
            goal (str): Description of the goal of the task.
            eval (str | None, optional): Optional description of the preferred evaluation metric.
            planner_model (str | None, optional): Model for planning (Worker 1).
            coder_model (str | None, optional): Model for coding (Worker 2).
            plan_review_mode (str, optional): Plan review mode:
                - "none": No review, execute immediately (default, fastest)
                - "critic": GPT-4o critic reviews plans automatically (end-to-end autonomous)
                - "human": Human reviews plans via CLI (most control)
        """
        _cfg = _load_cfg(use_cli_args=False)
        _cfg.data_dir = data_dir
        _cfg.goal = goal
        _cfg.eval = eval
        
        # Configure dual-model architecture
        if planner_model is not None:
            _cfg.agent.planner.model = planner_model
        if coder_model is not None:
            _cfg.agent.coder.model = coder_model
        
        # Configure plan review mode
        _cfg.agent.plan_review.mode = plan_review_mode
        
        self.cfg = prep_cfg(_cfg)
        self.task_desc = load_task_desc(self.cfg)

        with Status("Preparing agent workspace (copying and extracting files) ..."):
            prep_agent_workspace(self.cfg)

        self.journal = Journal()
        self.agent = Agent(
            task_desc=self.task_desc,
            cfg=self.cfg,
            journal=self.journal,
        )
        self.interpreter = Interpreter(
            self.cfg.workspace_dir,
            **OmegaConf.to_container(self.cfg.exec),  # type: ignore
        )

    def run(self, steps: int) -> Solution:
        """
        Run the experiment for a specified number of steps.
        
        Args:
            steps: Number of improvement iterations.
            
        Returns:
            Solution: Best solution found.
        """
        for _i in range(steps):
            result = self.agent.step(exec_callback=self.interpreter.run)
            
            # Handle plan review if needed (auto-approve in this class)
            if result.state == AgentState.AWAITING_PLAN_REVIEW:
                # Auto-approve the plan
                self.agent.continue_with_approved_plan(
                    result.pending_plan,
                    self.interpreter.run,
                )
            
            save_run(self.cfg, self.journal)
        
        self.interpreter.cleanup_session()

        best_node = self.journal.get_best_node(only_good=False)
        return Solution(
            code=best_node.code, 
            valid_metric=best_node.metric.value,
            plan=best_node.plan,
            was_human_reviewed=best_node.was_human_reviewed,
        )


class InteractiveExperiment:
    """
    Interactive AIDE experiment with human-in-the-loop plan review.
    
    Use this class when you want to review and modify plans before execution.
    The experiment runs step-by-step, pausing when a plan needs review.
    
    Example:
        exp = aide.InteractiveExperiment(
            data_dir="data/",
            goal="Predict house prices",
            eval="RMSE",
        )
        
        while not exp.is_complete:
            result = exp.step()
            
            if result.needs_review:
                # Present plan to user in your UI
                modified_plan = my_review_ui(result.pending_plan)
                exp.continue_with_plan(modified_plan, comments="Looks good!")
            else:
                # Step completed, check result
                print(f"Step {result.step}: metric = {result.metric}")
        
        best = exp.get_best_solution()
    """
    
    def __init__(
        self,
        data_dir: str,
        goal: str,
        eval: str | None = None,
        planner_model: str = "o1-preview",
        coder_model: str = "claude-3-5-sonnet-20241022",
        steps: int = 20,
    ):
        """Initialize an interactive experiment.

        Args:
            data_dir (str): Path to the data directory.
            goal (str): Task goal description.
            eval (str | None, optional): Evaluation metric.
            planner_model (str, optional): Model for planning.
            coder_model (str, optional): Model for coding.
            steps (int, optional): Maximum number of steps.
        """
        _cfg = _load_cfg(use_cli_args=False)
        _cfg.data_dir = data_dir
        _cfg.goal = goal
        _cfg.eval = eval
        _cfg.agent.steps = steps
        _cfg.agent.planner.model = planner_model
        _cfg.agent.coder.model = coder_model
        _cfg.agent.human_review.enabled = True
        _cfg.agent.human_review.auto_approve = False
        
        self.cfg = prep_cfg(_cfg)
        self.task_desc = load_task_desc(self.cfg)
        self.max_steps = steps
        self._current_step = 0

        with Status("Preparing agent workspace (copying and extracting files) ..."):
            prep_agent_workspace(self.cfg)

        self.journal = Journal()
        self.agent = Agent(
            task_desc=self.task_desc,
            cfg=self.cfg,
            journal=self.journal,
        )
        self.interpreter = Interpreter(
            self.cfg.workspace_dir,
            **OmegaConf.to_container(self.cfg.exec),  # type: ignore
        )
        
        self._last_result: StepResult | None = None
        self._completed = False

    @property
    def is_complete(self) -> bool:
        """Check if the experiment has completed all steps."""
        return self._completed or self._current_step >= self.max_steps

    @property
    def current_step(self) -> int:
        """Get the current step number."""
        return self._current_step

    @property
    def needs_review(self) -> bool:
        """Check if a plan is awaiting review."""
        return (
            self._last_result is not None and 
            self._last_result.state == AgentState.AWAITING_PLAN_REVIEW
        )

    def step(self) -> StepResult:
        """
        Execute one step of the experiment.
        
        If the step requires plan review, the result will have
        state=AWAITING_PLAN_REVIEW and pending_plan will contain the plan.
        Call continue_with_plan() to proceed after review.
        
        Returns:
            StepResult: Result of the step.
        """
        if self.is_complete:
            return StepResult(
                state=AgentState.COMPLETED,
                message="Experiment is complete",
            )
        
        result = self.agent.step(exec_callback=self.interpreter.run)
        self._last_result = result
        
        if result.state != AgentState.AWAITING_PLAN_REVIEW:
            # Step completed
            self._current_step = len(self.journal)
            save_run(self.cfg, self.journal)
            
            if self._current_step >= self.max_steps:
                self._completed = True
                self.interpreter.cleanup_session()
        
        return result

    def continue_with_plan(
        self, 
        approved_plan: str, 
        comments: str | None = None,
    ) -> StepResult:
        """
        Continue execution after approving a plan.
        
        Args:
            approved_plan: The approved (possibly modified) plan.
            comments: Optional reviewer comments.
            
        Returns:
            StepResult: Result of the execution.
        """
        if not self.needs_review:
            raise RuntimeError("No plan is awaiting review")
        
        result = self.agent.continue_with_approved_plan(
            approved_plan,
            self.interpreter.run,
            reviewer_comments=comments,
        )
        
        self._current_step = len(self.journal)
        self._last_result = result
        save_run(self.cfg, self.journal)
        
        if self._current_step >= self.max_steps:
            self._completed = True
            self.interpreter.cleanup_session()
        
        return result

    def skip_plan(self) -> None:
        """Skip the current pending plan and continue to the next step."""
        if not self.needs_review:
            raise RuntimeError("No plan is awaiting review")
        
        # Reset state without executing
        self._last_result = None

    def get_best_solution(self) -> Solution | None:
        """Get the best solution found so far."""
        best_node = self.journal.get_best_node(only_good=False)
        if best_node is None:
            return None
        
        return Solution(
            code=best_node.code,
            valid_metric=best_node.metric.value,
            plan=best_node.plan,
            was_human_reviewed=best_node.was_human_reviewed,
        )

    def cleanup(self) -> None:
        """Clean up resources."""
        self.interpreter.cleanup_session()
