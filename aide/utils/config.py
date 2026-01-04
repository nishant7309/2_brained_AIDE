"""configuration and setup utils"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Hashable, Optional, cast

import coolname
import rich
from omegaconf import OmegaConf, MISSING
from rich.syntax import Syntax
import shutup
from rich.logging import RichHandler
import logging

from . import tree_export
from . import copytree, preproc_data, serialize

shutup.mute_warnings()
logging.basicConfig(
    level="WARNING", format="%(message)s", datefmt="[%X]", handlers=[RichHandler()]
)
logger = logging.getLogger("aide")
logger.setLevel(logging.WARNING)


""" these dataclasses are just for type hinting, the actual config is in config.yaml """


@dataclass
class StageConfig:
    """Configuration for a single LLM stage (feedback, report, etc.)."""
    model: str
    temp: float


@dataclass
class PlannerConfig:
    """
    Configuration for Worker 1: The Planner (Reasoning Model).
    
    Used during the DRAFT phase to create detailed implementation plans.
    Should use a high-capability reasoning model like o1-preview or Gemini Pro.
    """
    model: str = "o1-preview"
    temp: float = 1.0
    thinking_level: str = "high"  # low, medium, high - controls reasoning depth


@dataclass
class CoderConfig:
    """
    Configuration for Worker 2: The Coder (Fast Model).
    
    Used for code generation from approved plans, and for DEBUG/IMPROVE iterations.
    Should use a fast, cost-effective model like Claude Sonnet or GPT-4o-mini.
    """
    model: str = "claude-3-5-sonnet-20241022"
    temp: float = 0.0  # Deterministic for consistent coding


@dataclass
class PlanReviewConfig:
    """
    Configuration for plan review before code execution.
    
    Modes:
        - "none": No review, execute plan immediately (fastest, legacy behavior)
        - "human": Human reviews plan via CLI or web UI (most control)
        - "critic": Feedback model (GPT-4o) reviews plan automatically (end-to-end autonomous)
    """
    mode: str = "none"  # "none", "human", or "critic"
    save_plans: bool = True  # Save all plans to log directory


@dataclass
class HumanReviewConfig:
    """
    Legacy configuration for human-in-the-loop plan review.
    Deprecated: Use PlanReviewConfig instead.
    """
    enabled: bool = False
    auto_approve: bool = True
    timeout: Optional[int] = None


@dataclass
class SearchConfig:
    """Configuration for tree search hyperparameters."""
    max_debug_depth: int
    debug_prob: float
    num_drafts: int


@dataclass
class AgentConfig:
    """
    Configuration for the AIDE agent.
    
    Supports dual-model architecture with separate Planner and Coder models,
    as well as backward compatibility with legacy single-model config.
    """
    steps: int
    k_fold_validation: int
    expose_prediction: bool
    data_preview: bool

    # Dual-model configuration (The Lab)
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    coder: CoderConfig = field(default_factory=CoderConfig)
    
    # Legacy single-model config (deprecated, for backward compatibility)
    code: StageConfig = field(default_factory=lambda: StageConfig(model="o4-mini", temp=0.5))
    
    # Feedback/Critic model
    feedback: StageConfig = field(default_factory=lambda: StageConfig(model="gpt-4.1-mini", temp=0.5))
    
    # Plan review configuration
    plan_review: PlanReviewConfig = field(default_factory=PlanReviewConfig)
    
    # Legacy human-in-the-loop review (deprecated, use plan_review)
    human_review: HumanReviewConfig = field(default_factory=HumanReviewConfig)

    search: SearchConfig = field(default_factory=lambda: SearchConfig(
        max_debug_depth=3, debug_prob=0.5, num_drafts=5
    ))
    
    def get_review_mode(self) -> str:
        """
        Get the plan review mode.
        
        Returns one of: "none", "human", "critic"
        
        Handles backward compatibility with legacy human_review config.
        """
        # Check new plan_review config first
        if hasattr(self, 'plan_review') and self.plan_review is not None:
            return self.plan_review.mode
        
        # Fall back to legacy human_review config
        if hasattr(self, 'human_review') and self.human_review is not None:
            if self.human_review.enabled and not self.human_review.auto_approve:
                return "human"
        
        return "none"
    
    def get_planner_model(self) -> str:
        """Get the planner model, with fallback to legacy code.model."""
        if hasattr(self, 'planner') and self.planner is not None:
            return self.planner.model
        return self.code.model
    
    def get_planner_temp(self) -> float:
        """Get the planner temperature, with fallback to legacy code.temp."""
        if hasattr(self, 'planner') and self.planner is not None:
            return self.planner.temp
        return self.code.temp
    
    def get_coder_model(self) -> str:
        """Get the coder model, with fallback to legacy code.model."""
        if hasattr(self, 'coder') and self.coder is not None:
            return self.coder.model
        return self.code.model
    
    def get_coder_temp(self) -> float:
        """Get the coder temperature, with fallback to legacy code.temp."""
        if hasattr(self, 'coder') and self.coder is not None:
            return self.coder.temp
        return self.code.temp


@dataclass
class ExecConfig:
    """Configuration for code execution."""
    timeout: int
    agent_file_name: str
    format_tb_ipython: bool


@dataclass
class Config(Hashable):
    """Main configuration object for AIDE experiments."""
    data_dir: Path
    desc_file: Path | None

    goal: str | None
    eval: str | None

    log_dir: Path
    workspace_dir: Path

    preprocess_data: bool
    copy_data: bool

    exp_name: str

    exec: ExecConfig
    generate_report: bool
    report: StageConfig
    agent: AgentConfig


def _get_next_logindex(dir: Path) -> int:
    """Get the next available index for a log directory."""
    max_index = -1
    for p in dir.iterdir():
        try:
            if current_index := int(p.name.split("-")[0]) > max_index:
                max_index = current_index
        except ValueError:
            pass
    return max_index + 1


def _migrate_legacy_config(cfg) -> None:
    """
    Migrate legacy single-model config to dual-model format.
    
    If only agent.code is specified (legacy format), copy its settings
    to both planner and coder for backward compatibility.
    """
    agent_cfg = cfg.get("agent", {})
    
    # Check if using legacy format (has code but no planner/coder)
    has_legacy_code = "code" in agent_cfg
    has_planner = "planner" in agent_cfg
    has_coder = "coder" in agent_cfg
    
    if has_legacy_code and not has_planner and not has_coder:
        logger.warning(
            "Using legacy 'agent.code' config. Consider migrating to "
            "'agent.planner' and 'agent.coder' for dual-model architecture."
        )
        # Legacy mode: use code.model for both planner and coder
        code_cfg = agent_cfg["code"]
        if "planner" not in agent_cfg:
            agent_cfg["planner"] = {
                "model": code_cfg.get("model", "o4-mini"),
                "temp": code_cfg.get("temp", 0.5),
                "thinking_level": "medium",
            }
        if "coder" not in agent_cfg:
            agent_cfg["coder"] = {
                "model": code_cfg.get("model", "o4-mini"),
                "temp": code_cfg.get("temp", 0.5),
            }


def _load_cfg(
    path: Path = Path(__file__).parent / "config.yaml", use_cli_args=True
) -> Config:
    cfg = OmegaConf.load(path)
    if use_cli_args:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_cli())
    
    # Migrate legacy config if needed
    _migrate_legacy_config(cfg)
    
    return cfg


def load_cfg(path: Path = Path(__file__).parent / "config.yaml") -> Config:
    """Load config from .yaml file and CLI args, and set up logging directory."""
    return prep_cfg(_load_cfg(path))


def prep_cfg(cfg: Config):
    if cfg.data_dir is None:
        raise ValueError("`data_dir` must be provided.")

    if cfg.desc_file is None and cfg.goal is None:
        raise ValueError(
            "You must provide either a description of the task goal (`goal=...`) or a path to a plaintext file containing the description (`desc_file=...`)."
        )

    if cfg.data_dir.startswith("example_tasks/"):
        cfg.data_dir = Path(__file__).parent.parent / cfg.data_dir
    cfg.data_dir = Path(cfg.data_dir).resolve()

    if cfg.desc_file is not None:
        cfg.desc_file = Path(cfg.desc_file).resolve()

    top_log_dir = Path(cfg.log_dir).resolve()
    top_log_dir.mkdir(parents=True, exist_ok=True)

    top_workspace_dir = Path(cfg.workspace_dir).resolve()
    top_workspace_dir.mkdir(parents=True, exist_ok=True)

    # generate experiment name and prefix with consecutive index
    ind = max(_get_next_logindex(top_log_dir), _get_next_logindex(top_workspace_dir))
    cfg.exp_name = cfg.exp_name or coolname.generate_slug(3)
    cfg.exp_name = f"{ind}-{cfg.exp_name}"

    cfg.log_dir = (top_log_dir / cfg.exp_name).resolve()
    cfg.workspace_dir = (top_workspace_dir / cfg.exp_name).resolve()

    # validate the config
    cfg_schema: Config = OmegaConf.structured(Config)
    cfg = OmegaConf.merge(cfg_schema, cfg)

    return cast(Config, cfg)


def print_cfg(cfg: Config) -> None:
    rich.print(Syntax(OmegaConf.to_yaml(cfg), "yaml", theme="paraiso-dark"))


def load_task_desc(cfg: Config):
    """Load task description from markdown file or config str."""

    # either load the task description from a file
    if cfg.desc_file is not None:
        if not (cfg.goal is None and cfg.eval is None):
            logger.warning(
                "Ignoring goal and eval args because task description file is provided."
            )

        with open(cfg.desc_file) as f:
            return f.read()

    # or generate it from the goal and eval args
    if cfg.goal is None:
        raise ValueError(
            "`goal` (and optionally `eval`) must be provided if a task description file is not provided."
        )

    task_desc = {"Task goal": cfg.goal}
    if cfg.eval is not None:
        task_desc["Task evaluation"] = cfg.eval

    return task_desc


def prep_agent_workspace(cfg: Config):
    """Setup the agent's workspace and preprocess data if necessary."""
    (cfg.workspace_dir / "input").mkdir(parents=True, exist_ok=True)
    (cfg.workspace_dir / "working").mkdir(parents=True, exist_ok=True)

    copytree(cfg.data_dir, cfg.workspace_dir / "input", use_symlinks=not cfg.copy_data)
    if cfg.preprocess_data:
        preproc_data(cfg.workspace_dir / "input")


def save_run(cfg: Config, journal):
    cfg.log_dir.mkdir(parents=True, exist_ok=True)

    # save journal
    serialize.dump_json(journal, cfg.log_dir / "journal.json")
    # save config
    OmegaConf.save(config=cfg, f=cfg.log_dir / "config.yaml")
    # create the tree + code visualization
    tree_export.generate(cfg, journal, cfg.log_dir / "tree_plot.html")
    # save the best found solution
    best_node = journal.get_best_node(only_good=False)
    with open(cfg.log_dir / "best_solution.py", "w") as f:
        f.write(best_node.code)
