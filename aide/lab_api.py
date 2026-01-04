"""
AIDE Lab API - Web API for "The Lab" Frontend Integration

This module provides a FastAPI-based web API for running AIDE experiments
with human-in-the-loop plan review through a web interface.

Features:
- Start and manage AIDE experiments
- Get real-time experiment status
- Review and approve/modify plans via API
- Retrieve results and metrics

Usage:
    uvicorn aide.lab_api:app --reload --port 8000

    Or run directly:
    python -m aide.lab_api
"""

import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import asdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logger = logging.getLogger("aide.lab_api")


# ============================================
# Pydantic Models for API
# ============================================

class ExperimentState(str, Enum):
    """State of an AIDE experiment."""
    PENDING = "pending"                      # Created but not started
    RUNNING = "running"                      # Actively running
    AWAITING_REVIEW = "awaiting_review"      # Waiting for human plan approval
    PAUSED = "paused"                        # Paused by user
    COMPLETED = "completed"                  # All steps finished
    FAILED = "failed"                        # Failed with error
    CANCELLED = "cancelled"                  # Cancelled by user


class CreateExperimentRequest(BaseModel):
    """Request to create a new AIDE experiment."""
    data_dir: str = Field(..., description="Path to the data directory")
    goal: str = Field(..., description="Task goal description")
    eval_metric: Optional[str] = Field(None, description="Evaluation metric")
    steps: int = Field(20, description="Number of improvement steps")
    planner_model: str = Field("o1-preview", description="Model for planning")
    coder_model: str = Field("claude-3-5-sonnet-20241022", description="Model for coding")
    plan_review_mode: str = Field("none", description="Plan review mode: 'none', 'human', or 'critic'")


class ExperimentStatus(BaseModel):
    """Current status of an AIDE experiment."""
    session_id: str
    state: ExperimentState
    current_step: int
    total_steps: int
    pending_plan: Optional[str] = None
    pending_plan_id: Optional[str] = None
    best_metric: Optional[float] = None
    best_metric_is_maximizing: Optional[bool] = None
    nodes_count: int = 0
    buggy_count: int = 0
    good_count: int = 0
    human_reviewed_count: int = 0
    last_updated: float
    error_message: Optional[str] = None


class PlanReviewRequest(BaseModel):
    """Request to approve or modify a pending plan."""
    approved_plan: str = Field(..., description="The approved (possibly modified) plan")
    reviewer_comments: Optional[str] = Field(None, description="Optional reviewer comments")


class NodeSummary(BaseModel):
    """Summary of a solution node."""
    id: str
    step: int
    stage: str  # "draft", "debug", "improve"
    is_buggy: bool
    metric_value: Optional[float] = None
    was_human_reviewed: bool = False
    plan_summary: Optional[str] = None
    exec_time: Optional[float] = None


class ExperimentResults(BaseModel):
    """Full results of an AIDE experiment."""
    session_id: str
    state: ExperimentState
    nodes: List[NodeSummary]
    best_solution_code: Optional[str] = None
    best_solution_plan: Optional[str] = None
    best_metric: Optional[float] = None
    review_report: Optional[str] = None


# ============================================
# In-Memory Session Storage
# ============================================

class ExperimentSession:
    """Tracks a running AIDE experiment."""
    
    def __init__(
        self,
        session_id: str,
        data_dir: str,
        goal: str,
        eval_metric: Optional[str],
        steps: int,
        planner_model: str,
        coder_model: str,
        plan_review_mode: str = "none",
    ):
        self.session_id = session_id
        self.data_dir = data_dir
        self.goal = goal
        self.eval_metric = eval_metric
        self.steps = steps
        self.planner_model = planner_model
        self.coder_model = coder_model
        self.plan_review_mode = plan_review_mode
        
        self.state = ExperimentState.PENDING
        self.current_step = 0
        self.pending_plan: Optional[str] = None
        self.pending_plan_id: Optional[str] = None
        self.plan_approved_event: Optional[asyncio.Event] = None
        self.approved_plan: Optional[str] = None
        self.reviewer_comments: Optional[str] = None
        self.last_updated = time.time()
        self.error_message: Optional[str] = None
        
        # AIDE components (initialized when experiment starts)
        self.cfg = None
        self.agent = None
        self.journal = None
        self.interpreter = None
        self.task = None  # asyncio Task
    
    def update(self):
        """Update the last_updated timestamp."""
        self.last_updated = time.time()


# Global session storage (use Redis in production)
sessions: Dict[str, ExperimentSession] = {}


# ============================================
# FastAPI App
# ============================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    logger.info("AIDE Lab API starting up...")
    yield
    logger.info("AIDE Lab API shutting down...")
    # Cleanup any running experiments
    for session_id, session in list(sessions.items()):
        if session.task and not session.task.done():
            session.task.cancel()
            try:
                await session.task
            except asyncio.CancelledError:
                pass


app = FastAPI(
    title="AIDE Lab API",
    description="Web API for running AIDE ML experiments with human-in-the-loop plan review",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================
# API Endpoints
# ============================================

@app.get("/")
async def root():
    """API root endpoint."""
    return {
        "name": "AIDE Lab API",
        "version": "1.0.0",
        "docs": "/docs",
        "status": "running",
    }


@app.post("/experiments", response_model=ExperimentStatus)
async def create_experiment(
    request: CreateExperimentRequest,
    background_tasks: BackgroundTasks,
) -> ExperimentStatus:
    """
    Create a new AIDE experiment.
    
    The experiment will start running in the background.
    Use the returned session_id to track status and interact with the experiment.
    """
    session_id = f"exp_{uuid.uuid4().hex[:12]}"
    
    session = ExperimentSession(
        session_id=session_id,
        data_dir=request.data_dir,
        goal=request.goal,
        eval_metric=request.eval_metric,
        steps=request.steps,
        planner_model=request.planner_model,
        coder_model=request.coder_model,
        plan_review_mode=request.plan_review_mode,
    )
    
    sessions[session_id] = session
    
    # Start experiment in background
    background_tasks.add_task(run_experiment_async, session_id)
    
    return ExperimentStatus(
        session_id=session_id,
        state=ExperimentState.PENDING,
        current_step=0,
        total_steps=request.steps,
        last_updated=session.last_updated,
    )


@app.get("/experiments/{session_id}", response_model=ExperimentStatus)
async def get_experiment_status(session_id: str) -> ExperimentStatus:
    """Get the current status of an experiment."""
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    # Get metrics from journal if available
    best_metric = None
    best_is_maximizing = None
    nodes_count = 0
    buggy_count = 0
    good_count = 0
    human_reviewed = 0
    
    if session.journal:
        nodes_count = len(session.journal.nodes)
        buggy_count = len(session.journal.buggy_nodes)
        good_count = len(session.journal.good_nodes)
        human_reviewed = len(session.journal.human_reviewed_nodes)
        
        best_node = session.journal.get_best_node()
        if best_node and best_node.metric:
            best_metric = best_node.metric.value
            best_is_maximizing = best_node.metric.maximize
    
    return ExperimentStatus(
        session_id=session_id,
        state=session.state,
        current_step=session.current_step,
        total_steps=session.steps,
        pending_plan=session.pending_plan,
        pending_plan_id=session.pending_plan_id,
        best_metric=best_metric,
        best_metric_is_maximizing=best_is_maximizing,
        nodes_count=nodes_count,
        buggy_count=buggy_count,
        good_count=good_count,
        human_reviewed_count=human_reviewed,
        last_updated=session.last_updated,
        error_message=session.error_message,
    )


@app.post("/experiments/{session_id}/approve-plan", response_model=ExperimentStatus)
async def approve_plan(
    session_id: str,
    request: PlanReviewRequest,
) -> ExperimentStatus:
    """
    Approve or modify a pending plan.
    
    This endpoint is called when a human has reviewed the plan and is ready
    to proceed with execution.
    """
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    if session.state != ExperimentState.AWAITING_REVIEW:
        raise HTTPException(
            status_code=400,
            detail=f"Experiment is not awaiting review (current state: {session.state})"
        )
    
    if not session.pending_plan:
        raise HTTPException(status_code=400, detail="No pending plan to approve")
    
    # Store approved plan and signal continuation
    session.approved_plan = request.approved_plan
    session.reviewer_comments = request.reviewer_comments
    
    if session.plan_approved_event:
        session.plan_approved_event.set()
    
    session.update()
    
    return await get_experiment_status(session_id)


@app.post("/experiments/{session_id}/skip-plan", response_model=ExperimentStatus)
async def skip_plan(session_id: str) -> ExperimentStatus:
    """
    Skip the current pending plan and generate a new one.
    """
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    if session.state != ExperimentState.AWAITING_REVIEW:
        raise HTTPException(
            status_code=400,
            detail=f"Experiment is not awaiting review (current state: {session.state})"
        )
    
    # Signal to skip
    session.approved_plan = None  # None signals skip
    session.reviewer_comments = None
    
    if session.plan_approved_event:
        session.plan_approved_event.set()
    
    session.update()
    
    return await get_experiment_status(session_id)


@app.post("/experiments/{session_id}/pause", response_model=ExperimentStatus)
async def pause_experiment(session_id: str) -> ExperimentStatus:
    """Pause a running experiment."""
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    if session.state not in [ExperimentState.RUNNING, ExperimentState.AWAITING_REVIEW]:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot pause experiment in state: {session.state}"
        )
    
    session.state = ExperimentState.PAUSED
    session.update()
    
    return await get_experiment_status(session_id)


@app.post("/experiments/{session_id}/resume", response_model=ExperimentStatus)
async def resume_experiment(
    session_id: str,
    background_tasks: BackgroundTasks,
) -> ExperimentStatus:
    """Resume a paused experiment."""
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    if session.state != ExperimentState.PAUSED:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot resume experiment in state: {session.state}"
        )
    
    session.state = ExperimentState.RUNNING
    session.update()
    
    # Resume experiment in background
    background_tasks.add_task(run_experiment_async, session_id)
    
    return await get_experiment_status(session_id)


@app.post("/experiments/{session_id}/cancel", response_model=ExperimentStatus)
async def cancel_experiment(session_id: str) -> ExperimentStatus:
    """Cancel a running experiment."""
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    if session.state in [ExperimentState.COMPLETED, ExperimentState.CANCELLED]:
        raise HTTPException(
            status_code=400,
            detail=f"Experiment is already {session.state}"
        )
    
    # Cancel the background task
    if session.task and not session.task.done():
        session.task.cancel()
    
    session.state = ExperimentState.CANCELLED
    session.update()
    
    # Cleanup interpreter if exists
    if session.interpreter:
        session.interpreter.cleanup_session()
    
    return await get_experiment_status(session_id)


@app.get("/experiments/{session_id}/results", response_model=ExperimentResults)
async def get_experiment_results(session_id: str) -> ExperimentResults:
    """Get full results of an experiment including all nodes and best solution."""
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    nodes = []
    best_code = None
    best_plan = None
    best_metric = None
    review_report = None
    
    if session.journal:
        # Build node summaries
        for node in session.journal.nodes:
            nodes.append(NodeSummary(
                id=node.id,
                step=node.step,
                stage=node.stage_name,
                is_buggy=node.is_buggy if node.is_buggy is not None else True,
                metric_value=node.metric.value if node.metric else None,
                was_human_reviewed=node.was_human_reviewed,
                plan_summary=node.plan[:200] + "..." if node.plan and len(node.plan) > 200 else node.plan,
                exec_time=node.exec_time,
            ))
        
        # Get best solution
        best_node = session.journal.get_best_node()
        if best_node:
            best_code = best_node.code
            best_plan = best_node.plan
            best_metric = best_node.metric.value if best_node.metric else None
        
        # Generate review report
        if session.journal.human_reviewed_nodes:
            review_report = session.journal.generate_review_report()
    
    return ExperimentResults(
        session_id=session_id,
        state=session.state,
        nodes=nodes,
        best_solution_code=best_code,
        best_solution_plan=best_plan,
        best_metric=best_metric,
        review_report=review_report,
    )


@app.get("/experiments/{session_id}/best-solution")
async def get_best_solution(session_id: str) -> Dict[str, Any]:
    """Get the best solution code and plan."""
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    if not session.journal:
        raise HTTPException(status_code=400, detail="Experiment has not started yet")
    
    best_node = session.journal.get_best_node()
    if not best_node:
        raise HTTPException(status_code=404, detail="No valid solution found")
    
    return {
        "code": best_node.code,
        "plan": best_node.plan,
        "metric": best_node.metric.value if best_node.metric else None,
        "is_maximizing": best_node.metric.maximize if best_node.metric else None,
        "step": best_node.step,
        "exec_time": best_node.exec_time,
        "was_human_reviewed": best_node.was_human_reviewed,
    }


@app.get("/experiments")
async def list_experiments() -> List[ExperimentStatus]:
    """List all experiments."""
    statuses = []
    for session_id in sessions:
        statuses.append(await get_experiment_status(session_id))
    return statuses


@app.delete("/experiments/{session_id}")
async def delete_experiment(session_id: str) -> Dict[str, str]:
    """Delete an experiment from memory."""
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    # Cancel if running
    if session.task and not session.task.done():
        session.task.cancel()
        try:
            await session.task
        except asyncio.CancelledError:
            pass
    
    # Cleanup
    if session.interpreter:
        session.interpreter.cleanup_session()
    
    del sessions[session_id]
    
    return {"message": f"Experiment {session_id} deleted"}


# ============================================
# Background Experiment Runner
# ============================================

async def run_experiment_async(session_id: str):
    """
    Run an AIDE experiment asynchronously.
    
    This function handles the experiment loop, including human review interrupts.
    """
    session = sessions.get(session_id)
    if not session:
        return
    
    try:
        # Import AIDE components
        from .agent import Agent, AgentState
        from .interpreter import Interpreter
        from .journal import Journal
        from .utils.config import load_task_desc, prep_agent_workspace, _load_cfg, prep_cfg
        from omegaconf import OmegaConf
        from rich.status import Status
        
        # Initialize if not already done
        if session.cfg is None:
            # Build config
            cfg = _load_cfg(use_cli_args=False)
            cfg.data_dir = session.data_dir
            cfg.goal = session.goal
            cfg.eval = session.eval_metric
            cfg.agent.steps = session.steps
            cfg.agent.planner.model = session.planner_model
            cfg.agent.coder.model = session.coder_model
            cfg.agent.plan_review.mode = session.plan_review_mode
            
            session.cfg = prep_cfg(cfg)
            
            # Prepare workspace
            prep_agent_workspace(session.cfg)
            
            # Initialize components
            task_desc = load_task_desc(session.cfg)
            session.journal = Journal()
            session.agent = Agent(
                task_desc=task_desc,
                cfg=session.cfg,
                journal=session.journal,
            )
            session.interpreter = Interpreter(
                session.cfg.workspace_dir,
                **OmegaConf.to_container(session.cfg.exec),
            )
        
        session.state = ExperimentState.RUNNING
        session.update()
        
        # Execution callback
        def exec_callback(code, reset=True):
            return session.interpreter.run(code, reset)
        
        # Main loop
        while session.current_step < session.steps:
            if session.state == ExperimentState.PAUSED:
                return  # Exit, will be resumed later
            
            if session.state == ExperimentState.CANCELLED:
                return
            
            # Execute step
            result = session.agent.step(exec_callback=exec_callback)
            
            if result.state == AgentState.AWAITING_PLAN_REVIEW:
                # Human review interrupt
                session.state = ExperimentState.AWAITING_REVIEW
                session.pending_plan = result.pending_plan
                session.pending_plan_id = result.plan_id
                session.plan_approved_event = asyncio.Event()
                session.update()
                
                # Wait for human approval
                await session.plan_approved_event.wait()
                
                if session.approved_plan is None:
                    # User chose to skip
                    session.pending_plan = None
                    session.pending_plan_id = None
                    session.state = ExperimentState.RUNNING
                    session.update()
                    continue
                
                # Continue with approved plan
                session.agent.continue_with_approved_plan(
                    session.approved_plan,
                    exec_callback,
                    reviewer_comments=session.reviewer_comments,
                )
                
                # Clear pending state
                session.approved_plan = None
                session.reviewer_comments = None
                session.pending_plan = None
                session.pending_plan_id = None
                session.state = ExperimentState.RUNNING
            
            session.current_step = len(session.journal)
            session.update()
            
            # Save progress
            from .utils.config import save_run
            save_run(session.cfg, session.journal)
        
        # Completed
        session.state = ExperimentState.COMPLETED
        session.interpreter.cleanup_session()
        session.update()
        
    except asyncio.CancelledError:
        session.state = ExperimentState.CANCELLED
        if session.interpreter:
            session.interpreter.cleanup_session()
        raise
    
    except Exception as e:
        logger.exception(f"Experiment {session_id} failed")
        session.state = ExperimentState.FAILED
        session.error_message = str(e)
        if session.interpreter:
            session.interpreter.cleanup_session()


# ============================================
# Main Entry Point
# ============================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
