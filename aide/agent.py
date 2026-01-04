"""
AIDE Agent with Dual-Model Architecture ("The Lab")

This module implements the core AIDE agent with a decoupled architecture:
- Worker 1 (Planner): Reasoning model for high-quality plan generation
- Worker 2 (Coder): Fast model for code execution and iteration
- Human-in-the-Loop: Optional plan review before execution
"""

import logging
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional, cast

import humanize
from .backend import FunctionSpec, query
from .interpreter import ExecutionResult
from .journal import Journal, Node, PlanArtifact
from .utils import data_preview
from .utils.config import Config
from .utils.metric import MetricValue, WorstMetricValue
from .utils.response import extract_code, extract_text_up_to_code, wrap_code

logger = logging.getLogger("aide")


ExecCallbackType = Callable[[str, bool], ExecutionResult]


class AgentState(Enum):
    """State of the agent in the execution loop."""
    READY = "ready"                          # Ready to execute next step
    AWAITING_PLAN_REVIEW = "awaiting_plan_review"  # Waiting for human plan approval
    EXECUTING = "executing"                  # Currently executing code
    COMPLETED = "completed"                  # All steps completed


@dataclass
class StepResult:
    """Result of a single agent step."""
    state: AgentState
    pending_plan: Optional[str] = None      # Plan awaiting review (if any)
    completed_node: Optional[Node] = None   # Completed node (if execution finished)
    plan_id: Optional[str] = None           # ID for tracking pending plan
    message: Optional[str] = None           # Status message


review_func_spec = FunctionSpec(
    name="submit_review",
    json_schema={
        "type": "object",
        "properties": {
            "is_bug": {
                "type": "boolean",
                "description": "true if the output log shows that the execution failed or has some bug, otherwise false.",
            },
            "summary": {
                "type": "string",
                "description": "if there is a bug, propose a fix. Otherwise, write a short summary (2-3 sentences) describing the empirical findings.",
            },
            "metric": {
                "type": "number",
                "description": "If the code ran successfully, report the value of the validation metric. Otherwise, leave it null.",
            },
            "lower_is_better": {
                "type": "boolean",
                "description": "true if the metric should be minimized (i.e. a lower metric value is better, such as with MSE), false if the metric should be maximized (i.e. a higher metric value is better, such as with accuracy).",
            },
        },
        "required": ["is_bug", "summary", "metric", "lower_is_better"],
    },
    description="Submit a review evaluating the output of the training script.",
)


# FunctionSpec for critic plan review (autonomous end-to-end mode)
plan_review_func_spec = FunctionSpec(
    name="review_plan",
    json_schema={
        "type": "object",
        "properties": {
            "approved": {
                "type": "boolean",
                "description": "true if the plan is good enough to proceed, false if it needs significant revision.",
            },
            "quality_score": {
                "type": "number",
                "description": "A score from 1-10 rating the plan quality. 1=poor, 5=adequate, 10=excellent.",
            },
            "feedback": {
                "type": "string",
                "description": "Constructive feedback on the plan. If approved, note any minor improvements. If not approved, explain what's missing or wrong.",
            },
            "improved_plan": {
                "type": "string",
                "description": "If the plan needs improvement, provide the corrected/enhanced version. If approved as-is, return the original plan unchanged.",
            },
        },
        "required": ["approved", "quality_score", "feedback", "improved_plan"],
    },
    description="Review an implementation plan and optionally improve it before code execution.",
)


class Agent:
    """
    AIDE Agent with Dual-Model Architecture.
    
    Implements a tree-search algorithm over the space of ML code solutions,
    with separate models for planning (reasoning) and coding (execution).
    
    Architecture:
        - Planner (Worker 1): High-capability reasoning model for DRAFT planning
        - Coder (Worker 2): Fast model for code execution and DEBUG/IMPROVE iterations
        - Feedback (Critic): Model for evaluating execution results
    
    The agent supports human-in-the-loop plan review, where users can
    review and modify plans before code execution.
    """
    
    def __init__(
        self,
        task_desc: str,
        cfg: Config,
        journal: Journal,
    ):
        super().__init__()
        self.task_desc = task_desc
        self.cfg = cfg
        self.acfg = cfg.agent
        self.journal = journal
        self.data_preview: str | None = None
        
        # State for human-in-the-loop
        self._pending_plan: str | None = None
        self._pending_plan_id: str | None = None
        self._state: AgentState = AgentState.READY
        
        # Store last critic review info
        self._last_critic_review: dict | None = None

    def search_policy(self) -> Node | None:
        """Select a node to work on (or None to draft a new node)."""
        search_cfg = self.acfg.search

        # initial drafting
        if len(self.journal.draft_nodes) < search_cfg.num_drafts:
            logger.debug("[search policy] drafting new node (not enough drafts)")
            return None

        # debugging
        if random.random() < search_cfg.debug_prob:
            # nodes that are buggy + leaf nodes + debug depth < max debug depth
            debuggable_nodes = [
                n
                for n in self.journal.buggy_nodes
                if (n.is_leaf and n.debug_depth <= search_cfg.max_debug_depth)
            ]
            if debuggable_nodes:
                logger.debug("[search policy] debugging")
                return random.choice(debuggable_nodes)
            logger.debug("[search policy] not debugging by chance")

        # back to drafting if no nodes to improve
        good_nodes = self.journal.good_nodes
        if not good_nodes:
            logger.debug("[search policy] drafting new node (no good nodes)")
            return None

        # greedy
        greedy_node = self.journal.get_best_node()
        logger.debug("[search policy] greedy node selected")
        return greedy_node

    @property
    def _prompt_environment(self):
        pkgs = [
            "numpy",
            "pandas",
            "scikit-learn",
            "statsmodels",
            "xgboost",
            "lightGBM",
            "torch",
            "torchvision",
            "torch-geometric",
            "bayesian-optimization",
            "timm",
        ]
        random.shuffle(pkgs)
        pkg_str = ", ".join([f"`{p}`" for p in pkgs])

        env_prompt = {
            "Installed Packages": f"Your solution can use any relevant machine learning packages such as: {pkg_str}. Feel free to use any other packages too (all packages are already installed!). For neural networks we suggest using PyTorch rather than TensorFlow."
        }
        return env_prompt

    @property
    def _prompt_impl_guideline(self):
        impl_guideline = [
            "The code should **implement the proposed solution** and **print the value of the evaluation metric computed on a hold-out validation set**.",
            "The code should be a single-file python program that is self-contained and can be executed as-is.",
            "No parts of the code should be skipped, don't terminate the before finishing the script.",
            "Your response should only contain a single code block.",
            f"Be aware of the running time of the code, it should complete within {humanize.naturaldelta(self.cfg.exec.timeout)}.",
            'All the provided input data is stored in "./input" directory.',
            '**If there is test data provided for this task, please save the test predictions in a `submission.csv` file in the "./working" directory as described in the task description** This is extremely important since this file is used for grading/evaluation. DO NOT FORGET THE submission.csv file!',
            'You can also use the "./working" directory to store any temporary files that your code needs to create.',
        ]
        if self.acfg.expose_prediction:
            impl_guideline.append(
                "The implementation should include a predict() function, "
                "allowing users to seamlessly reuse the code to make predictions on new data. "
                "The prediction function should be well-documented, especially the function signature."
            )

        if self.acfg.k_fold_validation > 1:
            impl_guideline.append(
                f"The evaluation should be based on {self.acfg.k_fold_validation}-fold cross-validation but only if that's an appropriate evaluation for the task at hand."
            )

        return {"Implementation guideline": impl_guideline}

    @property
    def _prompt_resp_fmt(self):
        return {
            "Response format": (
                "Your response should be a brief outline/sketch of your proposed solution in natural language (3-5 sentences), "
                "followed by a single markdown code block (wrapped in ```) which implements this solution and prints out the evaluation metric. "
                "There should be no additional headings or text in your response. Just natural language text followed by a newline and then the markdown code block. "
            )
        }

    @property
    def _prompt_plan_only_fmt(self):
        """Prompt format for plan-only generation (Planner model)."""
        return {
            "Response format": (
                "Your response should be a detailed implementation plan in markdown format. "
                "DO NOT include any code. Focus on:\n"
                "1. **Approach**: Overall strategy and model architecture\n"
                "2. **Data Processing**: How to load, clean, and preprocess the data\n"
                "3. **Model Selection**: Which algorithms/models to use and why\n"
                "4. **Training Strategy**: Training procedure, hyperparameters, validation approach\n"
                "5. **Evaluation**: How to compute and report the metric\n"
                "6. **Potential Issues**: Edge cases and how to handle them\n"
            )
        }

    @property
    def _prompt_code_from_plan_fmt(self):
        """Prompt format for code generation from approved plan (Coder model)."""
        return {
            "Response format": (
                "Implement the approved plan exactly as specified. "
                "Your response should contain ONLY a single Python code block (wrapped in ```). "
                "No explanations or additional text. Just the code."
            )
        }

    def _generate_plan(self) -> str:
        """
        Worker 1 (Planner): Generate a high-quality implementation plan.
        
        Uses the reasoning model (e.g., o1-preview) to create a detailed
        architectural plan without writing any code.
        
        Returns:
            str: The implementation plan in markdown format.
        """
        logger.info("Planner generating implementation plan...")
        
        prompt: Any = {
            "Introduction": (
                "You are a senior ML architect and Kaggle grandmaster. "
                "Your task is to design a winning solution for this competition. "
                "Create a detailed implementation plan that another engineer can follow. "
                "DO NOT write any code - focus only on architecture and strategy."
            ),
            "Task description": self.task_desc,
            "Memory": self.journal.generate_summary(),
            "Instructions": {},
        }
        
        prompt["Instructions"] |= self._prompt_plan_only_fmt
        prompt["Instructions"] |= {
            "Plan requirements": [
                "Describe the overall approach and model architecture in detail.",
                "Specify which libraries and algorithms to use with justification.",
                "Outline the complete data preprocessing pipeline.",
                "Define the training and evaluation strategy.",
                "Specify hyperparameters and their recommended values.",
                "Identify potential pitfalls and how to handle them.",
                "The plan should be detailed enough for another developer to implement.",
                "Take the Memory section into consideration - don't repeat failed approaches.",
                "Propose a solution that is different from previous attempts.",
            ],
        }
        prompt["Instructions"] |= self._prompt_environment

        if self.acfg.data_preview:
            prompt["Data Overview"] = self.data_preview

        # Use Planner model (reasoning model)
        planner_model = self.acfg.get_planner_model()
        planner_temp = self.acfg.get_planner_temp()
        
        logger.info(f"Using Planner model: {planner_model}")
        
        plan = query(
            system_message=prompt,
            user_message=None,
            model=planner_model,
            temperature=planner_temp,
        )
        
        return plan

    def _critic_review_plan(self, plan: str) -> tuple[str, dict]:
        """
        Critic (Feedback Model): Review and optionally improve a plan.
        
        Uses the feedback model (e.g., GPT-4o) to evaluate the plan,
        provide a quality score, and suggest improvements. This enables
        fully autonomous end-to-end operation without human intervention.
        
        Args:
            plan: The implementation plan to review.
            
        Returns:
            tuple: (approved_plan, review_info)
                - approved_plan: The plan (possibly improved) ready for execution
                - review_info: Dict with quality_score, feedback, and approval status
        """
        logger.info("Critic reviewing plan...")
        
        prompt: Any = {
            "Introduction": (
                "You are a senior ML reviewer and Kaggle expert. "
                "Your job is to review implementation plans before they are coded. "
                "Evaluate the plan's completeness, correctness, and likelihood of success. "
                "If the plan is good, approve it. If it has issues, improve it."
            ),
            "Task description": self.task_desc,
            "Plan to Review": plan,
            "Memory": self.journal.generate_summary(),
            "Instructions": {
                "Evaluation criteria": [
                    "Does the plan address all aspects of the task?",
                    "Is the approach technically sound and likely to work?",
                    "Are the library and model choices appropriate?",
                    "Is the evaluation strategy correct for this task?",
                    "Are there any obvious issues or missing steps?",
                    "Is the plan different from previously tried (and possibly failed) approaches?",
                ],
                "Response requirements": [
                    "If the plan is acceptable (score >= 6), approve it with minor suggestions.",
                    "If the plan has issues (score < 6), provide an improved version.",
                    "The improved_plan should be a complete, standalone plan.",
                    "Be constructive and specific in your feedback.",
                ],
            },
        }

        if self.acfg.data_preview:
            prompt["Data Overview"] = self.data_preview

        # Use Feedback model (the critic)
        response = cast(
            dict,
            query(
                system_message=prompt,
                user_message=None,
                func_spec=plan_review_func_spec,
                model=self.acfg.feedback.model,
                temperature=self.acfg.feedback.temp,
            ),
        )
        
        # Extract approved plan (improved if needed)
        approved_plan = response.get("improved_plan", plan)
        if not approved_plan or len(approved_plan.strip()) < 50:
            # Fallback to original if improved plan is empty/too short
            approved_plan = plan
        
        review_info = {
            "approved": response.get("approved", True),
            "quality_score": response.get("quality_score", 5),
            "feedback": response.get("feedback", ""),
            "was_improved": approved_plan.strip() != plan.strip(),
        }
        
        logger.info(f"Critic review: score={review_info['quality_score']}, approved={review_info['approved']}")
        
        return approved_plan, review_info

    def _execute_plan(self, approved_plan: str) -> Node:
        """
        Worker 2 (Coder): Generate code based on an approved plan.
        
        Uses the fast coding model (e.g., Claude Sonnet) to implement
        the plan that was created by the Planner and approved by the human.
        
        Args:
            approved_plan: The implementation plan approved by the human.
            
        Returns:
            Node: A new node containing the plan and generated code.
        """
        logger.info("Coder executing approved plan...")
        
        prompt: Any = {
            "Introduction": (
                "You are an expert ML engineer implementing a solution designed by the team architect. "
                "The implementation plan below has been reviewed and approved. "
                "Follow the plan precisely and implement a complete, working solution."
            ),
            "Approved Implementation Plan": approved_plan,
            "Task description": self.task_desc,
            "Instructions": {},
        }
        
        prompt["Instructions"] |= self._prompt_code_from_plan_fmt
        prompt["Instructions"] |= self._prompt_impl_guideline
        prompt["Instructions"] |= self._prompt_environment

        if self.acfg.data_preview:
            prompt["Data Overview"] = self.data_preview

        # Use Coder model (fast model)
        coder_model = self.acfg.get_coder_model()
        coder_temp = self.acfg.get_coder_temp()
        
        logger.info(f"Using Coder model: {coder_model}")
        
        code_response = query(
            system_message=prompt,
            user_message=None,
            model=coder_model,
            temperature=coder_temp,
        )
        
        code = extract_code(code_response) or code_response
        
        # Create plan artifact to track review status
        plan_artifact = PlanArtifact(
            original_plan=approved_plan,
            approved_plan=approved_plan,
            review_timestamp=time.time(),
            was_modified=False,
        )
        
        return Node(
            plan=approved_plan, 
            code=code,
            plan_artifact=plan_artifact,
            was_human_reviewed=self.acfg.human_review.enabled and not self.acfg.human_review.auto_approve,
        )

    def plan_and_code_query(
        self, 
        prompt, 
        model: str | None = None,
        temperature: float | None = None,
        retries: int = 3
    ) -> tuple[str, str]:
        """
        Generate a natural language plan + code in the same LLM call and split them apart.
        
        Used for IMPROVE and DEBUG operations where plan and code are generated together.
        
        Args:
            prompt: The prompt to send to the LLM.
            model: Optional model override (defaults to coder model).
            temperature: Optional temperature override.
            retries: Number of retries on failure.
            
        Returns:
            tuple: (plan_text, code) extracted from the response.
        """
        # Default to coder model for combined plan+code generation
        model = model or self.acfg.get_coder_model()
        temperature = temperature if temperature is not None else self.acfg.get_coder_temp()
        
        completion_text = None
        for _ in range(retries):
            completion_text = query(
                system_message=prompt,
                user_message=None,
                model=model,
                temperature=temperature,
            )

            code = extract_code(completion_text)
            nl_text = extract_text_up_to_code(completion_text)

            if code and nl_text:
                # merge all code blocks into a single string
                return nl_text, code

            print("Plan + code extraction failed, retrying...")
        print("Final plan + code extraction attempt failed, giving up...")
        return "", completion_text  # type: ignore

    def _draft(self) -> tuple[str, Node | None]:
        """
        DRAFT operator: Generate initial solution.
        
        In dual-model mode:
        1. Planner generates detailed implementation plan
        2. Based on plan_review.mode:
           - "none": Execute immediately
           - "human": Pause for human approval
           - "critic": Feedback model reviews and approves
        3. Coder executes the approved plan
        
        Returns:
            tuple: (plan_text, node_or_none)
                - If awaiting human review: (plan, None)
                - If auto-approved, critic-reviewed, or no review: (plan, executed_node)
        """
        # Step 1: Planner generates the implementation plan
        plan = self._generate_plan()
        
        # Step 2: Handle plan review based on mode
        review_mode = self.acfg.get_review_mode()
        logger.info(f"Plan review mode: {review_mode}")
        
        if review_mode == "human":
            # Return plan for human review, node will be created after approval
            logger.info("Plan generated, awaiting human review...")
            return plan, None
        
        elif review_mode == "critic":
            # Critic (feedback model) reviews the plan automatically
            logger.info("Plan generated, critic reviewing...")
            approved_plan, review_info = self._critic_review_plan(plan)
            
            # Store review info for later
            self._last_critic_review = review_info
            
            # Use the (possibly improved) plan
            logger.info(f"Critic approved plan (score: {review_info['quality_score']})")
            node = self._execute_plan(approved_plan)
            
            # Mark as critic-reviewed in the plan artifact
            if node.plan_artifact:
                node.plan_artifact.original_plan = plan
                node.plan_artifact.approved_plan = approved_plan
                node.plan_artifact.was_modified = review_info.get("was_improved", False)
                node.plan_artifact.reviewer_comments = review_info.get("feedback", "")
            
            return approved_plan, node
        
        else:  # mode == "none"
            # No review: execute immediately
            logger.info("No review, executing plan immediately...")
            node = self._execute_plan(plan)
            return plan, node

    def _draft_legacy(self) -> Node:
        """
        Legacy DRAFT operator (single-model mode).
        
        Used when dual-model architecture is disabled or for backward compatibility.
        Generates plan and code in a single LLM call.
        """
        prompt: Any = {
            "Introduction": (
                "You are a Kaggle grandmaster attending a competition. "
                "In order to win this competition, you need to come up with an excellent and creative plan "
                "for a solution and then implement this solution in Python. We will now provide a description of the task."
            ),
            "Task description": self.task_desc,
            "Memory": self.journal.generate_summary(),
            "Instructions": {},
        }
        prompt["Instructions"] |= self._prompt_resp_fmt
        prompt["Instructions"] |= {
            "Solution sketch guideline": [
                "This first solution design should be relatively simple, without ensembling or hyper-parameter optimization.",
                "Take the Memory section into consideration when proposing the design,"
                " don't propose the same modelling solution but keep the evaluation the same.",
                "The solution sketch should be 3-5 sentences.",
                "Propose an evaluation metric that is reasonable for this task.",
                "Don't suggest to do EDA.",
                "The data is already prepared and available in the `./input` directory. There is no need to unzip any files.",
            ],
        }
        prompt["Instructions"] |= self._prompt_impl_guideline
        prompt["Instructions"] |= self._prompt_environment

        if self.acfg.data_preview:
            prompt["Data Overview"] = self.data_preview

        plan, code = self.plan_and_code_query(prompt)
        return Node(plan=plan, code=code)

    def _improve(self, parent_node: Node) -> Node:
        """
        IMPROVE operator: Enhance a working solution.
        
        Uses the Coder model (fast model) for quick iterations.
        """
        prompt: Any = {
            "Introduction": (
                "You are a Kaggle grandmaster attending a competition. You are provided with a previously developed "
                "solution below and should improve it in order to further increase the (test time) performance. "
                "For this you should first outline a brief plan in natural language for how the solution can be improved and "
                "then implement this improvement in Python based on the provided previous solution. "
            ),
            "Task description": self.task_desc,
            "Memory": self.journal.generate_summary(),
            "Instructions": {},
        }
        prompt["Previous solution"] = {
            "Code": wrap_code(parent_node.code),
        }

        prompt["Instructions"] |= self._prompt_resp_fmt
        prompt["Instructions"] |= {
            "Solution improvement sketch guideline": [
                "The solution sketch should be a brief natural language description of how the previous solution can be improved.",
                "You should be very specific and should only propose a single actionable improvement.",
                "This improvement should be atomic so that we can experimentally evaluate the effect of the proposed change.",
                "Take the Memory section into consideration when proposing the improvement.",
                "The solution sketch should be 3-5 sentences.",
                "Don't suggest to do EDA.",
            ],
        }
        prompt["Instructions"] |= self._prompt_impl_guideline

        # Use Coder model for fast iterations
        plan, code = self.plan_and_code_query(prompt)
        return Node(
            plan=plan,
            code=code,
            parent=parent_node,
        )

    def _debug(self, parent_node: Node) -> Node:
        """
        DEBUG operator: Fix a buggy solution.
        
        Uses the Coder model (fast model) for quick bug fixes.
        """
        prompt: Any = {
            "Introduction": (
                "You are a Kaggle grandmaster attending a competition. "
                "Your previous solution had a bug, so based on the information below, you should revise it in order to fix this bug. "
                "Your response should be an implementation outline in natural language,"
                " followed by a single markdown code block which implements the bugfix/solution."
            ),
            "Task description": self.task_desc,
            "Previous (buggy) implementation": wrap_code(parent_node.code),
            "Execution output": wrap_code(parent_node.term_out, lang=""),
            "Instructions": {},
        }
        prompt["Instructions"] |= self._prompt_resp_fmt
        prompt["Instructions"] |= {
            "Bugfix improvement sketch guideline": [
                "You should write a brief natural language description (3-5 sentences) of how the issue in the previous implementation can be fixed.",
                "Don't suggest to do EDA.",
            ],
        }
        prompt["Instructions"] |= self._prompt_impl_guideline

        if self.acfg.data_preview:
            prompt["Data Overview"] = self.data_preview

        # Use Coder model for fast bug fixes
        plan, code = self.plan_and_code_query(prompt)
        return Node(plan=plan, code=code, parent=parent_node)

    def update_data_preview(self):
        """Update the data preview for the current workspace."""
        self.data_preview = data_preview.generate(self.cfg.workspace_dir)

    def step(self, exec_callback: ExecCallbackType) -> StepResult:
        """
        Execute one step of the agent loop.
        
        This method handles the dual-model workflow with human-in-the-loop:
        1. Checks if there's a pending plan awaiting review
        2. Otherwise, selects action via search policy
        3. For DRAFT: generates plan and may pause for review
        4. For IMPROVE/DEBUG: uses Coder model directly
        
        Args:
            exec_callback: Callback function to execute code.
            
        Returns:
            StepResult: Contains state info for UI integration.
        """
        if not self.journal.nodes or self.data_preview is None:
            self.update_data_preview()

        parent_node = self.search_policy()
        logger.debug(f"Agent is generating code, parent node type: {type(parent_node)}")

        if parent_node is None:
            # DRAFT phase - may interrupt for human review
            plan, result_node = self._draft()
            
            if result_node is None:
                # Awaiting human review
                self._pending_plan = plan
                self._pending_plan_id = f"plan_{len(self.journal)}_{int(time.time())}"
                self._state = AgentState.AWAITING_PLAN_REVIEW
                
                return StepResult(
                    state=AgentState.AWAITING_PLAN_REVIEW,
                    pending_plan=plan,
                    plan_id=self._pending_plan_id,
                    message="Plan generated, awaiting human review.",
                )
        elif parent_node.is_buggy:
            result_node = self._debug(parent_node)
        else:
            result_node = self._improve(parent_node)

        # Execute and evaluate
        self._state = AgentState.EXECUTING
        self.parse_exec_result(
            node=result_node,
            exec_result=exec_callback(result_node.code, True),
        )
        self.journal.append(result_node)
        self._state = AgentState.READY

        return StepResult(
            state=AgentState.READY,
            completed_node=result_node,
            message=f"Step completed. Metric: {result_node.metric}",
        )

    def continue_with_approved_plan(
        self, 
        approved_plan: str, 
        exec_callback: ExecCallbackType,
        reviewer_comments: str | None = None,
    ) -> StepResult:
        """
        Continue execution after human approves/modifies the plan.
        
        Called by the UI/CLI after user reviews the plan.
        
        Args:
            approved_plan: The approved (possibly modified) plan.
            exec_callback: Callback function to execute code.
            reviewer_comments: Optional comments from the reviewer.
            
        Returns:
            StepResult: Contains state info for UI integration.
        """
        logger.info("Continuing with approved plan...")
        
        # Check if plan was modified
        was_modified = (
            self._pending_plan is not None and 
            approved_plan.strip() != self._pending_plan.strip()
        )
        
        if was_modified:
            logger.info("Plan was modified by reviewer.")
        
        # Execute the approved plan
        result_node = self._execute_plan(approved_plan)
        
        # Update plan artifact with review info
        if result_node.plan_artifact is not None:
            result_node.plan_artifact.original_plan = self._pending_plan or approved_plan
            result_node.plan_artifact.approved_plan = approved_plan
            result_node.plan_artifact.was_modified = was_modified
            result_node.plan_artifact.reviewer_comments = reviewer_comments
        
        # Clear pending state
        self._pending_plan = None
        self._pending_plan_id = None
        
        # Execute and evaluate
        self._state = AgentState.EXECUTING
        self.parse_exec_result(
            node=result_node,
            exec_result=exec_callback(result_node.code, True),
        )
        self.journal.append(result_node)
        self._state = AgentState.READY
        
        return StepResult(
            state=AgentState.READY,
            completed_node=result_node,
            message=f"Plan executed. Metric: {result_node.metric}",
        )

    def get_pending_plan(self) -> tuple[str | None, str | None]:
        """
        Get the currently pending plan awaiting review.
        
        Returns:
            tuple: (plan_text, plan_id) or (None, None) if no pending plan.
        """
        return self._pending_plan, self._pending_plan_id

    def get_state(self) -> AgentState:
        """Get the current agent state."""
        return self._state

    def parse_exec_result(self, node: Node, exec_result: ExecutionResult):
        """Parse execution results and update node with evaluation."""
        logger.info(f"Agent is parsing execution results for node {node.id}")

        node.absorb_exec_result(exec_result)

        prompt = {
            "Introduction": (
                "You are a Kaggle grandmaster attending a competition. "
                "You have written code to solve this task and now need to evaluate the output of the code execution. "
                "You should determine if there were any bugs as well as report the empirical findings."
            ),
            "Task description": self.task_desc,
            "Implementation": wrap_code(node.code),
            "Execution output": wrap_code(node.term_out, lang=""),
        }

        response = cast(
            dict,
            query(
                system_message=prompt,
                user_message=None,
                func_spec=review_func_spec,
                model=self.acfg.feedback.model,
                temperature=self.acfg.feedback.temp,
            ),
        )

        # if the metric isn't a float then fill the metric with the worst metric
        if not isinstance(response["metric"], float):
            response["metric"] = None

        node.analysis = response["summary"]
        node.is_buggy = (
            response["is_bug"]
            or node.exc_type is not None
            or response["metric"] is None
        )

        if node.is_buggy:
            node.metric = WorstMetricValue()
        else:
            node.metric = MetricValue(
                response["metric"], maximize=not response["lower_is_better"]
            )
