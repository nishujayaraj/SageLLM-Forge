"""
governance_agent.py
-------------------
SageLLM-Forge | Governance Layer

WHAT THIS FILE DOES:
    The entry point for the entire governance layer.
    Step Functions calls this file after every training run.

    It coordinates three things in order:
      1. Fetch MLflow data         (mlflow_reader.py)
      2. Generate the memo         (memo_writer.py)
      3. Store the memo to S3      (memo_store.py)

    Then it returns a structured result back to Step Functions
    so the pipeline knows what to do next.

WHY IS THIS THE ENTRY POINT AND NOT memo_writer.py?
    Because Step Functions needs one clean interface to call.
    It passes in a run_id and model_name. It gets back a
    recommendation. Everything in between is hidden inside
    this coordinator. Step Functions doesn't care HOW the
    memo is generated — just what the verdict is.

WHAT IS THE HANDLER PATTERN?
    Lambda functions and Step Functions tasks follow a pattern:
        def handler(event, context):
            ...
    - event   : the input payload (JSON dict) passed by Step Functions
    - context : AWS runtime metadata (memory, timeout, request ID)
    This is the standard AWS entry point convention.
"""

import os
import logging
import json
from datetime import datetime, timezone
from typing import Optional

from mlflow_reader import MLflowReader
from memo_writer import MemoWriter
from memo_store import MemoStore

# ── Logging ────────────────────────────────────────────────────────────────────
# WHY BASICCONFIG HERE AND NOT IN THE OTHER FILES?
#   governance_agent.py is the entry point — it's the first file that runs.
#   We configure logging once here. All other files use getLogger(__name__)
#   and inherit this configuration automatically.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Constants ──────────────────────────────────────────────────────────────────
# These come from environment variables set in Step Functions task definition
# WHY ENV VARS AGAIN? The same reason as before — config lives outside code.
# When we deploy to staging vs production, only the env vars change.
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
AWS_REGION          = os.environ.get("AWS_REGION", "us-east-1")
MEMO_S3_BUCKET      = os.environ.get("MEMO_S3_BUCKET", "sagelllm-forge-memos")


class GovernanceAgent:
    """
    Coordinates the full governance workflow for a single training run.

    USAGE (called by Step Functions via Lambda handler below):
        agent = GovernanceAgent()
        result = agent.evaluate(run_id="abc123", model_name="churn-model")

    DESIGN:
        GovernanceAgent owns the workflow.
        It delegates all real work to the three specialist files.
        If any step fails, it catches the error, logs it, and returns
        a safe HOLD decision — never crashes the pipeline silently.
    """

    def __init__(self):
        """
        Initialize all three specialists.

        WHY INITIALIZE IN __init__ AND NOT IN evaluate()?
            Creating clients (boto3, MLflowClient) has a small overhead.
            If we initialize them in evaluate(), we pay that cost every call.
            Initializing once in __init__ means if the Lambda container
            is reused (which AWS does for warm starts), the clients are
            already ready.

        WHAT IS A LAMBDA WARM START?
            When Lambda runs your function, it spins up a container.
            If the same function is called again quickly, AWS reuses
            that container — this is called a "warm start".
            Anything initialized outside the handler() function persists
            across warm starts. This is a real AWS optimization technique.
        """
        logger.info("Initializing GovernanceAgent")

        self.reader = MLflowReader(tracking_uri=MLFLOW_TRACKING_URI)
        self.writer = MemoWriter(aws_region=AWS_REGION)
        self.store  = MemoStore(bucket_name=MEMO_S3_BUCKET, aws_region=AWS_REGION)

    def evaluate(self, run_id: str, model_name: str) -> dict:
        """
        Run the full governance evaluation for a training run.

        This is the core method. Three steps, in order:
          1. Read    → fetch metrics + comparison from MLflow
          2. Write   → generate memo via Bedrock
          3. Store   → save memo to S3

        Args:
            run_id     : MLflow run ID from the completed SageMaker job
            model_name : Registered model name in MLflow registry

        Returns:
            A dict that Step Functions reads to decide next action.
            Always returns something — never raises an unhandled exception.

        WHAT DOES STEP FUNCTIONS DO WITH THE RETURN VALUE?
            Step Functions can inspect the output of each task.
            We return a "recommendation" field. Step Functions has
            a Choice state that branches based on this value:
              "PROMOTE TO STAGING"     → canary deployment branch
              "PROMOTE TO PRODUCTION"  → blue-green deployment branch
              "HOLD"                   → notification branch
              "REJECT"                 → rollback branch
        """
        logger.info(
            f"Starting governance evaluation | "
            f"run_id={run_id} model={model_name}"
        )

        start_time = datetime.now(timezone.utc)

        try:
            # ── Step 1: Read ───────────────────────────────────────────────
            # Fetch candidate metrics and compare against production
            logger.info("Step 1/3 — Fetching MLflow data")

            candidate_metrics = self.reader.get_run_metrics(
                run_id=run_id,
                model_name=model_name,
            )

            comparison = self.reader.compare_with_production(
                candidate=candidate_metrics,
            )

            logger.info(
                f"MLflow data fetched | "
                f"candidate_version={candidate_metrics.model_version} "
                f"candidate_is_better={comparison.candidate_is_better}"
            )

            # ── Step 2: Write ──────────────────────────────────────────────
            # Generate the governance memo via Bedrock
            logger.info("Step 2/3 — Generating governance memo via Bedrock")

            memo = self.writer.generate_memo(comparison=comparison)

            logger.info(
                f"Memo generated | recommendation={memo.recommendation}"
            )

            # ── Step 3: Store ──────────────────────────────────────────────
            # Save the memo to S3 as the permanent audit trail
            logger.info("Step 3/3 — Storing memo to S3")

            s3_uri = self.store.save_memo(memo=memo)

            logger.info(f"Memo stored | s3_uri={s3_uri}")

            # ── Build result ───────────────────────────────────────────────
            duration_seconds = (
                datetime.now(timezone.utc) - start_time
            ).total_seconds()

            result = self._build_result(
                run_id=run_id,
                model_name=model_name,
                memo=memo,
                s3_uri=s3_uri,
                duration_seconds=duration_seconds,
                success=True,
            )

            logger.info(
                f"Governance evaluation complete | "
                f"recommendation={result['recommendation']} "
                f"duration={duration_seconds:.1f}s"
            )

            return result

        except Exception as e:
            # WHY CATCH ALL EXCEPTIONS HERE?
            #   If mlflow_reader fails, or Bedrock is down, or S3 is unavailable,
            #   we don't want the governance step to crash the entire pipeline
            #   with an unhandled exception. Instead we:
            #     1. Log the full error for debugging
            #     2. Return a HOLD decision — safe default
            #     3. Let Step Functions handle the failure gracefully
            logger.error(
                f"Governance evaluation failed | "
                f"run_id={run_id} error={str(e)}",
                exc_info=True,   # logs the full stack trace
            )

            return self._build_error_result(
                run_id=run_id,
                model_name=model_name,
                error=str(e),
            )

    # ── Private helpers ────────────────────────────────────────────────────────

    def _build_result(
        self,
        run_id: str,
        model_name: str,
        memo: "GovernanceMemo",
        s3_uri: str,
        duration_seconds: float,
        success: bool,
    ) -> dict:
        """
        Build the structured result dict that Step Functions reads.

        WHY A DICT AND NOT A DATACLASS?
            Step Functions receives JSON. A dict serializes to JSON natively.
            A dataclass would need extra conversion. Keep it simple at
            the boundary between your code and AWS services.

        WHAT STEP FUNCTIONS READS:
            - recommendation : the branching signal
            - success        : did the governance step succeed?
            - memo_s3_uri    : where the audit trail lives
            - run_id         : traceability back to the training run
        """
        return {
            "success": success,
            "run_id": run_id,
            "model_name": model_name,
            "model_version": memo.model_version,
            "recommendation": memo.recommendation,
            "candidate_is_better": memo.candidate_is_better,
            "memo_s3_uri": s3_uri,
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
            "duration_seconds": round(duration_seconds, 2),
        }

    def _build_error_result(
        self, run_id: str, model_name: str, error: str
    ) -> dict:
        """
        Build a safe HOLD result when evaluation fails.

        WHY HOLD ON ERROR?
            If we can't evaluate a model, we definitely should not
            promote it. HOLD is the safest possible default.
            The error is logged and included in the result so
            CloudWatch can alert the team.
        """
        return {
            "success": False,
            "run_id": run_id,
            "model_name": model_name,
            "model_version": "unknown",
            "recommendation": "HOLD",
            "candidate_is_better": False,
            "memo_s3_uri": None,
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
            "error": error,
        }


# ── Lambda handler ─────────────────────────────────────────────────────────────
# This is what Step Functions actually invokes.
# Step Functions calls handler(event, context) — this is the AWS convention.

# We create the agent OUTSIDE the handler so it's initialized once
# and reused across warm Lambda invocations (the optimization we discussed)
_agent = GovernanceAgent()


def handler(event: dict, context) -> dict:
    """
    AWS Lambda entry point — called by Step Functions.

    WHAT DOES event LOOK LIKE?
        Step Functions passes the pipeline state as a JSON dict.
        After the SageMaker training step, it will contain:
        {
            "run_id": "abc123def456",
            "model_name": "churn-prediction",
            "training_job_name": "sagelllm-forge-2024-01-15-09-30"
        }

    WHAT DOES context CONTAIN?
        AWS Lambda metadata: function name, memory limit, time remaining.
        We don't use it here but it must be in the signature.

    Args:
        event  : JSON payload from Step Functions
        context: Lambda runtime context (unused here)

    Returns:
        dict that Step Functions uses for the next state transition
    """
    logger.info(f"Lambda handler invoked | event={json.dumps(event)}")

    # Extract required fields from the Step Functions event
    # WHY .get() WITH NO DEFAULT?
    #   If run_id or model_name is missing, we want a clear KeyError,
    #   not a silent None that causes a confusing error deeper in the code.
    run_id     = event["run_id"]
    model_name = event["model_name"]

    return _agent.evaluate(run_id=run_id, model_name=model_name)