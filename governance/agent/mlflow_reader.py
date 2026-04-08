"""
mlflow_reader.py
----------------
SageLLM-Forge | Governance Layer

WHAT THIS FILE DOES:
    Reads experiment results, model metrics, and registry status from MLflow.
    This is the single source of truth for the governance agent — it never
    talks to MLflow directly. Everything flows through here.

WHY A DEDICATED READER?
    If MLflow's API changes, or we swap it for a different tracking server,
    we only update this one file. Nothing else breaks. This is called
    the "adapter pattern" in software engineering.

WHAT IS MLFLOW?
    MLflow is an open-source platform that tracks ML experiments.
    Think of it like Git for model training runs:
      - Git commits  = MLflow runs
      - Git diff     = MLflow metric comparison
      - Git tags     = MLflow model registry stages (Staging, Production)

MLFLOW CONCEPTS USED HERE:
    - Experiment : a named group of runs (e.g. "churn-prediction")
    - Run        : one training execution with its own metrics + params
    - Artifact   : files saved during a run (model, plots, confusion matrix)
    - Registry   : a catalog of model versions with lifecycle stages
    - Stage      : None → Staging → Production → Archived
"""

import os
import logging
from dataclasses import dataclass, field
from typing import Optional

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities import Run
from mlflow.entities.model_registry import ModelVersion

# ── Logging setup ──────────────────────────────────────────────────────────────
# WHY: We use Python's built-in logger instead of print() because:
#   1. We can control log levels (DEBUG, INFO, WARNING, ERROR)
#   2. Logs can be shipped to CloudWatch without changing the code
#   3. Each log line automatically carries the module name for debugging
logger = logging.getLogger(__name__)


# ── Data classes ───────────────────────────────────────────────────────────────
# WHY DATACLASSES?
#   A dataclass is a clean way to define a data structure in Python.
#   Instead of passing raw dictionaries around (which have no structure
#   or type hints), we define exactly what fields we expect.
#   The governance agent and memo writer will receive these objects —
#   they always know what fields are available.

@dataclass
class RunMetrics:
    """
    Holds the evaluation metrics from a single MLflow training run.

    These are the numbers the governance agent uses to decide
    whether a model should be promoted.
    """
    run_id: str                          # Unique ID for this training run
    experiment_name: str                 # e.g. "churn-prediction-v2"
    model_name: str                      # Registered model name in MLflow
    model_version: str                   # e.g. "7"

    # Core evaluation metrics
    # WHY OPTIONAL? Not every model type tracks every metric.
    # A regression model won't have F1. We handle missing metrics gracefully.
    accuracy: Optional[float] = None
    f1_score: Optional[float] = None
    auc_roc: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    rmse: Optional[float] = None         # For regression models

    # Training health signals
    training_loss: Optional[float] = None
    validation_loss: Optional[float] = None
    training_duration_seconds: Optional[float] = None

    # Hyperparameters used in this run
    # WHY DICT? Parameters vary per model — we don't hardcode them.
    parameters: dict = field(default_factory=dict)

    # Artifact paths in S3 (stored by SageMaker, logged to MLflow)
    artifact_uri: Optional[str] = None   # e.g. s3://sagelllm-forge/artifacts/run_id/


@dataclass
class ComparisonResult:
    """
    Holds the side-by-side comparison between a candidate model
    and the current production model.

    The governance agent uses this to write the delta analysis
    section of the evaluation memo.
    """
    candidate: RunMetrics                # The new model we just trained
    production: Optional[RunMetrics]     # Current production model (None if first deploy)

    # Computed deltas — positive means candidate is better
    accuracy_delta: Optional[float] = None
    f1_delta: Optional[float] = None
    auc_delta: Optional[float] = None

    # Did the candidate beat the production model?
    # WHY BOOL? The Step Functions gate needs a clear pass/fail signal.
    candidate_is_better: bool = False

    # Human-readable summary of what changed
    # The governance agent uses this as context for the LLM prompt
    change_summary: str = ""


# ── MLflow Reader ──────────────────────────────────────────────────────────────

class MLflowReader:
    """
    Reads experiment results and model registry data from MLflow.

    USAGE:
        reader = MLflowReader(tracking_uri="http://your-mlflow-server:5000")
        metrics = reader.get_run_metrics(run_id="abc123", model_name="churn-model")
        comparison = reader.compare_with_production(candidate_metrics=metrics)

    DESIGN DECISION:
        We instantiate this once and reuse it. The MlflowClient is a lightweight
        HTTP client — creating it is cheap, but keeping one instance is cleaner.
    """

    def __init__(self, tracking_uri: Optional[str] = None):
        """
        Args:
            tracking_uri: URL of your MLflow tracking server.
                          Falls back to MLFLOW_TRACKING_URI env variable.
                          Example: "http://ec2-xx-xx-xx-xx.compute.amazonaws.com:5000"

        WHY ENV VARIABLE FALLBACK?
            We never hardcode server URLs in code. URLs change between
            dev/staging/production. Environment variables let us swap
            them without touching the code. This is called "12-factor app"
            design — a best practice for cloud applications.
        """
        self.tracking_uri = tracking_uri or os.environ.get(
            "MLFLOW_TRACKING_URI", "http://localhost:5000"
        )

        # Set the tracking URI globally for this process
        mlflow.set_tracking_uri(self.tracking_uri)

        # MlflowClient is the low-level API for reading runs and registry
        # mlflow.* functions are high-level helpers for logging during training
        # MlflowClient is for reading/querying — which is all we do here
        self.client = MlflowClient(tracking_uri=self.tracking_uri)

        logger.info(f"MLflowReader initialized | tracking_uri={self.tracking_uri}")

    # ── Public methods ─────────────────────────────────────────────────────────

    def get_run_metrics(self, run_id: str, model_name: str) -> RunMetrics:
        """
        Fetch all metrics, parameters, and metadata for a specific training run.

        Args:
            run_id    : The MLflow run ID (passed in by Step Functions after training)
            model_name: The registered model name in MLflow Model Registry

        Returns:
            RunMetrics dataclass with everything the governance agent needs

        WHAT IS A RUN ID?
            Every time SageMaker runs train.py, MLflow creates a run and assigns
            it a unique ID (like a UUID). Step Functions captures this ID and
            passes it downstream. We use it to look up exactly that run.
        """
        logger.info(f"Fetching run metrics | run_id={run_id} model={model_name}")

        # Fetch the raw run object from MLflow
        run: Run = self.client.get_run(run_id)

        # Get the model version linked to this run
        model_version = self._get_model_version_for_run(model_name, run_id)
        version_number = model_version.version if model_version else "unknown"

        # Extract metrics — MLflow stores them as {key: value} dicts
        # WHY .get()? If a metric wasn't logged, .get() returns None
        # instead of throwing a KeyError. Safer for optional metrics.
        metrics = run.data.metrics
        params  = run.data.params

        return RunMetrics(
            run_id=run_id,
            experiment_name=self._get_experiment_name(run.info.experiment_id),
            model_name=model_name,
            model_version=version_number,

            # Core metrics
            accuracy=metrics.get("accuracy"),
            f1_score=metrics.get("f1_score"),
            auc_roc=metrics.get("auc_roc"),
            precision=metrics.get("precision"),
            recall=metrics.get("recall"),
            rmse=metrics.get("rmse"),

            # Training health
            training_loss=metrics.get("train_loss"),
            validation_loss=metrics.get("val_loss"),
            training_duration_seconds=metrics.get("training_duration_seconds"),

            # Parameters (hyperparameters used in this run)
            parameters=dict(params),

            # Where the model artifact lives in S3
            artifact_uri=run.info.artifact_uri,
        )

    def compare_with_production(self, candidate: RunMetrics) -> ComparisonResult:
        """
        Compare the candidate model against the current production model.

        This is the core input to the governance memo:
        "New model improved F1 by 4% but accuracy dropped 1%"

        Args:
            candidate: Metrics from the newly trained model

        Returns:
            ComparisonResult with deltas and a plain-English change_summary
        """
        logger.info(
            f"Comparing candidate v{candidate.model_version} against production "
            f"| model={candidate.model_name}"
        )

        # Find the current production model version
        production_metrics = self._get_production_metrics(candidate.model_name)

        # If there's no production model yet, this is the first deployment
        if production_metrics is None:
            logger.info("No production model found — first deployment for this model")
            return ComparisonResult(
                candidate=candidate,
                production=None,
                candidate_is_better=True,
                change_summary="First version of this model. No baseline to compare against.",
            )

        # Compute deltas (positive = candidate is better for accuracy/F1/AUC)
        accuracy_delta = self._safe_delta(candidate.accuracy, production_metrics.accuracy)
        f1_delta       = self._safe_delta(candidate.f1_score, production_metrics.f1_score)
        auc_delta      = self._safe_delta(candidate.auc_roc,  production_metrics.auc_roc)

        # Determine if candidate is better overall
        # DECISION: We use F1 as the primary metric if available, else accuracy
        # WHY F1 OVER ACCURACY? Accuracy is misleading on imbalanced datasets.
        # F1 balances precision and recall — more reliable for real-world models.
        primary_delta = f1_delta if f1_delta is not None else accuracy_delta
        candidate_is_better = primary_delta is not None and primary_delta > 0

        # Build a human-readable change summary for the LLM prompt
        change_summary = self._build_change_summary(
            candidate, production_metrics, accuracy_delta, f1_delta, auc_delta
        )

        return ComparisonResult(
            candidate=candidate,
            production=production_metrics,
            accuracy_delta=accuracy_delta,
            f1_delta=f1_delta,
            auc_delta=auc_delta,
            candidate_is_better=candidate_is_better,
            change_summary=change_summary,
        )

    def get_recent_runs(self, experiment_name: str, n: int = 5) -> list[RunMetrics]:
        """
        Fetch the N most recent runs for an experiment.

        Used by the governance agent to provide historical context:
        "This model has been declining for 3 runs — the trend is concerning."

        Args:
            experiment_name: The MLflow experiment name
            n              : How many recent runs to fetch (default 5)
        """
        logger.info(f"Fetching {n} recent runs | experiment={experiment_name}")

        experiment = self.client.get_experiment_by_name(experiment_name)
        if experiment is None:
            logger.warning(f"Experiment not found: {experiment_name}")
            return []

        runs = self.client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"],
            max_results=n,
        )

        result = []
        for run in runs:
            try:
                metrics = RunMetrics(
                    run_id=run.info.run_id,
                    experiment_name=experiment_name,
                    model_name=experiment_name,
                    model_version=run.info.run_id[:8],
                    accuracy=run.data.metrics.get("accuracy"),
                    f1_score=run.data.metrics.get("f1_score"),
                    auc_roc=run.data.metrics.get("auc_roc"),
                    parameters=dict(run.data.params),
                    artifact_uri=run.info.artifact_uri,
                )
                result.append(metrics)
            except Exception as e:
                logger.warning(f"Skipping run {run.info.run_id}: {e}")

        return result

    # ── Private helpers ────────────────────────────────────────────────────────

    def _get_production_metrics(self, model_name: str) -> Optional[RunMetrics]:
        """Find the current Production-stage model and fetch its metrics."""
        try:
            production_versions: list[ModelVersion] = self.client.get_latest_versions(
                name=model_name,
                stages=["Production"],
            )
            if not production_versions:
                return None
            prod_version = production_versions[0]
            return self.get_run_metrics(
                run_id=prod_version.run_id,
                model_name=model_name,
            )
        except Exception as e:
            logger.warning(f"Could not fetch production model for {model_name}: {e}")
            return None

    def _get_model_version_for_run(
        self, model_name: str, run_id: str
    ) -> Optional[ModelVersion]:
        """Find the registry version that corresponds to a specific run."""
        try:
            versions = self.client.search_model_versions(f"name='{model_name}'")
            for v in versions:
                if v.run_id == run_id:
                    return v
            return None
        except Exception as e:
            logger.warning(f"Could not find model version for run {run_id}: {e}")
            return None

    def _get_experiment_name(self, experiment_id: str) -> str:
        """Convert an experiment ID to its human-readable name."""
        try:
            exp = self.client.get_experiment(experiment_id)
            return exp.name if exp else experiment_id
        except Exception:
            return experiment_id

    def _safe_delta(
        self, candidate_val: Optional[float], production_val: Optional[float]
    ) -> Optional[float]:
        """
        Compute (candidate - production) safely.
        Returns None if either value is missing — we never fabricate deltas.
        """
        if candidate_val is None or production_val is None:
            return None
        return round(candidate_val - production_val, 4)

    def _build_change_summary(
        self,
        candidate: RunMetrics,
        production: RunMetrics,
        accuracy_delta: Optional[float],
        f1_delta: Optional[float],
        auc_delta: Optional[float],
    ) -> str:
        """
        Build a plain-English summary of metric changes for the LLM prompt.

        WHY PLAIN ENGLISH?
            This string goes directly into the Bedrock prompt as context.
            The LLM works better with pre-digested summaries than raw numbers.
            We do the math here so the LLM focuses on reasoning, not arithmetic.

        Example output:
            "F1 improved by +4.2% (0.83 -> 0.87). Accuracy dropped -0.8%
             (0.91 -> 0.903). AUC unchanged. Training time increased by 12s."
        """
        lines = []

        if f1_delta is not None:
            direction = "improved" if f1_delta > 0 else "dropped"
            lines.append(
                f"F1 {direction} by {f1_delta:+.1%} "
                f"({production.f1_score:.3f} -> {candidate.f1_score:.3f})."
            )

        if accuracy_delta is not None:
            direction = "improved" if accuracy_delta > 0 else "dropped"
            lines.append(
                f"Accuracy {direction} by {accuracy_delta:+.1%} "
                f"({production.accuracy:.3f} -> {candidate.accuracy:.3f})."
            )

        if auc_delta is not None:
            if abs(auc_delta) < 0.001:
                lines.append("AUC unchanged.")
            else:
                direction = "improved" if auc_delta > 0 else "dropped"
                lines.append(f"AUC {direction} by {auc_delta:+.4f}.")

        param_changes = self._diff_params(candidate.parameters, production.parameters)
        if param_changes:
            lines.append(f"Parameter changes: {param_changes}.")

        return " ".join(lines) if lines else "No comparable metrics available."

    def _diff_params(self, candidate_params: dict, production_params: dict) -> str:
        """Find hyperparameters that changed between candidate and production."""
        changes = []
        all_keys = set(candidate_params) | set(production_params)
        for key in sorted(all_keys):
            c_val = candidate_params.get(key, "—")
            p_val = production_params.get(key, "—")
            if c_val != p_val:
                changes.append(f"{key}: {p_val} -> {c_val}")
        return ", ".join(changes) if changes else ""