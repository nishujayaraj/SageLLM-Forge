"""
memo_writer.py
--------------
SageLLM-Forge | Governance Layer

WHAT THIS FILE DOES:
    Takes MLflow data (from mlflow_reader.py), loads the prompt template
    (evaluation_memo.txt), fills in the placeholders, calls AWS Bedrock,
    and returns the completed governance memo as a string.

    This is the only file in the entire project that talks to Bedrock.
    Everything LLM-related is contained here.

WHAT IS AWS BEDROCK?
    Bedrock is AWS's managed LLM service. Instead of hosting your own
    language model (which requires GPUs, infrastructure, and expertise),
    you make an API call to Bedrock and it runs the model for you.

    You pay per token (input + output). For our use case:
      ~2000 input tokens + ~800 output tokens per memo = ~$0.005 per memo

BEDROCK CONCEPTS USED HERE:
    - Model ID    : which LLM to use (we use Claude Haiku for cost efficiency)
    - Prompt      : the filled-in template we send to the model
    - Max tokens  : maximum length of the response (we cap at 1500)
    - Temperature : how creative vs deterministic the response is
                    0.0 = very deterministic (same output every time)
                    1.0 = very creative (different output every time)
                    We use 0.2 — mostly deterministic, slight variation allowed
                    WHY? Governance memos need consistency, not creativity.
"""

import os
import json
import logging
from datetime import date
from pathlib import Path
from typing import Optional

import boto3
from botocore.exceptions import ClientError

from mlflow_reader import RunMetrics, ComparisonResult

# ── Logging ────────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
# WHY CONSTANTS AT THE TOP?
#   If these values need to change (e.g. we upgrade to Claude Sonnet),
#   there is exactly one place to update. No hunting through code.

# The Bedrock model ID for Claude Haiku
# WHY HAIKU FOR DEV? It's the cheapest Claude model on Bedrock.
# Switch to "anthropic.claude-3-sonnet-20240229-v1:0" for production.
DEFAULT_MODEL_ID = "anthropic.claude-3-haiku-20240307-v1:0"

# Where our prompt template lives relative to this file
# Path(__file__) = the path of memo_writer.py itself
# .parent = the directory it's in (governance/agent/)
# Then we go into prompts/evaluation_memo.txt
PROMPT_TEMPLATE_PATH = Path(__file__).parent / "prompts" / "evaluation_memo.txt"

# Max tokens Bedrock will generate in the response
# 1500 tokens ≈ ~1100 words — enough for a detailed memo
MAX_TOKENS = 1500

# Temperature for the LLM
# Low value = consistent, professional, deterministic output
TEMPERATURE = 0.2


class MemoWriter:
    """
    Calls AWS Bedrock to generate a governance evaluation memo.

    USAGE:
        writer = MemoWriter(aws_region="us-east-1")
        memo = writer.generate_memo(comparison=comparison_result)
        print(memo.text)

    DESIGN DECISION:
        MemoWriter owns the entire Bedrock interaction — prompt loading,
        API call, response parsing, error handling. memo_writer.py is the
        only file that imports boto3 for Bedrock. Clean boundary.
    """

    def __init__(
        self,
        aws_region: Optional[str] = None,
        model_id: str = DEFAULT_MODEL_ID,
    ):
        """
        Args:
            aws_region: AWS region where Bedrock is available.
                        Falls back to AWS_REGION env variable, then us-east-1.
            model_id  : Which Bedrock model to use.

        WHAT IS boto3?
            boto3 is the official AWS SDK for Python.
            Every AWS service has a boto3 client — it handles authentication,
            request signing, retries, and error formatting automatically.
            You never write raw HTTP calls to AWS; boto3 does it for you.

        WHAT IS A BEDROCK CLIENT VS BEDROCK-RUNTIME CLIENT?
            - bedrock          : manages models (list, describe) — admin operations
            - bedrock-runtime  : actually invokes models — this is what we use
        """
        self.model_id = model_id
        self.aws_region = aws_region or os.environ.get("AWS_REGION", "us-east-1")

        # Create the Bedrock runtime client
        # WHY NOT HARDCODE CREDENTIALS?
        #   boto3 automatically picks up credentials from:
        #   1. Environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
        #   2. ~/.aws/credentials file
        #   3. IAM role (when running on EC2/Lambda — the right way in production)
        #   We never hardcode keys in code. Ever.
        self.bedrock = boto3.client(
            service_name="bedrock-runtime",
            region_name=self.aws_region,
        )

        # Load the prompt template once at init time
        # WHY AT INIT? Reading a file on every memo generation would be wasteful.
        # Load it once, reuse it for every call.
        self.prompt_template = self._load_prompt_template()

        logger.info(
            f"MemoWriter initialized | model={self.model_id} region={self.aws_region}"
        )

    # ── Public methods ─────────────────────────────────────────────────────────

    def generate_memo(self, comparison: ComparisonResult) -> "GovernanceMemo":
        """
        Generate a governance evaluation memo for a model training run.

        This is the main method. It does three things:
          1. Fills the prompt template with real MLflow data
          2. Calls Bedrock with the filled prompt
          3. Returns a GovernanceMemo object with the text + metadata

        Args:
            comparison: ComparisonResult from mlflow_reader.py
                        Contains both candidate and production metrics

        Returns:
            GovernanceMemo with the memo text and metadata
        """
        logger.info(
            f"Generating governance memo | "
            f"model={comparison.candidate.model_name} "
            f"version={comparison.candidate.model_version}"
        )

        # Step 1 — fill the prompt template with real data
        filled_prompt = self._fill_prompt_template(comparison)

        # Step 2 — call Bedrock
        memo_text = self._call_bedrock(filled_prompt)

        # Step 3 — wrap in a GovernanceMemo and return
        return GovernanceMemo(
            model_name=comparison.candidate.model_name,
            model_version=comparison.candidate.model_version,
            run_id=comparison.candidate.run_id,
            memo_text=memo_text,
            recommendation=self._extract_recommendation(memo_text),
            candidate_is_better=comparison.candidate_is_better,
        )

    # ── Private methods ────────────────────────────────────────────────────────

    def _load_prompt_template(self) -> str:
        """
        Load the evaluation_memo.txt template from disk.

        WHY PATHLIB INSTEAD OF OPEN()?
            pathlib.Path is the modern Python way to handle file paths.
            It works correctly on both Mac/Linux and Windows — no slash issues.
        """
        if not PROMPT_TEMPLATE_PATH.exists():
            raise FileNotFoundError(
                f"Prompt template not found at {PROMPT_TEMPLATE_PATH}. "
                f"Make sure evaluation_memo.txt exists in governance/agent/prompts/"
            )

        template = PROMPT_TEMPLATE_PATH.read_text(encoding="utf-8")
        logger.info(f"Prompt template loaded | path={PROMPT_TEMPLATE_PATH}")
        return template

    def _fill_prompt_template(self, comparison: ComparisonResult) -> str:
        """
        Fill the prompt template with real values from MLflow.

        This replaces every {placeholder} in evaluation_memo.txt
        with actual data from the ComparisonResult.

        WHAT DOES .format() DO?
            Python's str.format() replaces {key} with the value
            you pass as a keyword argument.

            Example:
                "Hello {name}".format(name="Alice") → "Hello Alice"

            We do the same thing but with ~20 placeholders.
        """
        candidate = comparison.candidate
        production = comparison.production

        # Format the parameters dict as readable key: value lines
        # WHY FORMAT SEPARATELY?
        #   {parameters} in the template expects a string, not a dict.
        #   We convert it here before passing to .format()
        params_formatted = "\n".join(
            f"  {k}: {v}" for k, v in candidate.parameters.items()
        ) or "  No parameters logged"

        # Format recent runs summary
        # This will be filled properly once we have get_recent_runs data
        # For now, we pass it as a placeholder — governance_agent.py will
        # fetch this and pass it in
        recent_runs_summary = comparison.change_summary or "No recent run history available."

        # Safely get production values
        # WHY "N/A"? If there's no production model, we show N/A
        # rather than None, which would look bad in the memo
        prod_version = production.model_version if production else "N/A (first deployment)"

        filled = self.prompt_template.format(
            # Model identity
            model_name=candidate.model_name,
            model_version=candidate.model_version,
            run_id=candidate.run_id,
            experiment_name=candidate.experiment_name,
            artifact_uri=candidate.artifact_uri or "N/A",

            # Candidate metrics
            accuracy=self._fmt(candidate.accuracy),
            f1_score=self._fmt(candidate.f1_score),
            auc_roc=self._fmt(candidate.auc_roc),
            precision=self._fmt(candidate.precision),
            recall=self._fmt(candidate.recall),
            training_loss=self._fmt(candidate.training_loss),
            validation_loss=self._fmt(candidate.validation_loss),
            training_duration_seconds=self._fmt(
                candidate.training_duration_seconds, decimals=0
            ),
            parameters=params_formatted,

            # Comparison
            production_version=prod_version,
            is_first_deploy=str(production is None),
            accuracy_delta=self._fmt(comparison.accuracy_delta, show_sign=True),
            f1_delta=self._fmt(comparison.f1_delta, show_sign=True),
            auc_delta=self._fmt(comparison.auc_delta, show_sign=True),
            change_summary=comparison.change_summary,

            # History + date
            recent_runs_summary=recent_runs_summary,
            today_date=date.today().isoformat(),
        )

        return filled

    def _call_bedrock(self, prompt: str) -> str:
        """
        Send the filled prompt to AWS Bedrock and return the response text.

        WHAT IS THE MESSAGES API?
            Claude on Bedrock uses the "messages" format — the same format
            you see in claude.ai. Each message has a role (user/assistant)
            and content (the text).

            We send one "user" message with our full prompt.
            Claude responds as "assistant".

        WHAT IS invoke_model()?
            This is the boto3 method that calls Bedrock.
            It takes:
              - modelId     : which model to use
              - body        : the request payload as JSON
            It returns:
              - body        : the response as a streaming byte object
                              we read() it and parse the JSON

        WHY TRY/EXCEPT?
            Network calls fail. Bedrock can return throttling errors,
            timeout errors, or model errors. We catch them specifically
            so the governance agent knows what went wrong and can retry
            or alert appropriately.
        """
        # Build the request body for Claude on Bedrock
        # This follows the Anthropic Messages API format
        request_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": MAX_TOKENS,
            "temperature": TEMPERATURE,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        }

        try:
            logger.info(f"Calling Bedrock | model={self.model_id}")

            response = self.bedrock.invoke_model(
                modelId=self.model_id,
                body=json.dumps(request_body),
                contentType="application/json",
                accept="application/json",
            )

            # Parse the response
            # response["body"] is a streaming object — we read() it first
            # then parse the JSON
            response_body = json.loads(response["body"].read())

            # Claude's response is in content[0].text
            # WHY [0]? Claude can return multiple content blocks.
            # For a simple text response, there's always exactly one.
            memo_text = response_body["content"][0]["text"]

            logger.info(
                f"Bedrock call successful | "
                f"input_tokens={response_body.get('usage', {}).get('input_tokens')} "
                f"output_tokens={response_body.get('usage', {}).get('output_tokens')}"
            )

            return memo_text

        except ClientError as e:
            # ClientError covers all AWS-specific errors:
            # throttling, model not found, permission denied, etc.
            error_code = e.response["Error"]["Code"]
            error_msg = e.response["Error"]["Message"]
            logger.error(f"Bedrock ClientError | code={error_code} msg={error_msg}")
            raise RuntimeError(f"Bedrock call failed [{error_code}]: {error_msg}") from e

        except Exception as e:
            logger.error(f"Unexpected error calling Bedrock: {e}")
            raise

    def _extract_recommendation(self, memo_text: str) -> str:
        """
        Extract the final recommendation from the memo text.

        The prompt instructs Claude to end with one of:
          PROMOTE TO STAGING | PROMOTE TO PRODUCTION | HOLD | REJECT

        We scan the memo for these keywords so governance_agent.py
        can make a programmatic decision (e.g. trigger canary deployment).

        WHY PARSE THE TEXT?
            We could ask Bedrock to return structured JSON instead of prose.
            But prose memos are more readable for humans. So we generate
            the memo as prose AND extract the structured signal from it.
            Best of both worlds.
        """
        keywords = [
            "PROMOTE TO PRODUCTION",
            "PROMOTE TO STAGING",
            "HOLD",
            "REJECT",
        ]

        memo_upper = memo_text.upper()
        for keyword in keywords:
            if keyword in memo_upper:
                return keyword

        # If no keyword found, default to HOLD
        # WHY HOLD AS DEFAULT? When uncertain, do nothing.
        # It is always safer to hold a model than to promote an uncertain one.
        logger.warning("No recommendation keyword found in memo — defaulting to HOLD")
        return "HOLD"

    def _fmt(
        self,
        value: Optional[float],
        decimals: int = 4,
        show_sign: bool = False,
    ) -> str:
        """
        Format a float for display in the prompt.

        WHY A HELPER?
            Many metrics can be None (not logged). We need consistent
            formatting everywhere — 4 decimal places, N/A for missing values.
            A helper avoids copy-pasting the same if/else 15 times.
        """
        if value is None:
            return "N/A"
        if show_sign:
            return f"{value:+.{decimals}f}"
        return f"{value:.{decimals}f}"


# ── GovernanceMemo ─────────────────────────────────────────────────────────────

from dataclasses import dataclass

@dataclass
class GovernanceMemo:
    """
    The output of the MemoWriter — a completed governance memo
    with its metadata.

    This object is passed to memo_store.py which saves it to S3.

    WHY A DATACLASS HERE AND NOT JUST A STRING?
        The memo text alone isn't enough. We need:
        - model_name + version for the S3 filename
        - run_id for traceability
        - recommendation for Step Functions to parse
        These travel together, so we bundle them in one object.
    """
    model_name: str
    model_version: str
    run_id: str
    memo_text: str
    recommendation: str     # PROMOTE TO STAGING | PROMOTE TO PRODUCTION | HOLD | REJECT
    candidate_is_better: bool