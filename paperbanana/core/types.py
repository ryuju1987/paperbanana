"""Core data types for PaperBanana pipeline."""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator

# Supported aspect ratios for diagram/plot generation.
SUPPORTED_ASPECT_RATIOS = {
    "1:1",
    "2:3",
    "3:2",
    "3:4",
    "4:3",
    "9:16",
    "16:9",
    "21:9",
}


class PipelineProgressStage(str, Enum):
    """Pipeline stage identifiers for progress callbacks."""

    OPTIMIZER_START = "optimizer_start"
    OPTIMIZER_END = "optimizer_end"
    RETRIEVER_START = "retriever_start"
    RETRIEVER_END = "retriever_end"
    PLANNER_START = "planner_start"
    PLANNER_END = "planner_end"
    STYLIST_START = "stylist_start"
    STYLIST_END = "stylist_end"
    VISUALIZER_START = "visualizer_start"
    VISUALIZER_END = "visualizer_end"
    CRITIC_START = "critic_start"
    CRITIC_END = "critic_end"


class PipelineProgressEvent(BaseModel):
    """Single progress event emitted by the pipeline for callbacks."""

    stage: PipelineProgressStage = Field(description="Pipeline stage identifier")
    message: str = Field(description="Human-readable message")
    seconds: Optional[float] = Field(default=None, description="Elapsed seconds for this step")
    iteration: Optional[int] = Field(default=None, description="Refinement iteration (1-based)")
    extra: Optional[dict[str, Any]] = Field(default=None, description="Optional extra data")


class DiagramType(str, Enum):
    """Type of academic illustration to generate."""

    METHODOLOGY = "methodology"
    STATISTICAL_PLOT = "statistical_plot"


class GenerationInput(BaseModel):
    """Input to the PaperBanana generation pipeline."""

    source_context: str = Field(description="Methodology section text or relevant paper excerpt")
    communicative_intent: str = Field(description="Figure caption describing what to communicate")
    diagram_type: DiagramType = Field(default=DiagramType.METHODOLOGY)
    raw_data: Optional[dict[str, Any]] = Field(
        default=None, description="Raw data for statistical plots (CSV path or dict)"
    )
    aspect_ratio: Optional[str] = Field(
        default=None,
        description=(
            "Target aspect ratio. "
            "Supported: 1:1, 2:3, 3:2, 3:4, 4:3, 9:16, 16:9, 21:9. "
            "If None, uses provider default."
        ),
    )

    @field_validator("aspect_ratio")
    @classmethod
    def validate_aspect_ratio(cls, v: Optional[str]) -> Optional[str]:
        """Ensure aspect_ratio, when provided, is one of the supported values."""
        if v is None:
            return v
        if v not in SUPPORTED_ASPECT_RATIOS:
            supported = ", ".join(sorted(SUPPORTED_ASPECT_RATIOS))
            raise ValueError(f"aspect_ratio must be one of: {supported}")
        return v


class ReferenceExample(BaseModel):
    """A single reference example from the curated set."""

    id: str
    source_context: str
    caption: str
    image_path: str
    category: Optional[str] = None
    aspect_ratio: Optional[float] = None
    structure_hints: Optional[dict[str, Any] | list[Any] | str] = None


class AxisScore(BaseModel):
    """Structured score for a single evaluation axis (Harness Design rubric)."""

    score: float = Field(ge=1, le=10, description="Score from 1-10")
    feedback: str = Field(default="", description="Specific feedback for this axis")


class CritiqueRubric(BaseModel):
    """Structured 4-axis evaluation rubric inspired by Anthropic's Harness Design.

    The four axes are adapted from the blog post's frontend design criteria:
    - Design quality: coherence across colors, typography, layout, and mood
    - Originality: custom decisions over templates and AI-generated patterns
    - Craft: typography hierarchy, spacing, color harmony, contrast
    - Functionality: information clarity and task completion
    """

    design_quality: Optional[AxisScore] = Field(
        default=None,
        description="Coherence of colors, typography, layout, and visual mood",
    )
    originality: Optional[AxisScore] = Field(
        default=None,
        description="Custom design decisions vs generic templates/AI patterns",
    )
    craft: Optional[AxisScore] = Field(
        default=None,
        description="Typography hierarchy, spacing, color harmony, contrast",
    )
    functionality: Optional[AxisScore] = Field(
        default=None,
        description="Information clarity and communicative effectiveness",
    )

    @property
    def average_score(self) -> float | None:
        """Average across all non-None axes."""
        scores = [
            axis.score
            for axis in [self.design_quality, self.originality, self.craft, self.functionality]
            if axis is not None
        ]
        return sum(scores) / len(scores) if scores else None


class CritiqueResult(BaseModel):
    """Output from the Critic agent."""

    critic_suggestions: list[str] = Field(default_factory=list)
    revised_description: Optional[str] = Field(
        default=None, description="Revised description if revision needed"
    )
    rubric: Optional[CritiqueRubric] = Field(
        default=None,
        description="Structured 4-axis evaluation scores (Harness Design)",
    )

    @property
    def needs_revision(self) -> bool:
        return len(self.critic_suggestions) > 0

    @property
    def summary(self) -> str:
        if not self.critic_suggestions:
            score_info = ""
            if self.rubric and self.rubric.average_score is not None:
                score_info = f" (score: {self.rubric.average_score:.1f}/10)"
            return f"No issues found. Image is publication-ready.{score_info}"
        return "; ".join(self.critic_suggestions[:3])


class IterationRecord(BaseModel):
    """Record of a single refinement iteration."""

    iteration: int
    description: str
    image_path: str
    critique: Optional[CritiqueResult] = None


class GenerationOutput(BaseModel):
    """Output from the PaperBanana generation pipeline."""

    image_path: str = Field(description="Path to the final generated image")
    description: str = Field(description="Final optimized description")
    iterations: list[IterationRecord] = Field(
        default_factory=list, description="History of refinement iterations"
    )
    metadata: dict[str, Any] = Field(default_factory=dict)


VALID_WINNERS = {"Model", "Human", "Both are good", "Both are bad"}

WINNER_SCORE_MAP: dict[str, float] = {
    "Model": 100.0,
    "Human": 0.0,
    "Both are good": 50.0,
    "Both are bad": 50.0,
}


class DimensionResult(BaseModel):
    """Result for a single comparative evaluation dimension."""

    winner: str = Field(description="Model | Human | Both are good | Both are bad")
    score: float = Field(
        ge=0.0,
        le=100.0,
        description="100 (Model wins), 0 (Human wins), 50 (Tie)",
    )
    reasoning: str = Field(default="", description="Comparison reasoning")


class EvaluationScore(BaseModel):
    """Comparative evaluation scores for a generated illustration.

    Uses the paper's referenced comparison approach where a VLM judge
    compares model-generated vs human-drawn diagrams on four dimensions,
    with hierarchical aggregation (Primary: Faithfulness + Readability,
    Secondary: Conciseness + Aesthetics).
    """

    faithfulness: DimensionResult
    conciseness: DimensionResult
    readability: DimensionResult
    aesthetics: DimensionResult
    overall_winner: str = Field(description="Hierarchical aggregation result")
    overall_score: float = Field(
        ge=0.0,
        le=100.0,
        description="100 (Model wins), 0 (Human wins), 50 (Tie)",
    )


class RunMetadata(BaseModel):
    """Metadata for a single pipeline run, for reproducibility."""

    run_id: str
    timestamp: str
    vlm_provider: str
    vlm_model: str
    image_provider: str
    image_model: str
    refinement_iterations: int
    seed: Optional[int] = None
    config_snapshot: dict[str, Any] = Field(default_factory=dict)
