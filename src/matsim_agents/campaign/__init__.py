"""Public package for multi-formula discovery campaigns.

This layer coordinates *many* formulas at once (element-set-to-hull), on top
of the existing single-composition discovery machinery
(:mod:`matsim_agents.discovery`) and multi-LLM debate workflow
(:mod:`matsim_agents.workflows.debate`).
"""

from matsim_agents.campaign.acquisition import (
    AcquisitionSelectionRecord,
    AcquisitionWeights,
    CampaignAcquisitionPolicy,
    CampaignAcquisitionState,
    FormulaAcquisitionMetrics,
    FormulaAcquisitionScore,
    select_formula_batch,
    update_adaptive_lambda,
)
from matsim_agents.campaign.benchmark import (
    BenchmarkArm,
    BenchmarkMetric,
    BenchmarkObservation,
    BenchmarkProtocol,
    BenchmarkRunSpec,
    PairedComparison,
    compare_paired_metric,
    paired_metric_differences,
)
from matsim_agents.campaign.execution import (
    CampaignDFTRefinementConfig,
    CampaignFormulaExecutionConfig,
    CampaignRetrainingConfig,
    ReferenceStructureSpec,
    latest_promoted_model,
    make_formula_runner,
    run_formula_with_active_learning,
)
from matsim_agents.campaign.orchestrator import (
    CampaignReviewDecision,
    CampaignRunPolicy,
    CampaignRunResult,
    run_campaign,
    run_formula_discovery_stage,
)
from matsim_agents.campaign.registry import (
    CandidateEvaluationRecord,
    CandidateRegistry,
    CandidateSelectionPolicy,
    CandidateSelectionScore,
    StructureCandidateRecord,
    formula_acquisition_metrics,
    ingest_exploration_result,
    select_dft_refinement_candidates,
    soap_descriptor,
    soap_novelty,
    structure_content_hash,
)
from matsim_agents.campaign.review import (
    CampaignDebateReviewConfig,
    make_debate_review_runner,
    run_campaign_debate_review,
)
from matsim_agents.campaign.state import (
    CampaignReviewRecord,
    CampaignState,
    FormulaRunRecord,
    HullState,
)

__all__ = [
    "AcquisitionSelectionRecord",
    "AcquisitionWeights",
    "CampaignAcquisitionPolicy",
    "CampaignAcquisitionState",
    "CandidateEvaluationRecord",
    "CandidateRegistry",
    "CandidateSelectionPolicy",
    "CandidateSelectionScore",
    "BenchmarkArm",
    "BenchmarkMetric",
    "BenchmarkObservation",
    "BenchmarkProtocol",
    "BenchmarkRunSpec",
    "PairedComparison",
    "CampaignReviewDecision",
    "CampaignReviewRecord",
    "CampaignDebateReviewConfig",
    "CampaignDFTRefinementConfig",
    "CampaignFormulaExecutionConfig",
    "CampaignRetrainingConfig",
    "ReferenceStructureSpec",
    "CampaignRunPolicy",
    "CampaignRunResult",
    "CampaignState",
    "FormulaRunRecord",
    "HullState",
    "FormulaAcquisitionMetrics",
    "FormulaAcquisitionScore",
    "StructureCandidateRecord",
    "formula_acquisition_metrics",
    "compare_paired_metric",
    "ingest_exploration_result",
    "select_dft_refinement_candidates",
    "latest_promoted_model",
    "run_campaign",
    "make_debate_review_runner",
    "make_formula_runner",
    "paired_metric_differences",
    "run_campaign_debate_review",
    "run_formula_with_active_learning",
    "run_formula_discovery_stage",
    "select_formula_batch",
    "soap_descriptor",
    "soap_novelty",
    "structure_content_hash",
    "update_adaptive_lambda",
]
