"""
research_benchmark.registry
===========================
Definitive 12-Model Experimental Registry for:
"Verifiable Multi-Agent Fault-Tolerant Adjudication System with Cryptographic Causal Provenance"

Four Research Groups (3 Models Each = 12 Models) + External Reference Oracle:
- Group 1: Semantic Reasoning and NLI
- Group 2: Computational Dependency Prediction
- Group 3: Evidence Verification and Fault Detection
- Group 4: Generative Specialist Agents
- Reference: Deterministic Rule Engine Oracle (Arm F)
"""

from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
import csv
import os

@dataclass
class ModelMetadata:
    model_number: int
    group_id: str
    group_name: str
    model_name: str
    checkpoint_name: str
    architecture_paper_title: str
    architecture_publication_year: int
    checkpoint_release_year: int
    parameter_count: str
    parameters_numeric: int
    original_purpose: str
    original_objective: str
    project_objective: str
    paper_url: str
    checkpoint_url: str
    dataset_source: str
    license: str
    our_modifications: str
    historical_results: str
    final_project_role: str
    laptop_feasibility: str
    nvidia_feasibility: str
    priority: str

REGISTRY: List[ModelMetadata] = [
    # -------------------------------------------------------------
    # GROUP 1: SEMANTIC REASONING AND NLI
    # -------------------------------------------------------------
    ModelMetadata(
        model_number=1,
        group_id="group_1",
        group_name="Semantic Reasoning and NLI",
        model_name="MiniLM2-L6-H768",
        checkpoint_name="cross-encoder/nli-MiniLM2-L6-H768",
        architecture_paper_title="MiniLMv2: Multi-Head Self-Attention Relation Distillation for Compressing Pretrained Transformers",
        architecture_publication_year=2020,
        checkpoint_release_year=2021,
        parameter_count="66M",
        parameters_numeric=66_000_000,
        original_purpose="Compact pre-trained language model distilled from larger transformers for fast cross-attention classification.",
        original_objective="Self-attention relation distillation on masked language modeling; fine-tuned for 3-class NLI on SNLI/MNLI.",
        project_objective="Lightweight semantic consistency check between claim assertions and policy clauses; fast baseline comparator.",
        paper_url="https://arxiv.org/abs/2012.15828",
        checkpoint_url="https://huggingface.co/cross-encoder/nli-MiniLM2-L6-H768",
        dataset_source="MultiNLI + ContractNLI + repo baseline_tests.json",
        license="Apache 2.0",
        our_modifications="Pretrained cross-encoder evaluated on domain pairs; 1-epoch fine-tuning on ContractNLI clauses.",
        historical_results="Baseline tests: 5/6 (83.3%) match; fails nli-003 (pizza/bicycle predicted contradiction at 99.57%).",
        final_project_role="Low-latency advisory NLI baseline for preliminary consistency screening.",
        laptop_feasibility="Fully Feasible (CPU float32, bs=8, ~1.5 min/epoch)",
        nvidia_feasibility="Trivial (<20s/epoch, fp16/bf16)",
        priority="High"
    ),
    ModelMetadata(
        model_number=2,
        group_id="group_1",
        group_name="Semantic Reasoning and NLI",
        model_name="DeBERTa-v3-small",
        checkpoint_name="cross-encoder/nli-deberta-v3-small",
        architecture_paper_title="DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing",
        architecture_publication_year=2021,
        checkpoint_release_year=2021,
        parameter_count="141M",
        parameters_numeric=141_000_000,
        original_purpose="Disentangled attention encoder model for high-precision natural language understanding and NLI.",
        original_objective="Replaced token detection (ELECTRA style) with disentangled relative position embeddings; fine-tuned on MNLI/SNLI.",
        project_objective="Pinned primary baseline for semantic advice; cross-encoder NLI scoring; reference backbone for custom dependency classifier.",
        paper_url="https://arxiv.org/abs/2111.09543",
        checkpoint_url="https://huggingface.co/cross-encoder/nli-deberta-v3-small",
        dataset_source="MultiNLI + ContractNLI + repo baseline_tests.json",
        license="MIT",
        our_modifications="Pinned primary baseline in semantic_advice.py; evaluated across domain pairs and fine-tuned on ContractNLI.",
        historical_results="Baseline tests: 5/6 (83.3%) match; fails nli-003 (99.33% contradiction); serves as base architecture for Candidate V2.",
        final_project_role="Standard transformer-based NLI comparator against which custom models and ModernBERT are evaluated.",
        laptop_feasibility="Fully Feasible (CPU float32, bs=4, ~4.0 min/epoch)",
        nvidia_feasibility="Standard Full SFT (fp16/bf16, bs=32, ~1 min/epoch)",
        priority="High"
    ),
    ModelMetadata(
        model_number=3,
        group_id="group_1",
        group_name="Semantic Reasoning and NLI",
        model_name="ModernBERT-base",
        checkpoint_name="answerdotai/ModernBERT-base",
        architecture_paper_title="Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder for Fast, Memory-Efficient, and Long-Context Understanding",
        architecture_publication_year=2024,
        checkpoint_release_year=2024,
        parameter_count="149M",
        parameters_numeric=149_000_000,
        original_purpose="Modernized bidirectional transformer encoder supporting native 8,192 token sequences with FlashAttention-2 and RoPE.",
        original_objective="Masked language modeling with GeGLU, rotary embeddings, unpadding, and sequence length warmup up to 8k tokens.",
        project_objective="Long-context semantic reasoning and full-document insurance policy NLI without the 512-token truncation limit.",
        paper_url="https://arxiv.org/abs/2412.13663",
        checkpoint_url="https://huggingface.co/answerdotai/ModernBERT-base",
        dataset_source="ContractNLI + MultiNLI (Full context up to 8k tokens)",
        license="Apache 2.0",
        our_modifications="Official base checkpoint initialized with a 3-class sequence classification head; fine-tuned for long-context NLI.",
        historical_results="Remote Space client configured in MODEL.md (tasksource/ModernBERT-zero-shot-nli); unexecuted in local training.",
        final_project_role="Long-context document-level policy verification engine reading full contracts without truncation.",
        laptop_feasibility="Feasible on truncated context (seq_len=512, CPU float32, bs=2, ~6 min/epoch)",
        nvidia_feasibility="Ideal (Full 8k context training with FlashAttention-2 and bf16)",
        priority="High"
    ),

    # -------------------------------------------------------------
    # GROUP 2: COMPUTATIONAL DEPENDENCY PREDICTION
    # -------------------------------------------------------------
    ModelMetadata(
        model_number=4,
        group_id="group_2",
        group_name="Computational Dependency Prediction",
        model_name="TF-IDF + Logistic Regression",
        checkpoint_name="sklearn.linear_model.LogisticRegression",
        architecture_paper_title="Scikit-learn: Machine Learning in Python",
        architecture_publication_year=2011,
        checkpoint_release_year=2023,
        parameter_count="~50k sparse",
        parameters_numeric=50_000,
        original_purpose="Classical statistical text classification baseline based on n-gram frequency matrices and L2-regularized linear classification.",
        original_objective="Minimize L2-regularized logistic cross-entropy loss over high-dimensional n-gram bag-of-words representations.",
        project_objective="Classical non-neural baseline to measure whether deep cross-attention adds real value over lexical word overlap.",
        paper_url="https://jmlr.csail.mit.edu/papers/v12/pedregosa11a.html",
        checkpoint_url="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html",
        dataset_source="Candidate V2 & Candidate V3 Provenance Suites",
        license="BSD 3-Clause",
        our_modifications="Trained with unigram+bigram TF-IDF vectorizer and Platt scaling calibration on concatenated artifact text pairs.",
        historical_results="Val macro-F1: ~0.50 on unigram proxy; serves as classical lower bound.",
        final_project_role="Fast, transparent, zero-GPU baseline proving the necessity of neural cross-attention.",
        laptop_feasibility="Fully Feasible (CPU, <2 seconds total run time)",
        nvidia_feasibility="CPU Bound / Trivial (<1s)",
        priority="High"
    ),
    ModelMetadata(
        model_number=5,
        group_id="group_2",
        group_name="Computational Dependency Prediction",
        model_name="CUSTOM DeBERTa-v3-small Dependency Model",
        checkpoint_name="_private/runs/candidate_v2_20261005T160614_302922Z_906cc116/checkpoints/best",
        architecture_paper_title="DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training (Adapted for Artifact Dependency Classification)",
        architecture_publication_year=2021,
        checkpoint_release_year=2026,
        parameter_count="141M",
        parameters_numeric=141_000_000,
        original_purpose="DeBERTa-v3-small encoder backbone adapted with custom 2-layer MLP head to classify computational dependency edges.",
        original_objective="Task-specific binary cross-entropy loss over declared provenance pairs: independent (0) vs depends_on (1).",
        project_objective="Core custom research model: predicts non-authoritative candidate dependency edges between updated sources and assessments.",
        paper_url="https://arxiv.org/abs/2111.09543",
        checkpoint_url="Local Checkpoint: _private/runs/candidate_v2_20261005T160614_302922Z_906cc116/checkpoints/best",
        dataset_source="Candidate V2 (568 rows) & Candidate V3 Root-Explicit Contrast (512 rows)",
        license="Research Prototype (MIT base)",
        our_modifications="Domain-specific 2-layer MLP head, threshold calibration at tau=0.2, V3 root-explicit ancestral contrast rules.",
        historical_results="V2 Val macro-F1 0.873 (acc 87.5%), Final acc 91.7% (F1 0.916), Stress acc 82.8% (42 gains, 49 regressions). Failed on 16 nli_context pairs.",
        final_project_role="Core custom subject model; evaluated across backbone initialization, 1-epoch adaptation, V2 checkpoint, and V3 contrast variant.",
        laptop_feasibility="Fully Feasible (CPU float32, bs=2, lr=2e-5, ~4.5 min/epoch)",
        nvidia_feasibility="Full Multi-Epoch SFT across 10 epochs with multi-seed variance (~3 min)",
        priority="High"
    ),
    ModelMetadata(
        model_number=6,
        group_id="group_2",
        group_name="Computational Dependency Prediction",
        model_name="BGE-small-en-v1.5",
        checkpoint_name="BAAI/bge-small-en-v1.5",
        architecture_paper_title="C-Pack: Packaged Resources To Advance General Chinese and English Embedding",
        architecture_publication_year=2023,
        checkpoint_release_year=2023,
        parameter_count="33.5M",
        parameters_numeric=33_500_000,
        original_purpose="Dense bi-encoder for semantic vector retrieval, embedding similarity, and asymmetric search.",
        original_objective="Contrastive representation learning using multi-task pre-training and hard negative mining on text pairs.",
        project_objective="Dense bi-encoder comparator: test whether cosine similarity between pre-computed artifact vectors can substitute for cross-encoders.",
        paper_url="https://arxiv.org/abs/2309.07597",
        checkpoint_url="https://huggingface.co/BAAI/bge-small-en-v1.5",
        dataset_source="Candidate V2 & Candidate V3 Provenance Suites",
        license="MIT",
        our_modifications="Bi-encoder cosine similarity mapped to threshold; optional projection head fine-tuned for metric learning.",
        historical_results="Unexecuted in historical runs; serves as the critical bi-encoder vs cross-encoder architectural baseline.",
        final_project_role="Evaluates whether O(1) cached vector embeddings can accurately detect dependencies without cross-attention.",
        laptop_feasibility="Fully Feasible (CPU float32, bs=16, ~1.0 min/epoch)",
        nvidia_feasibility="Trivial (<30s/epoch)",
        priority="Medium"
    ),

    # -------------------------------------------------------------
    # GROUP 3: EVIDENCE VERIFICATION AND FAULT DETECTION
    # -------------------------------------------------------------
    ModelMetadata(
        model_number=7,
        group_id="group_3",
        group_name="Evidence Verification and Fault Detection",
        model_name="DistilBERT-base-uncased",
        checkpoint_name="distilbert/distilbert-base-uncased",
        architecture_paper_title="DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter",
        architecture_publication_year=2019,
        checkpoint_release_year=2019,
        parameter_count="66M",
        parameters_numeric=66_000_000,
        original_purpose="Knowledge-distilled lightweight transformer encoder retaining 97% of BERT's capabilities with 40% fewer parameters.",
        original_objective="Triple loss distillation (masked language modeling, distillation loss, and cosine embedding loss).",
        project_objective="Trainable compact transformer baseline for evidence verification (replaces deterministic rule oracle in 12-model slots).",
        paper_url="https://arxiv.org/abs/1910.01108",
        checkpoint_url="https://huggingface.co/distilbert/distilbert-base-uncased",
        dataset_source="FEVER + HaluEval compatible subset + synthetic corrupted outputs",
        license="Apache 2.0",
        our_modifications="Initialized with binary/3-class evidence verification head [SUPPORTED, UNSUPPORTED, CONTRADICTORY]; fine-tuned on evidence pairs.",
        historical_results="Newly introduced trainable baseline; measures whether heavier models (DeBERTa-base) justify additional computational overhead.",
        final_project_role="Compact evidence classification baseline establishing minimum parameter requirements for fact checking.",
        laptop_feasibility="Fully Feasible (CPU float32, bs=4, ~2.5 min/epoch)",
        nvidia_feasibility="Fast SFT (<40s/epoch, fp16/bf16)",
        priority="High"
    ),
    ModelMetadata(
        model_number=8,
        group_id="group_3",
        group_name="Evidence Verification and Fault Detection",
        model_name="DeBERTa-v3-base-FEVER",
        checkpoint_name="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
        architecture_paper_title="Less Annotating, More Classifying: Addressing the Data Scarcity Problem with Zero-Shot NLI",
        architecture_publication_year=2022,
        checkpoint_release_year=2023,
        parameter_count="435M",
        parameters_numeric=435_000_000,
        original_purpose="DeBERTa-v3-base cross-encoder explicitly pre-trained on FEVER fact-verification, MNLI, and Adversarial NLI (ANLI).",
        original_objective="Multi-dataset supervised classification for evidence entailment, contradiction, and insufficient evidence.",
        project_objective="High-capacity evidence verification engine checking whether specialist rationales are factually grounded in attached evidence.",
        paper_url="https://arxiv.org/abs/2204.10842",
        checkpoint_url="https://huggingface.co/MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
        dataset_source="FEVER + Claim-Invoice Evidence Verification Pairs",
        license="MIT",
        our_modifications="Fine-tuned and calibrated on domain claim-evidence pairs; evaluates detection of hallucinated dollar amounts and dates.",
        historical_results="Evaluated conceptually in evidence audit; primary fact-checking model for interception of unsupported specialist outputs.",
        final_project_role="Primary fact-checking specialist model intercepting ungrounded claims before consensus insertion.",
        laptop_feasibility="Feasible for 1-epoch evaluation (CPU float32, bs=2, ~9.5 min/epoch)",
        nvidia_feasibility="Standard SFT (fp16/bf16, bs=16, ~2 min/epoch)",
        priority="High"
    ),
    ModelMetadata(
        model_number=9,
        group_id="group_3",
        group_name="Evidence Verification and Fault Detection",
        model_name="Llama-Guard-3-1B",
        checkpoint_name="meta-llama/Llama-Guard-3-1B",
        architecture_paper_title="Llama Guard 3: Guardrail Modeling with Llama-3.2",
        architecture_publication_year=2024,
        checkpoint_release_year=2024,
        parameter_count="1.23B",
        parameters_numeric=1_230_000_000,
        original_purpose="Specialized safety and policy classifier adapted from Llama-3.2 1B to detect conversational safety violations and prompt injections.",
        original_objective="Instruction-tuned safety classification predicting binary safe/unsafe labels with specific MLCommons safety violation codes.",
        project_objective="Specialist agent output guardrail: classifies whether generated assessments contain policy violations or corrupt reasoning.",
        paper_url="https://arxiv.org/abs/2407.21783",
        checkpoint_url="https://huggingface.co/meta-llama/Llama-Guard-3-1B",
        dataset_source="HaluEval + Synthetic Corrupted Specialist Outputs + Policy Risk Cases",
        license="Llama 3.2 Community License",
        our_modifications="Explicitly mapped safety/fault taxonomy separating content safety from factual evidence checking; evaluated on output anomalies.",
        historical_results="Documented as safety comparator; task formulation strictly isolates safety classification from factual support verification.",
        final_project_role="Advisory output inspector detecting policy violations, malformed schema, and prompt injection attempts.",
        laptop_feasibility="Inference Smoke Test Only (CPU int4/float32, 10 cases, ~8 min; Lab Only for Training)",
        nvidia_feasibility="LoRA / QLoRA Fine-Tuning with bf16 on 8GB-16GB VRAM GPU (~4 min/epoch)",
        priority="Medium"
    ),

    # -------------------------------------------------------------
    # GROUP 4: GENERATIVE SPECIALIST AGENTS
    # -------------------------------------------------------------
    ModelMetadata(
        model_number=10,
        group_id="group_4",
        group_name="Generative Specialist Agents",
        model_name="Qwen2.5-0.5B-Instruct",
        checkpoint_name="Qwen/Qwen2.5-0.5B-Instruct",
        architecture_paper_title="Qwen2.5: A Foundation Language Model Series",
        architecture_publication_year=2024,
        checkpoint_release_year=2024,
        parameter_count="494M",
        parameters_numeric=494_000_000,
        original_purpose="Compact generative instruction model with RoPE, SwiGLU, RMSNorm, tied embeddings, and 32k context window.",
        original_objective="Autoregressive language modeling with large-scale pretraining followed by SFT and RLHF for instruction adherence.",
        project_objective="Compact specialist agent proposer: generates structured JSON decisions for compliance, risk, and document specialists.",
        paper_url="https://arxiv.org/abs/2412.15115",
        checkpoint_url="https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct",
        dataset_source="InsuranceQA + Synthetic Claim Adjudication Cases (2-shot SFT)",
        license="Apache 2.0",
        our_modifications="Implemented in model_agent.py as INT8 ONNX; prompt v1 (0-shot) vs prompt v2 (2-shot); LoRA adapter training in PyTorch.",
        historical_results="Prompt v1: 25% acc, 0% accepted. Prompt v2: 75% acc, 100% valid JSON, 100% evidence fidelity, 75% verifier accepted. CPU latency: 36.98s.",
        final_project_role="Primary lightweight generative specialist agent drafting structured proposals under independent validator inspection.",
        laptop_feasibility="Feasible for LoRA SFT (CPU, bs=1, grad_accum=4, ~14 min/epoch) or INT8 ONNX evaluation (~2.5 min)",
        nvidia_feasibility="Full SFT or LoRA with bf16 on any GPU (bs=16, ~1 min/epoch)",
        priority="High"
    ),
    ModelMetadata(
        model_number=11,
        group_id="group_4",
        group_name="Generative Specialist Agents",
        model_name="Llama-3.2-1B-Instruct",
        checkpoint_name="meta-llama/Llama-3.2-1B-Instruct",
        architecture_paper_title="The Llama 3 Herd of Models",
        architecture_publication_year=2024,
        checkpoint_release_year=2024,
        parameter_count="1.23B",
        parameters_numeric=1_230_000_000,
        original_purpose="Lightweight multilingual instruction model featuring Grouped-Query Attention (GQA) and 128k context window.",
        original_objective="Autoregressive causal language modeling on 9T tokens, supervised fine-tuning, and DPO alignment.",
        project_objective="1B-tier specialist agent proposer: compares multi-clause policy reasoning and evidence extraction against 0.5B models.",
        paper_url="https://arxiv.org/abs/2407.21783",
        checkpoint_url="https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct",
        dataset_source="InsuranceQA + CUAD Contract SFT Split",
        license="Llama 3.2 Community License",
        our_modifications="Structured JSON schema prompting with few-shot demonstrations; parameter-efficient LoRA adapter training.",
        historical_results="Proposed SOTA 1B comparator; evaluated on structured assessment generation and verifier acceptance.",
        final_project_role="Intermediate-scale generative specialist agent balancing reasoning depth against inference throughput.",
        laptop_feasibility="Inference Smoke Test (CPU float32, 10 cases, ~12 min; Lab Only for Training)",
        nvidia_feasibility="LoRA / QLoRA with bf16 on 12GB+ GPU (bs=8, ~3 min/epoch)",
        priority="High"
    ),
    ModelMetadata(
        model_number=12,
        group_id="group_4",
        group_name="Generative Specialist Agents",
        model_name="Phi-3.5-mini-instruct",
        checkpoint_name="microsoft/Phi-3.5-mini-instruct",
        architecture_paper_title="Phi-3 Technical Report: Highly Capable Language Models Locally on Your Phone",
        architecture_publication_year=2024,
        checkpoint_release_year=2024,
        parameter_count="3.82B",
        parameters_numeric=3_820_000_000,
        original_purpose="Dense high-reasoning language model pre-trained on synthetic textbook data, web data, and extensive multi-turn DPO.",
        original_objective="Causal language modeling optimized for high-density multi-hop logic, math, and code reasoning with 128k context.",
        project_objective="University laboratory frontier reasoning ceiling: benchmarks whether 3.8B dense reasoning substantially improves verifier acceptance.",
        paper_url="https://arxiv.org/abs/2404.14219",
        checkpoint_url="https://huggingface.co/microsoft/Phi-3.5-mini-instruct",
        dataset_source="CUAD + Multi-Specialist Claim Adjudication Corpus",
        license="MIT",
        our_modifications="Evaluated strictly on laboratory GPUs via QLoRA/LoRA; role contracts enforced via independent deterministic validator.",
        historical_results="Laboratory upper-bound reasoning comparator; determines whether small models (0.5B/1B) suffer significant reasoning loss.",
        final_project_role="Theoretical performance ceiling representing the maximum capability of compact open-weight models in university lab.",
        laptop_feasibility="STRICTLY NVIDIA LAB ONLY (Loading causes immediate CPU RAM OOM crash; >16GB required)",
        nvidia_feasibility="QLoRA / LoRA on 16GB-24GB VRAM GPU (RTX 4090 / A100)",
        priority="Medium"
    ),
]

EXTERNAL_REFERENCE_ORACLE = {
    "model_name": "Deterministic Rule Engine Oracle",
    "architecture": "Symbolic Rule-Based Expert System (MandatoryRuleGate + RuleAgent)",
    "parameter_count": "0 (Pure symbolic logic)",
    "role": "External Reference Baseline (Arm F of Factorial Ablation)",
    "verified_accuracy": "100.0% decided safety (eliminates all 109 unsafe approvals across 150 ablation runs)",
    "decisional_authority": "Authoritative non-negotiable policy exclusion enforcement"
}

def get_model_registry() -> List[ModelMetadata]:
    """Returns the complete 12-model experimental registry."""
    return list(REGISTRY)

def get_models_by_group(group_id: str) -> List[ModelMetadata]:
    """Returns all models belonging to a specific research group."""
    return [m for m in REGISTRY if m.group_id == group_id]

def get_model_by_number(model_number: int) -> Optional[ModelMetadata]:
    """Returns a model by its sequential 1-12 index."""
    for m in REGISTRY:
        if m.model_number == model_number:
            return m
    return None

def export_registry_csv(output_path: str):
    """Exports the 12-model registry to a standard CSV file."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fieldnames = list(asdict(REGISTRY[0]).keys())
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for m in REGISTRY:
            writer.writerow(asdict(m))
    print(f"Exported 12-model registry to: {output_path}")

if __name__ == "__main__":
    export_registry_csv("results/12_Model_Registry.csv")
