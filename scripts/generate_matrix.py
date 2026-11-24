import os
import csv
import io
import zipfile
import xml.sax.saxutils as saxutils

# 17 Required Columns:
# Model | Publication Year | Release Year | Model Architecture | Parameters | Research Paper | 
# Paper URL | Checkpoint URL | Target Task | Dataset | Why Selected | Advantages | Disadvantages | 
# Laptop Feasibility | NVIDIA Feasibility | Metrics | Priority

COLUMNS = [
    "Model",
    "Publication Year",
    "Release Year",
    "Model Architecture",
    "Parameters",
    "Research Paper",
    "Paper URL",
    "Checkpoint URL",
    "Target Task",
    "Dataset",
    "Why Selected",
    "Advantages",
    "Disadvantages",
    "Laptop Feasibility",
    "NVIDIA Feasibility",
    "Metrics",
    "Priority"
]

DATA = [
    [
        "cross-encoder/nli-MiniLM2-L6-H768",
        "2020",
        "2021",
        "6-layer Transformer Cross-Encoder (H=768, 12 heads)",
        "66M",
        "MiniLMv2: Multi-Head Self-Attention Relation Distillation for Compressing Pretrained Transformers",
        "https://arxiv.org/abs/2012.15828",
        "https://huggingface.co/cross-encoder/nli-MiniLM2-L6-H768",
        "NLI & Semantic Consistency Baseline (Task Group 1)",
        "MultiNLI + ContractNLI + baseline_tests.json",
        "Historical repository baseline; ultra-lightweight transformer cross-encoder reference.",
        "Fast CPU inference (~15ms/pair); tiny memory footprint (<250MB RAM); low latency.",
        "High lexical mismatch penalty; fails nli-003 (pizza/bicycle treated as 99.57% contradiction).",
        "Fully Feasible (CPU float32, bs=8, ~1.5 min/epoch)",
        "Trivial (<20 sec/epoch, fp16/bf16)",
        "Macro-F1, Accuracy, Contradiction Precision, Cross-Entropy Loss",
        "High"
    ],
    [
        "cross-encoder/nli-deberta-v3-small",
        "2021",
        "2021",
        "Disentangled Attention Encoder + ELECTRA-style RTD Head",
        "141M",
        "DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing",
        "https://arxiv.org/abs/2111.09543",
        "https://huggingface.co/cross-encoder/nli-deberta-v3-small",
        "Primary NLI & Semantic Advice Benchmark (Task Group 1)",
        "MultiNLI + ContractNLI + baseline_tests.json",
        "Pinned primary baseline in repository (semantic_advice.py); SOTA NLI accuracy for small encoders.",
        "Superior syntactic reasoning and boundary modeling via disentangled matrices.",
        "~3x higher CPU latency than MiniLM; also fails nli-003 neutral test (99.33% contradiction).",
        "Fully Feasible (CPU float32, bs=4, ~4 min/epoch)",
        "Standard Full Fine-Tuning (fp16/bf16, bs=32, ~1 min/epoch)",
        "Macro-F1, Accuracy, Expected Calibration Error (ECE), Latency",
        "High"
    ],
    [
        "tasksource/ModernBERT-base-nli",
        "2024",
        "2024",
        "Modernized Bidirectional Encoder with RoPE, GeGLU, FlashAttention-2, Unpadding",
        "149M",
        "Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder for Fast, Memory-Efficient, and Long-Context Understanding",
        "https://arxiv.org/abs/2412.13663",
        "https://huggingface.co/tasksource/ModernBERT-base-nli",
        "Long-Context Document NLI & Policy Verification (Task Group 1)",
        "ContractNLI + MultiNLI (Full context up to 8,192 tokens)",
        "Cutting-edge 2024 architecture with native 8k context window; eliminates 512-token truncation limit.",
        "Native 8,192 tokens without chunking; FlashAttention-2 speedups; superior long-document modeling.",
        "Requires PyTorch 2.2+; newly published weights require rigorous CPU baseline evaluation.",
        "Feasible with truncated context (seq_len=512, CPU float32, bs=2, ~6 min/epoch)",
        "Ideal (Full 8k context training with FlashAttention-2 and bf16)",
        "Macro-F1, Long-Context Retrieval Accuracy, Truncation Penalty, Latency",
        "High"
    ],
    [
        "TF-IDF + Calibrated Logistic Regression",
        "2011",
        "2023",
        "Sparse N-gram Vectorizer (unigram/bigram) + L2 Logistic Regression with Platt Scaling",
        "~50k sparse",
        "Scikit-learn: Machine Learning in Python",
        "https://jmlr.csail.mit.edu/papers/v12/pedregosa11a.html",
        "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html",
        "Linear Causal Dependency Baseline (Task Group 2)",
        "Candidate V2 & Candidate V3 Provenance Suites",
        "Classical, fully interpretable ML baseline to test if deep neural representations are truly necessary.",
        "Instantaneous training (<1s); microsecond inference; 100% deterministic; zero GPU/memory load.",
        "Incapable of abstract semantic reasoning; strictly dependent on lexical co-occurrence.",
        "Fully Feasible (CPU, <2 seconds total run time)",
        "CPU Bound / Trivial (runs instantaneously on any server)",
        "Macro-F1, False Negative Rate (Missed Dependencies), ROC-AUC, Sparsity",
        "High"
    ],
    [
        "DeBERTa-v3-small-dep",
        "2021",
        "2026",
        "DeBERTa-v3-small Encoder + 2-layer MLP Binary Classification Head",
        "141M",
        "DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing",
        "https://arxiv.org/abs/2111.09543",
        "Local repo checkpoint _private/models/candidate_v2/",
        "Neural Computational Dependency Classifier (Task Group 2)",
        "Candidate V2 & Candidate V3 Root-Explicit Contrast Suites",
        "Core fine-tuned subject model; requires comparison against V3 dataset to evaluate fix for NLI confound.",
        "Deep cross-attention over artifact content and ancestral declarations.",
        "Past checkpoint exhibited NLI confound (failed on 16 nli_context test pairs by confusing semantic truth with support rules).",
        "Fully Feasible (CPU float32, bs=2, lr=2e-5, ~5 min/epoch)",
        "Full Fine-Tuning across 10 epochs with multi-seed variance analysis (~3 min total)",
        "Macro-F1, Missed Dependency False Negative Rate (FNR), Precision, ECE",
        "High"
    ],
    [
        "BAAI/bge-small-en-v1.5",
        "2023",
        "2023",
        "12-layer MiniLM-style Bi-Encoder producing 384-dimensional dense vectors",
        "33.5M",
        "C-Pack: Packaged Resources To Advance General Chinese and English Embedding",
        "https://arxiv.org/abs/2309.07597",
        "https://huggingface.co/BAAI/bge-small-en-v1.5",
        "Dense Embedding Similarity vs Causal Dependency Baseline (Task Group 2)",
        "Candidate V2 & Candidate V3 Provenance Suites",
        "Tests whether bi-encoder cosine similarity can substitute for cross-encoder dependency classification.",
        "Ultra-compact (33.5M params, <70MB); embeddings can be pre-computed and cached; rapid O(1) cosine similarity.",
        "Bi-encoders lack cross-attention; cannot resolve asymmetric directional dependency (A -> B != B -> A).",
        "Fully Feasible (CPU float32, bs=16, ~1 min/epoch)",
        "Trivial (<30 sec/epoch)",
        "ROC-AUC, Macro-F1 at optimal threshold, Directional Asymmetry Error Rate",
        "Medium"
    ],
    [
        "Deterministic Rule Engine Oracle",
        "2026",
        "2026",
        "Symbolic Rule-Based Expert System (MandatoryRuleGate + RuleAgent)",
        "0",
        "Verifiable Multi-Agent Fault-Tolerant Adjudication System (This Project)",
        "https://github.com/Its-soul/cap-stone",
        "Local repo implementation (src/cert_recovery/)",
        "Ground-Truth Invariant Oracle & Fault Baseline (Task Group 3)",
        "Factorial Ablation Scenarios (150 trials) + Domain Claim Scenarios",
        "Represents Arm F of Factorial Ablation; establishes the 100% deterministic safety ceiling.",
        "Zero hallucination; zero latency (<1ms); 100% reproducible; formal mathematical certainty.",
        "Cannot process unformatted free text or handle unanticipated syntactic variations without code updates.",
        "Fully Feasible (CPU, <100ms total runtime)",
        "CPU Bound / Trivial",
        "Unsafe Approvals (0 required), Decided Accuracy, Deterministic Invariant Preservation",
        "High"
    ],
    [
        "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
        "2022",
        "2023",
        "DeBERTa-v3-base Cross-Encoder fine-tuned on MNLI + FEVER + ANLI",
        "435M",
        "Less Annotating, More Classifying: Addressing the Data Scarcity Problem of Supervised Natural Language Technologies with Zero-Shot NLI",
        "https://arxiv.org/abs/2204.10842",
        "https://huggingface.co/MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
        "Claim-Evidence Verification & Hallucination Interception (Task Group 3)",
        "FEVER test set + Annotated Insurance Claim-Evidence Pairs",
        "Pre-trained on FEVER fact-verification; verifies whether specialist rationale is factually grounded in documents.",
        "High sensitivity to factual contradictions and unsupported assertions; robust to adversarial phrasing.",
        "Larger memory footprint (~800MB RAM); slower CPU inference (~180ms/pair).",
        "Feasible for 1-epoch evaluation (CPU float32, bs=2, ~10 min/epoch)",
        "Standard Fine-Tuning (fp16/bf16, bs=16, ~2 min/epoch)",
        "Macro-F1, Unsupported Claim Interception Rate, False Alarm Rate, ECE",
        "High"
    ],
    [
        "meta-llama/Llama-Guard-3-1B",
        "2024",
        "2024",
        "16-layer Transformer Decoder with Safety Classification Heads (Llama-3.2 backbone)",
        "1.23B",
        "Llama Guard 3: Guardrail Modeling with Llama-3.2",
        "https://arxiv.org/abs/2407.21783",
        "https://huggingface.co/meta-llama/Llama-Guard-3-1B",
        "Specialist Output Guardrailing & Corruption Interception (Task Group 3)",
        "HaluEval + Synthetic Corrupted Agent Outputs + Policy Risk Prompts",
        "Purpose-built 2024 guardrail model; classifies whether agent proposals violate policy or exhibit anomalies.",
        "High coverage of subtle policy violations, injection attempts, and corrupt JSON generation.",
        "1.23B parameters is heavy for laptop CPU training (requires 4-bit quantization or inference-only on laptop).",
        "Inference-Only via GGUF/INT4 or 1-epoch LoRA test on tiny batch (bs=1, grad_accum=4, ~25 min/epoch)",
        "LoRA / QLoRA Fine-Tuning on 8GB-16GB VRAM GPUs (~5 min/epoch)",
        "Policy Violation Precision/Recall, Latency Overhead, False Veto Rate",
        "Medium"
    ],
    [
        "Qwen/Qwen2.5-0.5B-Instruct",
        "2024",
        "2024",
        "24-layer Transformer Decoder with RoPE, RMSNorm, SwiGLU, tied embeddings",
        "494M",
        "Qwen2.5: A Foundation Language Model Series",
        "https://arxiv.org/abs/2412.15115",
        "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct",
        "Compact Generative Specialist Agent Proposer (Task Group 4)",
        "InsuranceQA + Synthetic Pilot Claim Scenarios (2-shot SFT)",
        "Integrated into repo (src/cert_recovery/model_agent.py); verified ability to generate valid JSON claim assessments.",
        "Ultra-compact (<500MB in INT8); 100% valid JSON adherence with 2-shot prompting in empirical testing.",
        "High CPU generation latency (mean 36.98s per call on laptop); limited multi-hop logical capacity.",
        "Feasible for 1-epoch LoRA fine-tuning (CPU float32, bs=1, grad_accum=4, ~15 min/epoch) or INT8 ONNX evaluation",
        "Full SFT or LoRA with bf16 on any GPU (bs=16, ~1 min/epoch)",
        "Valid JSON Rate (100% required), Decision Accuracy, Evidence Fidelity, Verifier Acceptance, Latency",
        "High"
    ],
    [
        "meta-llama/Llama-3.2-1B-Instruct",
        "2024",
        "2024",
        "16-layer Transformer Decoder with Grouped-Query Attention (GQA), 128k context",
        "1.23B",
        "The Llama 3 Herd of Models",
        "https://arxiv.org/abs/2407.21783",
        "https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct",
        "1B-Tier Generative Specialist Agent Proposer (Task Group 4)",
        "InsuranceQA + CUAD Contract SFT Split",
        "SOTA open-weight 1B instruction model released in late 2024; superior reasoning and instruction following.",
        "Grouped-Query Attention (GQA) enables fast generation; strong multi-clause reasoning; 128k context window.",
        "Requires parameter-efficient fine-tuning (LoRA); generation latency on CPU is high without GPU acceleration.",
        "Inference-Only / 1-epoch LoRA smoke trial (CPU float32, bs=1, grad_accum=8, ~30 min/epoch)",
        "LoRA / QLoRA with bf16 on 12GB+ GPU (bs=8, ~3 min/epoch)",
        "Decision Accuracy, Structured JSON Validity, Verifier Acceptance Rate, Rationale Faithfulness",
        "High"
    ],
    [
        "microsoft/Phi-3.5-mini-instruct",
        "2024",
        "2024",
        "32-layer Transformer Decoder, 128k context, synthetic textbook + DPO pretraining",
        "3.82B",
        "Phi-3 Technical Report: Highly Capable Language Models Locally on Your Phone",
        "https://arxiv.org/abs/2404.14219",
        "https://huggingface.co/microsoft/Phi-3.5-mini-instruct",
        "University Laboratory Frontier Reasoning Benchmark (Task Group 4)",
        "CUAD + Multi-Specialist Claim Adjudication Corpus",
        "Serves as the laboratory upper-bound reasoning ceiling; benchmarks whether 3.8B dense reasoning substantially improves verifier acceptance.",
        "World-class reasoning performance for a compact model; rivaling 7B-14B models on complex deductive logic.",
        "Exceeds laptop CPU training feasibility; strictly requires NVIDIA GPU laboratory hardware.",
        "Marked as 'NVIDIA Lab Only' (Training infeasible on laptop CPU; inference only via 4-bit quantization)",
        "QLoRA / LoRA on 16GB-24GB VRAM GPU (RTX 4090 / A100)",
        "Multi-Clause Adjudication Macro-F1, Verifier Rejection Rate, Reasoning Depth Score, Latency",
        "Medium"
    ]
]

def generate_csv(target_path):
    with open(target_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(COLUMNS)
        for row in DATA:
            writer.writerow(row)
    print(f"Generated CSV: {target_path}")

def col_name(col_idx):
    # 0 -> A, 1 -> B, ..., 16 -> Q
    name = ""
    idx = col_idx
    while True:
        name = chr(65 + (idx % 26)) + name
        idx = (idx // 26) - 1
        if idx < 0:
            break
    return name

def generate_xlsx(target_path):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        # [Content_Types].xml
        z.writestr("[Content_Types].xml", """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
  <Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
  <Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>
</Types>""")

        # _rels/.rels
        z.writestr("_rels/.rels", """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
</Relationships>""")

        # xl/_rels/workbook.xml.rels
        z.writestr("xl/_rels/workbook.xml.rels", """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>""")

        # xl/styles.xml
        z.writestr("xl/styles.xml", """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <fonts count="2">
    <font><name val="Calibri"/><sz val="11"/></font>
    <font><b/><name val="Calibri"/><sz val="11"/><color rgb="FFFFFFFF"/></font>
  </fonts>
  <fills count="3">
    <fill><patternFill patternType="none"/></fill>
    <fill><patternFill patternType="gray125"/></fill>
    <fill><patternFill patternType="solid"><fgColor rgb="FF1F497D"/><bgColor indexed="64"/></patternFill></fill>
  </fills>
  <borders count="1">
    <border><left/><right/><top/><bottom/></border>
  </borders>
  <cellStyleXfs count="1">
    <xf numFmtId="0" fontId="0" fillId="0" borderId="0"/>
  </cellStyleXfs>
  <cellXfs count="2">
    <xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>
    <xf numFmtId="0" fontId="1" fillId="2" borderId="0" xfId="0" applyFont="1" applyFill="1"/>
  </cellXfs>
</styleSheet>""")

        # xl/workbook.xml
        z.writestr("xl/workbook.xml", """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <sheets>
    <sheet name="12_Model_Research_Matrix" sheetId="1" r:id="rId1"/>
  </sheets>
</workbook>""")

        # xl/worksheets/sheet1.xml
        sheet_rows = []
        # Header row
        header_cells = []
        for c_idx, col_name_str in enumerate(COLUMNS):
            cell_ref = f"{col_name(c_idx)}1"
            escaped = saxutils.escape(col_name_str)
            header_cells.append(f'<c r="{cell_ref}" t="inlineStr" s="1"><is><t>{escaped}</t></is></c>')
        sheet_rows.append(f'<row r="1">{"".join(header_cells)}</row>')

        # Data rows
        for r_idx, row_vals in enumerate(DATA, start=2):
            row_cells = []
            for c_idx, val in enumerate(row_vals):
                cell_ref = f"{col_name(c_idx)}{r_idx}"
                escaped = saxutils.escape(str(val))
                row_cells.append(f'<c r="{cell_ref}" t="inlineStr"><is><t>{escaped}</t></is></c>')
            sheet_rows.append(f'<row r="{r_idx}">{"".join(row_cells)}</row>')

        sheet_xml = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <sheetData>
    {"".join(sheet_rows)}
  </sheetData>
</worksheet>"""
        z.writestr("xl/worksheets/sheet1.xml", sheet_xml)

    with open(target_path, "wb") as f:
        f.write(buf.getvalue())
    print(f"Generated XLSX: {target_path} ({len(buf.getvalue())} bytes)")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(base_dir, "12_model_research_matrix.csv")
    xlsx_path = os.path.join(base_dir, "12_model_research_matrix.xlsx")
    generate_csv(csv_path)
    generate_xlsx(xlsx_path)
