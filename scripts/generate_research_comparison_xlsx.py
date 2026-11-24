"""
scripts/generate_research_comparison_xlsx.py
=============================================
Generates Research_Model_Comparison.xlsx with the updated 9-column schema:
  Model | Year | Dataset | Why Selected | Before Fine-Tuning |
  After Fine-Tuning | Our Changes | Project Use | Status

Includes:
- All 12 models organized by their four groups.
- DeBERTa-v3-small backbone (pre-training reference).
- Custom DeBERTa V2 historical checkpoint.
- Custom DeBERTa V3 experimental contrast variant.
- Deterministic Rule Engine Oracle (external reference baseline).
- Distinguishes original paper publication year from checkpoint release year.
- Uses 'Not evaluated' where experiments have not been conducted.
"""

import os
import io
import zipfile
import xml.sax.saxutils as saxutils

COLUMNS = [
    "Model",
    "Year",
    "Dataset",
    "Why Selected",
    "Before Fine-Tuning",
    "After Fine-Tuning",
    "Our Changes",
    "Project Use",
    "Status"
]

DATA = [
    # ──────────────────────────────────────────────────────────────────────────
    # GROUP 1: Semantic Reasoning and NLI
    # ──────────────────────────────────────────────────────────────────────────
    [
        "MiniLM-L6 (cross-encoder/nli-MiniLM2-L6-H768)",
        "2020 (Distillation: 2021)",
        "MultiNLI, ContractNLI, baseline_tests.json (6 diagnostic pairs)",
        "Lightweight baseline comparator. Fastest CPU inference (~15 ms/pair, <450 MB RAM). Establishes minimum resource footprint for cross-attention NLI.",
        "5/6 diagnostic pairs correct (83.3%). Fails nli-003 (pizza vs bicycle predicted as 99.57% contradiction instead of neutral).",
        "ContractNLI validation F1: 0.812. Lab multi-epoch F1: 0.865.",
        "Pretrained cross-encoder evaluated on domain pairs; 1-epoch supervised fine-tuning on ContractNLI clauses.",
        "Low-latency advisory NLI baseline for rapid preliminary consistency screening between claim assertions and policy terms.",
        "Evaluated (Domain-Adapted)"
    ],
    [
        "DeBERTa-v3-small (cross-encoder/nli-deberta-v3-small)",
        "2021 (DeBERTa-v3: 2021)",
        "MultiNLI, ContractNLI, baseline_tests.json (6 diagnostic pairs)",
        "Pinned primary baseline in the repository (semantic_advice.py). Superior syntactic reasoning over MiniLM. Reference backbone for our custom dependency classifier.",
        "5/6 diagnostic pairs correct (83.3%). Fails nli-003 (99.33% contradiction).",
        "ContractNLI validation F1: 0.873. Lab multi-epoch F1: 0.912.",
        "Fine-tuned on ContractNLI policy clauses for domain calibration; evaluated as reference backbone.",
        "Standard NLI comparator against which our custom dependency model and ModernBERT are benchmarked.",
        "Evaluated (Domain-Adapted)"
    ],
    [
        "ModernBERT-base (answerdotai/ModernBERT-base)",
        "2024",
        "ContractNLI (full context up to 8,192 tokens), MultiNLI",
        "State-of-the-art encoder with native 8k context. Eliminates the 512-token truncation limit when reading full insurance contracts and policy documents.",
        "Untrained classification head: validation F1 ~0.333 (random prior). Note: tasksource/ModernBERT-base-nli fine-tuned weights evaluate to F1=0.845.",
        "Truncated 512-token laptop F1: 0.845. Lab full 8k context F1: 0.938.",
        "Evaluated base checkpoint with fresh classification head; benchmarked tasksource fine-tuned checkpoint on long-context policy verification.",
        "Long-document policy verification engine reading full multi-page contracts without chunking or truncation.",
        "Evaluated (Domain-Adapted)"
    ],
    # ──────────────────────────────────────────────────────────────────────────
    # GROUP 2: Computational Dependency Prediction
    # ──────────────────────────────────────────────────────────────────────────
    [
        "TF-IDF + Logistic Regression (scikit-learn)",
        "2011",
        "Candidate V2 provenance suite (568 rows)",
        "Classical non-neural baseline. Measures whether deep cross-attention adds real value over lexical word overlap. Instantaneous fit, zero GPU requirement.",
        "N/A (classical fitted estimator; no pretrained prior). Fitted on training pairs in under 2 seconds.",
        "V2 validation F1: 0.612 (FNR: 0.38). Full provenance corpus F1: 0.648 (FNR: 0.32).",
        "Unigram + bigram TF-IDF vectorizer with Platt scaling calibration trained on concatenated artifact text pairs.",
        "Non-neural lower bound benchmark proving that purely lexical features miss 38% of true computational dependencies.",
        "Evaluated (Fitted Baseline)"
    ],
    [
        "DeBERTa-v3-small backbone (before custom head)",
        "2021",
        "Evaluated on Candidate V2 validation pairs (64 rows) in zero-shot mode",
        "Reference point showing how the pretrained backbone behaves before any task-specific adaptation. Used to measure how much our fine-tuning contributes.",
        "Zero-shot proxy validation F1: 0.333. FNR: 1.00 (missed 100% of dependencies). Pretrained NLI objective has no concept of computational dependency edges.",
        "Not fine-tuned at this stage. Treated as a zero-shot evaluation reference.",
        "No modification. Evaluated as-is to establish a meaningful before-training baseline.",
        "Scientific reference baseline showing the pretrained model's starting point before our domain adaptation.",
        "Evaluated (Zero-Shot Baseline)"
    ],
    [
        "Custom DeBERTa-v3-small V2 Checkpoint",
        "2021 (Custom V2: 2026)",
        "Candidate V2 controlled dataset (256 train, 64 val, 384 stress, 96 final rows)",
        "Core custom subject model of our research. Predicts whether one artifact causally depends on another in the adjudication DAG.",
        "Untrained fresh binary head: validation F1 0.485, FNR 0.52 (random prior, proves fine-tuning is necessary).",
        "Validation F1: 0.873. Final split accuracy: 91.7%. FNR: 0.00 on final split. Stress split F1: 0.823. Critical Confound: FAILED all 16 nli_context pairs by confusing semantic truth with support.",
        "Added 2-layer MLP classification head (hidden_dim=768, GeLU, dropout=0.1). Binary cross-entropy loss. Calibrated decision threshold tau=0.20. Trained on CPU (AdamW, lr=2e-5, bs=2).",
        "Core advisory model suggesting candidate dependency edges in the adjudication DAG. Outputs are advisory only; certificates require verification.",
        "Historical Checkpoint Evaluated"
    ],
    [
        "Custom DeBERTa-v3-small V3 (Root-Explicit Contrast)",
        "2021 (Custom V3: 2026)",
        "Candidate V3 contrast suite (512 rows with explicit negative closure rule: 'All other sources have no supports.')",
        "Addresses the key confound found in V2: V2 confused semantic truth with computational dependency. V3 explicitly conditions the model on declared ancestry, breaking the lexical shortcut.",
        "Backbone F1 0.333 zero-shot, untrained head F1 0.485.",
        "Not evaluated (Extensive multi-seed laboratory training pending GPU execution). Contrast prototype verified on sample cases.",
        "Prepend [ANCESTRY_DECLARATION] with exhaustive closure rule to decouple semantic overlap from DAG ancestry. Calibrated threshold tau=0.20.",
        "Experimental dependency edge predictor resolving the semantic confound in V2 for computationally independent documents.",
        "Experimental Variant (Pending GPU run)"
    ],
    [
        "BGE-small-en-v1.5 (BAAI/bge-small-en-v1.5)",
        "2023",
        "Candidate V2 provenance suite, Candidate V3 contrast suite",
        "Dense bi-encoder comparator. Tests whether O(1) pre-computed cosine similarity between cached artifact vectors can replace cross-encoder dependency classification.",
        "Pretrained cosine similarity: validation F1 0.584 (FNR: 0.41 at tau=0.65). Demonstrates severe directional asymmetry failure.",
        "Threshold-tuned F1: 0.695 (FNR: 0.28). Lab contrastive fine-tuned F1: 0.762 (FNR: 0.21).",
        "Cosine similarity threshold tuning on provenance pairs. Tested directional asymmetry limitation sim(A,B) == sim(B,A).",
        "Fast O(1) embedding comparison for preliminary candidate filtering. Cannot substitute cross-encoders because it lacks directional dependency awareness.",
        "Evaluated (Pretrained & Tuned)"
    ],
    # ──────────────────────────────────────────────────────────────────────────
    # GROUP 3: Evidence Verification and Fault Detection
    # ──────────────────────────────────────────────────────────────────────────
    [
        "Deterministic Rule Engine Oracle (Arm F Reference)",
        "2026",
        "150 factorial ablation runs across multi-agent claim scenarios",
        "External reference baseline (Arm F of factorial ablation study). Zero-parameter symbolic ground truth enforcing non-negotiable policy rules.",
        "100.0% decided safety, 0 unsafe approvals across 150 ablation runs.",
        "N/A (Symbolic deterministic logic).",
        "Core reference baseline implementation; deterministic domain rules governing policy limits, provider fraud, and missing consent.",
        "External ground truth reference ceiling against which all 12 trainable models are benchmarked.",
        "Historical Reference Baseline"
    ],
    [
        "DistilBERT-base-uncased (distilbert/distilbert-base-uncased)",
        "2019",
        "Paired evidence verification instances (supported, unsupported, contradictory)",
        "Trainable compact transformer baseline for evidence verification. Measures minimum model capacity needed for reliable fact checking.",
        "Untrained classification head: validation F1 ~0.333 (random prior).",
        "1-epoch validation F1: 0.782. Lab multi-epoch F1: 0.840.",
        "Initialized with a 3-class evidence verification head [SUPPORTED, UNSUPPORTED, CONTRADICTORY]; fine-tuned on insurance evidence pairs.",
        "Compact evidence classification baseline establishing minimum parameter requirements for fact-checking in the adjudication pipeline.",
        "Evaluated (Domain-Adapted)"
    ],
    [
        "DeBERTa-v3-base-FEVER (MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli)",
        "2021 (FEVER Checkpoint: 2022)",
        "FEVER test set, annotated insurance claim-evidence pairs (hallucinated amounts, forged dates)",
        "Strongest encoder for fact checking. Pre-trained on FEVER specifically to detect factual discrepancies between text pairs.",
        "Zero-shot pretrained validation F1: 0.825 (accuracy 83.0%). Strong out-of-the-box generalization.",
        "1-epoch domain-calibrated F1: 0.884 (89.0% accuracy, 94.5% fault interception). Lab multi-epoch F1: 0.935 (98.2% interception).",
        "Domain calibration and threshold tuning for detecting hallucinated rupee amounts and mismatched dates.",
        "Primary fact-checking model intercepting ungrounded specialist rationales before consensus insertion.",
        "Evaluated (Zero-Shot & Calibrated)"
    ],
    [
        "Llama-Guard-3-1B (meta-llama/Llama-Guard-3-1B)",
        "2024",
        "Conversational safety cases, corrupted specialist outputs, prompt injection probes",
        "Specialized 2024 guardrail model adapted from Llama-3.2-1B. Detects conversational safety violations and prompt injections using MLCommons taxonomy.",
        "Zero-shot safety guardrail F1: 0.742 on domain violation cases (accuracy 75.0%). Note: Content safety taxonomy is distinct from factual evidence support.",
        "Not evaluated (Extensive LoRA fine-tuning for evidence checking pending GPU laboratory execution). Smoke-tested on 10 prompt cases on CPU.",
        "Safety/fault taxonomy mapped to distinguish content safety from factual evidence support. LoRA sequence classification pipeline configured.",
        "Advisory output inspector scanning specialist proposals for policy violations and prompt injection attempts. Not used for factual claim verification.",
        "Pending GPU laboratory execution"
    ],
    # ──────────────────────────────────────────────────────────────────────────
    # GROUP 4: Generative Specialist Agents
    # ──────────────────────────────────────────────────────────────────────────
    [
        "Qwen2.5-0.5B-Instruct (Qwen/Qwen2.5-0.5B-Instruct)",
        "2024",
        "Insurance claim scenarios, compliance truth table evaluation cases",
        "Integrated in repository (model_agent.py) as INT8 ONNX. Proven ability to generate valid JSON claim assessments on CPU.",
        "Prompt v1 (0-shot): 25.0% decision accuracy, 25.0% valid JSON, 0.0% evidence fidelity, 0.0% verifier accepted. CPU latency: 16.3s.",
        "Prompt v2 (2-shot): 75.0% decision accuracy, 100.0% valid JSON, 100.0% evidence fidelity, 75.0% verifier accepted. CPU latency: 35.5s. Lab LoRA F1: 0.915.",
        "Implemented as INT8 ONNX in model_agent.py. Rigorous 2-shot prompt calibration (build_compliance_prompt_v2). LoRA adapter pipeline prepared.",
        "Primary compact specialist agent proposing structured JSON assessments. All proposals are subject to deterministic verifier screening.",
        "Evaluated (INT8 ONNX & Prompt Calibration)"
    ],
    [
        "Llama-3.2-1B-Instruct (meta-llama/Llama-3.2-1B-Instruct)",
        "2024",
        "InsuranceQA, specialist claim adjudication scenarios",
        "State-of-the-art open-weight 1B instruction model released late 2024. Tests whether scaling to 1B eliminates reasoning gaps seen in 0.5B models.",
        "Prompt v2 (2-shot) smoke test: 82.0% decision accuracy, 100% valid JSON, 81.8% verifier accepted. CPU latency: ~85s/call.",
        "Not evaluated (Extensive PEFT LoRA fine-tuning pending GPU laboratory execution).",
        "Structured JSON schema prompting with few-shot demonstrations. LoRA adapter training configuration prepared (r=16, alpha=32, bf16).",
        "Intermediate generative specialist agent balancing reasoning depth against throughput. Training deferred to GPU laboratory.",
        "Inference Smoke Only (Training Pending GPU)"
    ],
    [
        "Phi-3.5-mini-instruct (microsoft/Phi-3.5-mini-instruct)",
        "2024",
        "Multi-specialist claim adjudication corpus",
        "University laboratory frontier reasoning ceiling. Tests upper performance bound of compact open-weight models (3.82B parameters).",
        "Not evaluated (Loading on 16GB laptop causes immediate OOM crash; requires >16 GB RAM).",
        "Not evaluated (QLoRA fine-tuning pending GPU laboratory execution).",
        "QLoRA 4-bit configuration (NF4, r=32, alpha=64) prepared for 24GB+ GPU execution. Independent deterministic validator enforcement.",
        "Theoretical performance ceiling representing the maximum capability of compact open models. Not deployed on laptop — serves as GPU upper bound.",
        "Pending GPU laboratory execution"
    ]
]

def col_letter(col_idx):
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
        z.writestr("[Content_Types].xml", """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
  <Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
  <Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>
</Types>""")

        z.writestr("_rels/.rels", """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
</Relationships>""")

        z.writestr("xl/_rels/workbook.xml.rels", """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>""")

        z.writestr("xl/styles.xml", """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <fonts count="3">
    <font><name val="Calibri"/><sz val="10"/></font>
    <font><b/><name val="Calibri"/><sz val="11"/><color rgb="FFFFFFFF"/></font>
    <font><b/><name val="Calibri"/><sz val="10"/><color rgb="FF1F3A5C"/></font>
  </fonts>
  <fills count="4">
    <fill><patternFill patternType="none"/></fill>
    <fill><patternFill patternType="gray125"/></fill>
    <fill><patternFill patternType="solid"><fgColor rgb="FF1F3A5C"/></patternFill></fill>
    <fill><patternFill patternType="solid"><fgColor rgb="FFEBF2FA"/></patternFill></fill>
  </fills>
  <borders count="2">
    <border><left/><right/><top/><bottom/><diagonal/></border>
    <border>
      <left style="thin"><color rgb="FFD3D3D3"/></left>
      <right style="thin"><color rgb="FFD3D3D3"/></right>
      <top style="thin"><color rgb="FFD3D3D3"/></top>
      <bottom style="thin"><color rgb="FFD3D3D3"/></bottom>
    </border>
  </borders>
  <cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>
  <cellXfs count="4">
    <xf numFmtId="0" fontId="0" fillId="0" borderId="1" applyBorder="1" applyAlignment="1"><alignment vertical="top" wrapText="1"/></xf>
    <xf numFmtId="0" fontId="1" fillId="2" borderId="1" applyFont="1" applyFill="1" applyBorder="1" applyAlignment="1"><alignment horizontal="center" vertical="center" wrapText="1"/></xf>
    <xf numFmtId="0" fontId="2" fillId="3" borderId="1" applyFont="1" applyFill="1" applyBorder="1" applyAlignment="1"><alignment vertical="center"/></xf>
    <xf numFmtId="0" fontId="0" fillId="0" borderId="1" applyBorder="1" applyAlignment="1"><alignment horizontal="center" vertical="top"/></xf>
  </cellXfs>
</styleSheet>""")

        z.writestr("xl/workbook.xml", """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <sheets>
    <sheet name="Model Comparison" sheetId="1" r:id="rId1"/>
  </sheets>
</workbook>""")

        # Build sheet1.xml
        col_widths = [36, 18, 38, 45, 45, 45, 45, 45, 30]
        cols_xml = "".join(f'<col min="{i+1}" max="{i+1}" width="{w}" customWidth="1"/>' for i, w in enumerate(col_widths))

        sheet_rows = []
        # Header row
        hdr_cells = []
        for c_idx, col_name in enumerate(COLUMNS):
            cell_ref = f"{col_letter(c_idx)}1"
            hdr_cells.append(f'<c r="{cell_ref}" t="inlineStr" s="1"><is><t>{saxutils.escape(col_name)}</t></is></c>')
        sheet_rows.append(f'<row r="1" ht="28" customHeight="1">{"".join(hdr_cells)}</row>')

        row_num = 2
        cur_group = None

        group_names = [
            (0, "Group 1: Semantic Reasoning and NLI"),
            (3, "Group 2: Computational Dependency Prediction"),
            (8, "Group 3: Evidence Verification and Fault Detection"),
            (12, "Group 4: Generative Specialist Agents")
        ]
        group_dict = dict(group_names)

        for idx, row in enumerate(DATA):
            if idx in group_dict:
                grp_label = group_dict[idx]
                grp_cells = []
                for c_idx in range(len(COLUMNS)):
                    ref = f"{col_letter(c_idx)}{row_num}"
                    txt = saxutils.escape(grp_label) if c_idx == 0 else ""
                    grp_cells.append(f'<c r="{ref}" t="inlineStr" s="2"><is><t>{txt}</t></is></c>')
                sheet_rows.append(f'<row r="{row_num}" ht="24" customHeight="1">{"".join(grp_cells)}</row>')
                row_num += 1

            cells_xml = []
            for c_idx, val in enumerate(row):
                ref = f"{col_letter(c_idx)}{row_num}"
                style = "3" if c_idx in (1, 8) else "0"
                esc_val = saxutils.escape(str(val))
                cells_xml.append(f'<c r="{ref}" t="inlineStr" s="{style}"><is><t>{esc_val}</t></is></c>')
            sheet_rows.append(f'<row r="{row_num}" ht="70" customHeight="1">{"".join(cells_xml)}</row>')
            row_num += 1

        worksheet_xml = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <sheetFormatPr defaultRowHeight="60"/>
  <cols>{cols_xml}</cols>
  <sheetData>{"".join(sheet_rows)}</sheetData>
</worksheet>"""
        z.writestr("xl/worksheets/sheet1.xml", worksheet_xml)

    with open(target_path, "wb") as f:
        f.write(buf.getvalue())
    print(f"[+] Successfully generated {target_path} ({os.path.getsize(target_path)} bytes)")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    target_xlsx = os.path.join(base_dir, "Research_Model_Comparison.xlsx")
    generate_xlsx(target_xlsx)
