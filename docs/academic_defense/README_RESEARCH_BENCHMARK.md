# Research Benchmark Framework
**Project:** Verifiable Multi-Agent Fault-Tolerant Adjudication System with Cryptographic Causal Provenance  
**Repository:** [https://github.com/Its-soul/cap-stone](https://github.com/Its-soul/cap-stone)  
**Author:** Senior AI/ML Research Scientist & Deep Learning Specialist  

---

## 1. Overview

This framework implements an academically defensible, reproducible experimental pipeline evaluating **12 AI/ML models across four distinct task groups**:

1. **Group 1: Semantic Reasoning and NLI** (MiniLM-L6, DeBERTa-v3-small, ModernBERT-base)
2. **Group 2: Computational Dependency Prediction** (TF-IDF + Logistic Regression, Custom DeBERTa-v3 Dependency, BGE-small Bi-Encoder)
3. **Group 3: Evidence Verification and Fault Detection** (DistilBERT-base, DeBERTa-v3 FEVER, Llama Guard 3)
4. **Group 4: Generative Specialist Agents** (Qwen2.5-0.5B, Llama-3.2-1B, Phi-3.5-mini)
5. **Reference Baseline:** Deterministic Rule Engine Oracle (Arm F of Factorial Ablation)

---

## 2. Directory Structure

```
├── research_benchmark/
│   ├── __init__.py               # Package metadata
│   ├── registry.py               # 12-model registry with 15 fields per model
│   ├── hardware.py               # Resource profiling and CPU memory guards
│   ├── dataset_loaders.py        # Split handlers, schema validators, fixtures
│   ├── model_adapters.py         # Unified model adapters for all 12 models
│   ├── trainers.py               # Laptop 1-Epoch & NVIDIA Multi-Epoch loops
│   ├── evaluation.py             # Metrics, McNemar's test, bootstrap CIs, FNR
│   ├── experiment_logger.py      # Telemetry, dataset fingerprints, run logs
│   ├── checkpoint_manager.py     # Safe isolated checkpoint management
│   └── visualization.py          # Generates the 12 scientific figures
│
├── notebooks/
│   ├── 01_Laptop_One_Epoch_12_Model_Benchmark.ipynb  # Feasibility experiment
│   └── 02_NVIDIA_Lab_Extensive_Training.ipynb        # Multi-epoch GPU roadmap
│
├── results/
│   ├── Model_Journey_For_Mentor.xlsx       # 20-column mentor workbook
│   ├── Custom_V2_V3_Model_Evolution.xlsx   # Custom model evolution workbook
│   ├── 12_Model_Registry.csv               # CSV export of registry
│   └── figures/                            # 12 publication-grade figures
│
├── docs/
│   ├── Research_Model_Selection_Report.md
│   ├── One_Epoch_Feasibility_Report.md
│   ├── NVIDIA_Extensive_Training_Report.md
│   ├── Custom_Model_Contribution.md
│   └── Teacher_Viva_Explanation.md
│
├── research_config.yaml          # Centralized hyperparameter configuration
├── requirements-laptop.txt       # Minimal CPU laptop requirements
└── requirements-nvidia.txt       # University GPU cluster requirements
```

---

## 3. Quickstart & Execution

### Running the Visualizations:
```bash
python -m research_benchmark.visualization
```

### Running Unit Tests on Framework Modules:
```bash
python -m research_benchmark.hardware
python -m research_benchmark.dataset_loaders
python -m research_benchmark.model_adapters
python -m research_benchmark.evaluation
```

### Launching the Jupyter Notebooks:
```bash
jupyter notebook notebooks/01_Laptop_One_Epoch_12_Model_Benchmark.ipynb
jupyter notebook notebooks/02_NVIDIA_Lab_Extensive_Training.ipynb
```
