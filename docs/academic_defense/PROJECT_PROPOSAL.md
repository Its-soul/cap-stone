# Original project proposal

The following description is preserved verbatim from the supplied proposal. Its guarantees describe the final research direction, not validated milestone results. See [prototype scope](RESEARCH_PROTOTYPE.md) for the implemented boundary.

Verifiable Multi-Agent Fault-Tolerant Adjudication System with Cryptographic Causal Provenance

Area/Domain of Project:
Trustworthy Multi-Agent Systems / Applied Cryptography & FinTech Compliance

ABSTRACT:
Multi-agent Large Language Model (LLM) pipelines are increasingly deployed to automate high-stakes decision workflows across insurance claims adjudication, financial fraud auditing, and enterprise regulatory compliance. However, existing multi-agent architectures suffer from severe vulnerability to cascading failures, where a single hallucinating or adversarially compromised agent silently poisons downstream reasoning by passing flawed outputs as ground truth. Furthermore, current frameworks rely on unstructured natural-language debates or naive majority voting, generating untraceable conversational transcripts rather than the deterministic, tamper-evident audit trails mandated by legal and regulatory frameworks. To bridge these critical gaps, this project designs and implements a verifiable, zero-trust multi-agent adjudication engine combining Confidence-Weighted Byzantine Fault Tolerance (CW-BFT) with cryptographic causal provenance. The proposed system replaces unweighted consensus with calibrated confidence probes and latent entropy vectors, verifies intermediate justifications against a structural causal model of governing rules, and notarizes all agent actions, tool payloads, and voting rounds into an immutable Merkle-DAG ledger. By doing so, the framework guarantees bounded error degradation under up to 40% adversarial prompt-injection rates and achieves deterministic fault recovery without re-executing entire pipelines. Solving this problem is critical for high-stakes enterprise AI adoption, transforming brittle probabilistic agent discussions into a resilient, tamper-proof, and regulatory-compliant autonomous decision system.

Problem Statement:
Autonomous multi-agent LLM architectures deployed in high-stakes adjudication lack mechanism-level fault tolerance and immutable auditability. A single hallucinating or compromised agent silently corrupts downstream reasoning nodes, while standard consensus approaches (such as multi-agent debate or unweighted majority voting) fail to bound adversarial errors or verify the causal faithfulness of decisions. Additionally, probabilistic text transcripts cannot provide the cryptographic tamper-evidence, deterministic recovery, and regulatory compliance required for mission-critical enterprise deployment.
