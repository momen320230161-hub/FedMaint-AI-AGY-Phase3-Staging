# Proposed System Architecture

> This is a **design document**, not an implemented system. The project proposal requires federated training, privacy protection, a client-local maintenance knowledge graph, and LLM explanations grounded in facts.

## Current Phase 1 detailed design

The exact proposed ingestion, training, and local inference paths now have separate specifications:

- [DATA_FLOW.md](DATA_FLOW.md): source tables, observed Azure model distribution, candidate 3-client + held-out-model protocol, leakage rules, graph-to-LLM evidence flow.
- [MODEL_DESIGN.md](MODEL_DESIGN.md): candidate GRU backbone, independent classifier/RUL heads, matched FedAvg/FedProx/local/centralized comparisons.
- [DECISIONS.md](DECISIONS.md): which values are still **candidates** rather than implemented or team-approved decisions.

## Components and ownership

| Component | Planned package | Receives | Produces |
| --- | --- | --- | --- |
| Data loaders / validation / splits | `src/fedmaint/data/` | Azure tables or NASA engine sequences | Validated local features, labels, split metadata |
| Predictive models | `src/fedmaint/models/` | Local time-series windows | Failure risk / RUL and model parameters |
| Federated coordinator + clients | `src/fedmaint/federated/` | Selected client updates | Aggregated model parameters, round metrics |
| Privacy mechanism | `src/fedmaint/privacy/` | Updates or aggregation messages | Protected messages, privacy/overhead metrics |
| KG builder / retrieval | `src/fedmaint/knowledge_graph/` | Local machine, error, component, maintenance records | Provenance-tagged factual evidence |
| Explanation generator | `src/fedmaint/explainability/` | Prediction, relevant evidence | Grounded text, referenced facts, uncertainty |
| Evaluation | `src/fedmaint/evaluation/` | Predictions, targets, evidence, run metadata | Reproducible metrics and ablations |

## Proposed flow

1. Split the source dataset into **non-IID simulated factories**, identifying an entirely held-out factory before training.
2. Inside each participating client: validate data, fit **local-only** transforms and normalize using training observations only; form leakage-safe feature windows.
3. Train the same backbone locally; FL clients communicate **model updates only** to the coordinator. The server applies FedAvg/FedProx and the chosen privacy design. Ordinary FL alone does not guarantee privacy.
4. Return updated parameters to participating factories. The held-out factory participates **only in evaluation**.
5. For local inference, create a risk or RUL estimate from history available at prediction time.
6. Retrieve relevant facts from the **local** knowledge graph under a shared ontology. Do not send private graph records to the server by default.
7. Ask the LLM to explain the **prediction and evidence**, flag uncertainties, and refrain from inventing causal failure diagnoses or safety-critical actions.
8. Evaluate prediction accuracy, generalization, privacy–utility, communication, and faithfulness of explanations.

## Privacy and data boundaries

| Remains client-local by default | Federated coordinator (intended) |
| --- | --- |
| Raw telemetry, failure and maintenance records, machine identifiers | Aggregatable model updates (after defined privacy processing) |
| Fitted preprocessing statistics | FL round and communication metadata |
| Local graph facts and retrieved evidence | Aggregated model parameters |
| Client train/validation/test records | **No** pooled raw records except in a separately controlled centralized baseline experiment |

Privacy technique and deployment threat model are **not yet selected**. Model updates can leak information, and exposing KG facts to a remote LLM can also breach the privacy boundary. A local LLM or a specifically sanitized evidence channel must be decided before implementation.

## Distinct tasks, common abstractions

**Azure:** time-indexed telemetry and linked errors, maintenance and failures; candidate label `failure_within_24h`, provisional. Event joins use only timestamps at or before each prediction cutoff. Stronger contextual evidence exists for the KG, but error/component codes do **not** automatically establish causality.

**NASA:** run-to-failure engine sequences, operational settings and sensors; target RUL in cycles. The official per-test-engine RUL label describes the endpoint of its observed test trajectory. C-MAPSS lacks Azure-style explicit maintenance actions.

Shared code can include a dataset interface, modeling/training loops, metric collectors and orchestration. **Labels, preprocessing, and KG evidence contracts remain dataset-specific.**

## Planned prediction and explanation output contract

A future prediction result should contain at least:

- `dataset`, `client_id`, `asset_id`, `prediction_time_or_cycle`
- `task` (`failure_risk` or `rul`), `model_version`, `prediction`, `units`
- `evidence_ids` and `evidence_sources` from the local KG (if supported)
- `explanation`, `limitations`, `candidate_actions` only where justified

This describes a **proposed interface**, not a deployed or implemented JSON schema.

See [EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md) for leakage and evaluation rules.
