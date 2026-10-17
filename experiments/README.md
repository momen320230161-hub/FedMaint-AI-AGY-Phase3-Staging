# Experiments

**Scaffold only:** no models have been trained, and there are no benchmark results yet.

For each future experiment, record:

1. Dataset and manifest revision, task/target and prediction horizon.
2. Per-client data partition and held-out factory/domain definition.
3. Local/centralized/FedAvg/FedProx and privacy settings.
4. Git commit, environment, random seeds, config path.
5. Prediction metrics, per-client and held-out results, communication overhead, and failure cases.
6. Explanation grounding sources and faithfulness checks when relevant.

Store lightweight **summaries/configs** in Git. Avoid committing private records, large model weights or generated sensitive content. Follow the [experiment protocol](../docs/EXPERIMENT_PROTOCOL.md).
