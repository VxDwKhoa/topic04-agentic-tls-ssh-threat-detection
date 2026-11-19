# Topic 04: Agentic ML/DL for Metadata-Only TLS/SSH Threat Detection, Hunting & Response

Course: **NT219 / NT2205 — Cryptography and Applications**

## Core Principle
- **No Raw Layer-7 Application Payload**: Strictly operates on L3/L4 flow metadata and L5/L6 cryptographic/session handshake metadata (TLS 1.3, X.509, SSH).
- **Reasoning Chain**: System Analysis -> Assets -> Threat Model -> Security Requirements -> Security Architecture -> Cryptographic Requirements -> Mechanisms -> Implementation -> Security Testing -> Evaluation

## Repository Structure
`	ext
project-root/
├── data/
│   ├── public_loaders/       # NF-UQ, IoT-23, CESNET, CTU...
│   ├── lab_metadata/         # JSON/Parquet only; no L7 raw payload
│   ├── manifests/            # scenario ground truth
│   └── schemas/              # allowlist + privacy validator
├── sensors/                  # Zeek/Suricata metadata configs
├── src/
│   ├── features/             # metadata extraction & fingerprinting
│   ├── models/               # tabular/anomaly/sequence/graph
│   ├── hunting/              # evidence graph + ATT&CK retrieval
│   ├── agent/                # policy guard, planner, verifier
│   └── response/             # simulator + bounded lab adapters
├── experiments/              # configs, splits, ablations, logs
├── tests/                    # privacy, policy, model, response tests
├── docker/                   # container setups
└── report/                   # project reports & presentation
`
