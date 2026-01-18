# Open-CloudInfra

This directory contains the **RL Training Set** and the **Test Set** for the Open-CloudInfra domain.

## Overview

In the Open-CloudInfra domain, the agent is required to assist users in provisioning, configuring, and managing cloud infrastructure across major providers (GCP, AWS, Azure). This domain focuses on Infrastructure-as-Code (IaC), Terraform/Pulumi workflows, Kubernetes orchestration, and multi-cloud resource management.

## Dataset

### Statistics

| Split           | Samples | Description                                    |
| :-------------- | :------ | :--------------------------------------------- |
| **RL Training** | **TBD** | Used for Reinforcement Learning (RL) training. |
| **Test**        | **TBD** | High-quality benchmark for evaluation.         |

### Files

*   [`train.jsonl`](train.jsonl)
    *   Contains RL training samples for cloud infrastructure tasks.
*   [`test.jsonl`](test.jsonl)
    *   Contains test samples for leaderboard-style evaluation.

## Tasks

The samples in these files cover the following categories:

1.  **Resource Provisioning:** Creating and configuring cloud resources (VMs, storage, networking, databases) via IaC.
2.  **Kubernetes Operations:** Deploying, scaling, and managing containerized workloads on GKE/EKS/AKS.
3.  **Multi-Cloud Orchestration:** Coordinating resources across multiple cloud providers with unified tooling.
4.  **Cost Optimization:** Analyzing and optimizing cloud spend, right-sizing resources, spot/preemptible instances.
5.  **Security & Compliance:** IAM policies, network security groups, secrets management, compliance auditing.
6.  **Monitoring & Observability:** Setting up metrics, alerts, logging pipelines, and distributed tracing.

## Tooling

Agents in this bracket have access to:
- `gcloud` CLI (Google Cloud)
- `aws` CLI (Amazon Web Services)
- `az` CLI (Microsoft Azure)
- `terraform` / `pulumi` (Infrastructure as Code)
- `kubectl` / `helm` (Kubernetes)
- `docker` (Container operations)

## Verification

Tasks are verified by:
- Terraform plan/apply success
- Resource existence checks via cloud APIs
- Kubernetes resource status validation
- Cost estimate validation against constraints

## License
The dataset files listed in this directory are licensed under the [Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/).
