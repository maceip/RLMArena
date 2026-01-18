# Open-Coder

This directory contains the **RL Training Set** and the **Test Set** (categorized by subtask) for the Open-Coder domain.

## Overview

In the Open-Coder domain, the agent is required to assist users in software development tasks including code generation, debugging, refactoring, testing, and repository-level modifications. This domain emphasizes multi-file reasoning, tool use (LSP, linters, test runners), and iterative development workflows.

## Dataset

### Statistics

| Split           | Samples | Description                                    |
| :-------------- | :------ | :--------------------------------------------- |
| **RL Training** | **TBD** | Used for Reinforcement Learning (RL) training. |
| **Test**        | **TBD** | Contains subtask files for evaluation.         |

### Files

*   [`train.jsonl`](train.jsonl)
    *   Contains RL training samples for coding tasks.
*   [`test/`](test/)
    *   Contains test samples distributed across distinct subtasks:

| File Name                                           | Samples | Task Type   | Description                                      |
| :-------------------------------------------------- | :------ | :---------- | :----------------------------------------------- |
| [`code_generation.jsonl`](test/code_generation.jsonl) | **TBD** | Generation  | Implement features from specifications.          |
| [`bug_fixing.jsonl`](test/bug_fixing.jsonl)         | **TBD** | Debug       | Diagnose and fix bugs from issue descriptions.   |
| [`refactoring.jsonl`](test/refactoring.jsonl)       | **TBD** | Refactor    | Improve code structure without changing behavior.|
| [`testing.jsonl`](test/testing.jsonl)               | **TBD** | Testing     | Write comprehensive test suites.                 |
| [`repo_level.jsonl`](test/repo_level.jsonl)         | **TBD** | Multi-file  | Cross-file changes requiring repo understanding. |

## Tasks

The samples in these files cover the following categories:

1.  **Code Generation:** Implementing new features, functions, or modules from natural language specifications.
2.  **Bug Fixing:** Diagnosing runtime errors, logic bugs, and edge cases from issue descriptions or failing tests.
3.  **Refactoring:** Improving code quality, extracting abstractions, reducing duplication while preserving behavior.
4.  **Test Writing:** Creating unit tests, integration tests, and property-based tests for existing code.
5.  **Repository-Level Changes:** Multi-file modifications requiring understanding of project structure and dependencies.

## Tooling

Agents in this bracket have access to:
- File read/write operations
- `git` (version control)
- Language servers (LSP) for code intelligence
- Linters and formatters (eslint, ruff, prettier, etc.)
- Test runners (pytest, jest, cargo test, go test)
- Build tools (make, npm, cargo, go build)
- Package managers (pip, npm, cargo)

## Verification

Tasks are verified by:
- Test suite pass/fail (primary signal)
- Linter/type checker success
- Build success
- Diff size and relevance heuristics
- Sandboxed execution in Firecracker microVMs

## License
The dataset files listed in this directory are licensed under the [Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/).
