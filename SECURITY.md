# Security Policy

## Supported versions

This is a personal research and training project. Only the latest
commit on `main` receives any attention; there are no release
branches or backports.

## Scope

In scope:
- Vulnerabilities in the code in this repository (training pipeline,
  CLI, data handling) that could crash, corrupt data, or execute
  unintended code.
- Malicious input handling in the URL path (e.g. crafted strings that
  cause unbounded memory use in `encode_url` or the loaders).

Out of scope:
- Model accuracy. See "Model limitations" below — false negatives are
  not a security vulnerability of the tool.
- Datasets fetched from third-party feeds (Tranco, PhishTank,
  OpenPhish). Report issues with those feeds upstream.

## Reporting a vulnerability

Please open a private security advisory on this repository
(GitHub: Security tab -> "Report a vulnerability") rather than a
public issue. Include reproduction steps and, where possible, the
exact input that triggers the problem.

Response expectations: best-effort, hobbyist timescale.

## Model limitations (read before deploying)

Sinker's models are classifiers trained on finite, versioned
datasets of URLs. They WILL produce both false positives and false
negatives, and phishing techniques drift constantly.

- Do NOT treat a score below threshold as proof a URL is safe.
- Do NOT wire this into an automated blocking pipeline without a
  human-reviewed fallback.
- Scores are probabilities conditioned on training-data similarity,
  not statements about intent.

The intended use is research, education, and triage support.

