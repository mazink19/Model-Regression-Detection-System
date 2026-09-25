# Model Regression Detector

An automated evaluation and regression detection system for LLM-powered applications.

The system evaluates an LLM against a golden dataset, stores evaluation runs, compares a new run against a baseline, detects regressions, and generates an HTML evaluation report.

## Project Goal

LLM applications can change behavior when we modify:

- Prompts
- Models
- Model parameters
- Application logic

A change that improves one type of request can accidentally make another type worse.

This project provides an automated way to detect those regressions before they reach production.

## Architecture

```text
Golden Dataset
      |
      v
   Evaluator
      |
      v
 Evaluation Run
      |
      +----------------+
      |                |
      v                v
 baseline.json     current run
      |                |
      +-------+--------+
              |
              v
        Run Comparator
              |
      +-------+--------+
      |                |
      v                v
 Regressions      Improvements
      |
      v
Regression Status
      |
      v
   HTML Report
