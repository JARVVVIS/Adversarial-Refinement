# Adversarial Refinement Pipeline

This repository contains the code for an adversarial refinement pipeline designed to modify a visual-QA dataset, making the questions and/or answer choices unanswerable by a model based solely on the provided question and answer options. The process uses adversarial techniques to iteratively refine QA pairs until a large-scale language model, such as LLaMa 3.1 70b (also referred to as **Deaf-Blind LLM**), is unable to answer correctly based only on the question and answer choices.

## Overview

The pipeline optimizes questions and answer choices by leveraging the following modules:

- **Question Generation Module**: Powered by GPT-4, this module is responsible for generating modified QA pairs.
- **Deaf-Blind LLM**: Uses LLaMa 3.1 70b to evaluate the modified QA pairs. The goal is to rewrite the QA until the Deaf-Blind QA system fails to answer correctly.

## Workflow

1. **Isolate Weak Questions**: The process begins by isolating the "weak" questions from the dataset, i.e., those that might still be answerable from the textual data alone, after adjusting for chance performance.
   
   To isolate weak questions, run the following script:
   ```bash
   python isolate_weak_qa.py
   ```

2. **Adversarial Refinement**: After isolating the weak questions, you can run the adversarial refinement process to rewrite the questions and answer choices.

    To begin adversarial refinement, run:
    ```bash
    python rewrite_qa.py
    ```