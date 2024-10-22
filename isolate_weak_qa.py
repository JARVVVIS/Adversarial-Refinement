import os
import argparse

import re
import tqdm
import pandas as pd
import logging
from openai import OpenAI
import random
from datasets import load_dataset
from datetime import datetime
import numpy as np

from utils import (
    RateLimiter,
    get_groq_prompt,
    get_groq_response_with_retry,
    eval_response,
)

# Set up logging
os.makedirs("logs/qa_isolation", exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
logging.basicConfig(
    filename=f"logs/qa_isolation/{timestamp}.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


# Ensure that the rotated row is well-formed
def test_rot_row(rot_row):
    assert rot_row["answer_key"] in rot_row["choices"], "Answer key not in choices"
    assert (
        rot_row["answer_key"] == rot_row["choices"][rot_row["answer_key_position"]]
    ), "Answer key position mismatch"
    assert rot_row["answer_key_position"] in [
        0,
        1,
        2,
        3,
        4,
    ], "Invalid answer key position"
    assert len(rot_row["choices"]) == 5, "Incorrect number of choices"


# Rotate the answer key by k positions; important to adjust for chance performance.
def rotate_answer_key(row, k=1):
    row_return = row.copy()

    original_choices = list(row_return["choices"])
    original_position = row_return["answer_key_position"]
    original_answer_key = row_return["answer_key"]

    choices = original_choices.copy()
    current_position = choices.index(row["answer_key"])
    assert current_position == original_position

    new_position = (current_position + k) % len(choices)
    rotated_choices = choices[-k:] + choices[:-k]

    row_return["choices"] = rotated_choices
    row_return["answer_key_position"] = new_position

    # Assert tests
    assert (
        new_position != original_position or k % len(choices) == 0
    ), "New position should be different unless rotated by a multiple of list length"
    assert sorted(rotated_choices) == sorted(
        original_choices
    ), "Elements in the list should remain the same"
    assert (
        row_return["answer_key"] == original_answer_key
    ), f"Answer key should remain unchanged, but {row_return['answer_key']} != {original_answer_key}"
    assert (
        rotated_choices[new_position] == original_answer_key
    ), "New position should contain the original answer key"

    return row_return


# Extract the LLaMA response from the raw response
def extract_llama_response(
    text,
):  ## we need to do this since the raw response has rationale as well
    text = text.replace("*", "")
    pattern = r"^Answer:\s*([A-Z])\s*Rationale:\s*(.*)$"
    match = re.search(pattern, text, re.DOTALL)

    if match:
        answer = match.group(1)
        rationale = match.group(2).strip()
        return (answer, rationale)
    else:
        return None


# Get the LLaMA response for a given QA
def get_llama_response_with_row(row, rate_limiter):
    ans_key_map = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E"}

    llama_prompt = get_groq_prompt(row)
    try:
        groq_response = get_groq_response_with_retry(llama_prompt, rate_limiter)
    except Exception as e:
        logging.error(f"Error getting LLaMA response: {e}")
        return None, None, None, None

    llama_response_tuple = extract_llama_response(groq_response)
    if llama_response_tuple is None:
        logging.error(f"Failed to extract LLaMA response from: {groq_response}")
        return None, None, None, None

    llama_response, llama_rationale = llama_response_tuple
    llama_response_tbi = (
        f"Student's Response: {llama_response}\nStudent's Rationale: {llama_rationale}"
    )

    ans_key_opt = ans_key_map[row["answer_key_position"]]
    is_corr = eval_response(llama_response, ans_key_opt, row["answer_key"])

    return (llama_response, llama_rationale, llama_response_tbi, is_corr)


# Main function to isolate a single row
def isolate_single_row(row, rate_limiter):
    rotations = [0, 1, 2, 3, 4]

    rot_responses, rot_rationale, rot_correct = (
        [],
        [],
        [],
    )

    for k in rotations:
        if k > 0:
            rot_row = rotate_answer_key(row, k)
        else:
            rot_row = row.copy()

        try:
            test_rot_row(rot_row)
        except AssertionError as e:
            logging.error(f"Modified row failed validation: {e}")
            continue

        llama_response, llama_rationale, llama_response_tbi, is_llama_corr = (
            get_llama_response_with_row(rot_row, rate_limiter)
        )

        if is_llama_corr is None:
            logging.error(
                f"Failed to get LLaMA response for modified row (rotation {k})"
            )
            continue

        rot_responses.append(llama_response)
        rot_rationale.append(llama_rationale)
        rot_correct.append(is_llama_corr)

    needs_fix = (
        np.mean(rot_correct) > 0.5
    )  ## True if majority still correct, False if majority incorrect. If False, we get out of the rotation check.

    return row, needs_fix, rot_responses, rot_rationale, rot_correct


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Fix train or test split",
        choices=["train", "test"],
    )
    parser.add_argument(
        "--ckpt_freq", type=int, default=100, help="Checkpoint save frequency"
    )
    args = parser.parse_args()

    checkpoint_file = f"cinepile_df_{args.split}_debug_ckpt.pkl"

    if os.path.exists(checkpoint_file):
        logging.info(f"Loading checkpoint from {checkpoint_file}")
        cinepile_debug_df = pd.read_pickle(checkpoint_file)
    else:
        logging.info("No checkpoint found. Starting from scratch.")
        cinepile_debug_df = load_dataset(
            "tomg-group-umd/cinepile", split=args.split
        ).to_pandas()

        cols_to_save = (
            ["needs_fix"]
            + [f"llama_correct{iso+1}" for iso in range(5)]
            + [f"llama_response{iso+1}" for iso in range(5)]
        )
        for col in cols_to_save:
            cinepile_debug_df[f"{col}"] = None

    total_questions = 300000 if args.split == "train" else 5000
    questions_to_check = (
        cinepile_debug_df["needs_fix"].isna().sum()
    )  ## For checkpointing logic

    logging.info(f"Total questions: {total_questions}")
    logging.info(
        f"Questions to check: {questions_to_check} [{(questions_to_check/total_questions)*100:.2f}% of dataset]"
    )

    rate_limiter = RateLimiter(
        requests_per_minute=100, requests_per_day=14400
    )  ## NOTE: Rate limitation for Groq API; change as needed.

    rows_to_process = cinepile_debug_df[cinepile_debug_df["needs_fix"].isna()].index

    count = 0

    for idx in tqdm.tqdm(rows_to_process, total=len(rows_to_process)):
        row = cinepile_debug_df.loc[idx]

        logging.info(f"Processing row {idx}")

        row, needs_fix, rot_responses, rot_rationale, rot_correct = isolate_single_row(
            row, rate_limiter
        )

        cinepile_debug_df.at[idx, "needs_fix"] = needs_fix
        logging.info(f"\tNeeds fix: {needs_fix}")
        for iso, (resp, cor) in enumerate(zip(rot_responses, rot_correct)):
            cinepile_debug_df.at[idx, f"llama_correct{iso+1}"] = cor
            cinepile_debug_df.at[idx, f"llama_response{iso+1}"] = resp
            logging.info(f"\tResponse {iso+1}: {resp} [Correct: {cor}]")

        count += 1
        if count % args.ckpt_freq == 0:
            logging.info(f"Saving checkpoint after {count} rows")
            cinepile_debug_df.to_pickle(checkpoint_file)

        logging.info("-" * 50)

    cinepile_debug_df.to_csv(f"cinepile_df_{args.split}_debug.csv")
    cinepile_debug_df.to_pickle(f"cinepile_df_{args.split}_debug.pkl")
    logging.info("Isolation process completed")


if __name__ == "__main__":
    main()
