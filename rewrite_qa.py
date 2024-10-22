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
    format_question_and_options,
    openai_completion,
    OPENAI_API_KEY,
    extract_complete_qa_pairs,
    RateLimiter,
    get_groq_prompt,
    get_groq_response_with_retry,
    eval_response,
)

# Set up logging
os.makedirs("logs/qa_debugging", exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
logging.basicConfig(
    filename=f"logs/qa_debugging/{timestamp}.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


# Get the debugging prompt with updated movie_scene information, original QA, and answer key
def get_debug_prompt(row, qa_prompt_path="prompt_fix.txt"):
    with open(qa_prompt_path, "r") as file:
        qa_prompt = file.read()

    qa = format_question_and_options(row["question"], row["choices"])
    qa_prompt = qa_prompt.replace("{MOVIE_SCENE_TS}", row["movie_scene"])
    qa_prompt = qa_prompt.replace("{ORIGINAL_QA}", qa)
    qa_prompt = qa_prompt.replace("{ORIGINAL_ANS}", row["answer_key"])

    return qa_prompt


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


# Ensure that a refined QA pair is well-formed
def test_extracted_qa(extracted_qa):
    alpha_to_pos = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
    assert len(extracted_qa) == 4, "Extracted QA should have 4 elements"
    assert extracted_qa[2] in ["A", "B", "C", "D", "E"], "Invalid answer marker"
    assert extracted_qa[3] in extracted_qa[1], "Answer not in choices"
    assert (
        extracted_qa[3] == extracted_qa[1][alpha_to_pos[extracted_qa[2]]]
    ), "Answer key marker mismatch"


# Ensure that the updated row is well-formed
def test_mod_row(mod_row):
    assert mod_row["answer_key"] in mod_row["choices"], "Answer key not in choices"
    assert (
        mod_row["answer_key"] == mod_row["choices"][mod_row["answer_key_position"]]
    ), "Answer key position mismatch"
    assert mod_row["answer_key_position"] in [
        0,
        1,
        2,
        3,
        4,
    ], "Invalid answer key position"
    assert len(mod_row["choices"]) == 5, "Incorrect number of choices"


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


# Main function to fix a single row
def fix_single_row(row, openai_client, openai_model, openai_temp, rate_limiter):
    alpha_to_pos = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}

    logging.info(f"Starting fix for row: {row['question']}")

    llama_response, llama_rationale, llama_response_tbi, is_llama_corr = (
        get_llama_response_with_row(row, rate_limiter)
    )

    if is_llama_corr is None:
        logging.error("Failed to get initial LLaMA response")
        return row, False

    assert llama_response in [
        "A",
        "B",
        "C",
        "D",
        "E",
    ], f"Invalid LLaMA response: {llama_response}"

    needs_fix = is_llama_corr
    curr_tries = 0

    while needs_fix and curr_tries <= 5:  ## NOTE: 5 max attempts to fix the question
        logging.info(f"Attempt {curr_tries + 1} to fix the question")

        fix_prompt = get_debug_prompt(row)
        try:
            fix_prompt = fix_prompt.replace(
                "STUDENT_RATIONALE", llama_response_tbi
            )  ## NOTE: Poor name, this atually contains both the student answer and rationale
        except:
            logging.error(
                "Failed to replace student rationale in prompt. Fall back to answer key"
            )  ## This is a fallback in case the student rationale is not available
            fix_prompt = fix_prompt.replace(
                "STUDENT_RATIONALE", f"Answer Picked: {row['answer_key']}"
            )

        prompt_conv = {"role": "user", "content": fix_prompt}
        conversations = [prompt_conv]
        conversations = openai_completion(
            client=openai_client,
            model_id=openai_model,
            conversation_log=conversations,
            temperature=openai_temp,
        )
        generated_qa = conversations[-1]["content"].strip()

        try:
            extracted_qa = extract_complete_qa_pairs(generated_qa.strip("```"))
            if not extracted_qa:
                logging.error(
                    f"No QA pairs extracted from generated content: {generated_qa}"
                )
                curr_tries += 1
                continue

            extracted_qa = extracted_qa[0]
            test_extracted_qa(extracted_qa)
        except AssertionError as e:
            logging.error(f"Extracted QA failed validation: {e}")
            curr_tries += 1
            continue

        mod_row = row.copy()
        mod_row["question"] = extracted_qa[0]
        mod_row["choices"] = extracted_qa[1]
        mod_row["answer_key_position"] = alpha_to_pos[extracted_qa[2]]
        mod_row["answer_key"] = extracted_qa[3]

        rotations = [0, 1, 2, 3, 4]  # 0 for original, then 4 rotations
        rot_responses, rot_rationale, rot_correct = (
            [],
            [],
            [],
        )

        for k in rotations:
            if k > 0:
                rot_row = rotate_answer_key(mod_row, k)
            else:
                rot_row = mod_row.copy()

            try:
                test_mod_row(rot_row)
            except AssertionError as e:
                logging.error(f"Modified row failed validation: {e}")
                curr_tries += 1
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
        curr_tries += 1

        logging.info(
            f"Current QA:\n{format_question_and_options(row['question'], row['choices'])}"
        )
        logging.info(
            f"Modified QA:\n{format_question_and_options(mod_row['question'], mod_row['choices'])}"
        )
        logging.info(f"Modified Answer Key: {mod_row['answer_key']}")
        logging.info(f"LLaMA Response: {rot_responses}")
        logging.info(f"LLaMA Rationale: {rot_rationale}")
        logging.info(f"LLaMA Correct: {rot_correct}")

        row = mod_row.copy()

    return row, not needs_fix


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

    checkpoint_file = f"cinepile_df_{args.split}_fixed_ckpt.pkl"
    cols_to_save = [
        "question",
        "choices",
        "answer_key_position",
        "answer_key",
        "is_fixed",
    ]

    if os.path.exists(checkpoint_file):
        logging.info(f"Loading checkpoint from {checkpoint_file}")
        cinepile_debug_df = pd.read_pickle(checkpoint_file)
    else:
        logging.info("No checkpoint found. Starting from scratch.")
        cinepile_debug_df = pd.read_pickle(
            f"cinepile_df_{args.split}_debug_ckpt.pkl"
        )  ## NOTE: Isolated weak questions

        cinepile_debug_df = cinepile_debug_df[
            cinepile_debug_df["needs_fix"] == True
        ]  ## Only save the rows that need fixing

        # Initialize new columns
        for col in cols_to_save:
            if col == "is_fixed":
                cinepile_debug_df[f"{col}"] = None
            else:
                cinepile_debug_df[f"mod_{col}"] = None

    total_questions = 300000 if args.split == "train" else 5000
    questions_to_fix = (
        cinepile_debug_df["is_fixed"].isna().sum()
    )  ## For checkpointing logic

    logging.info(f"Total questions: {total_questions}")
    logging.info(
        f"Questions to fix: {questions_to_fix} [{(questions_to_fix/total_questions)*100:.2f}% of dataset]"
    )

    openai_model = "gpt-4o"
    openai_temp = 0.8
    openai_client = OpenAI(api_key=OPENAI_API_KEY)

    rate_limiter = RateLimiter(
        requests_per_minute=100, requests_per_day=14400
    )  ## NOTE: Rate limitation for Groq API; change as needed.

    rows_to_process = cinepile_debug_df[cinepile_debug_df["is_fixed"].isna()].index

    count = 0

    for idx in tqdm.tqdm(rows_to_process, total=len(rows_to_process)):
        row = cinepile_debug_df.loc[idx]

        fixed_row, is_fixed = fix_single_row(
            row, openai_client, openai_model, openai_temp, rate_limiter
        )

        logging.info(
            f"---------------- Row: {idx} - Fixed: {is_fixed} ----------------"
        )

        for col in cols_to_save:
            if col == "is_fixed":
                cinepile_debug_df.at[idx, f"{col}"] = is_fixed
            else:
                cinepile_debug_df.at[idx, f"mod_{col}"] = fixed_row[col]

        count += 1
        if count % args.ckpt_freq == 0:
            logging.info(f"Saving checkpoint after {count} rows")
            cinepile_debug_df.to_pickle(checkpoint_file)

    cinepile_debug_df.to_csv(f"cinepile_df_{args.split}_fixed.csv")
    cinepile_debug_df.to_pickle(f"cinepile_df_{args.split}_fixed.pkl")
    logging.info("Debugging process completed")


if __name__ == "__main__":
    main()
