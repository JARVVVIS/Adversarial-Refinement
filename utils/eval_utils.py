import re


def normalize_string(input_string):
    """
    Extracts and returns the option number and option text from a given string.
    The option number is expected to be a single letter followed by an optional bracket and/or period.
    The option text is any text following the option number and its bracket/period.
    If the string does not contain an option number, the entire string is considered as the option text.
    """
    input_string = input_string.replace("*", "").strip()
    if re.match(r"^[A-E]$", input_string, re.IGNORECASE):
        return input_string.upper(), ""

    match = re.search(r"Answer:\s*([A-E])\)?\.?\s*(.*)", input_string, re.IGNORECASE)
    if match:
        option_number = match.group(1).upper()  # Normalize option number to uppercase
        option_text = match.group(2).strip()
        return option_number, option_text
    else:
        # If no option number is found after 'Answer:', consider it as no valid answer provided
        return None, input_string.strip()


def evaluate_semantic_similarity(
    response, answer_key_number, answer_key_text, normalize_fn
):
    """
    Evaluates whether the answer key and student response are semantically the same.
    Returns a score of 1 if they match, otherwise 0.
    """
    student_response_number, student_response_text = eval(normalize_fn)(response)

    # Compare option numbers and option texts (if available) to determine a match
    if answer_key_number and student_response_number:
        if answer_key_number == student_response_number:
            if answer_key_text and student_response_text:
                # If both strings have option texts, they must match as well
                return (
                    1 if answer_key_text.lower() == student_response_text.lower() else 0
                )
            # If only option numbers are provided or one string lacks option text, it's a match
            return 1
    elif answer_key_text.lower() == student_response_text.lower():
        # If no option numbers are present, but the option texts match, it's also considered a match
        return 1

    return 0


def eval_response(
    response, answer_key_number, answer_key_text, normalize_fn="normalize_string"
):
    return evaluate_semantic_similarity(
        response, answer_key_number, answer_key_text, normalize_fn
    )
