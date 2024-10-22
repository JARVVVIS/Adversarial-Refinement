import re


def extract_complete_qa_pairs(dump):
    """
    Extracts complete question, choices, and answer key text units from the given text dump.
    This approach ensures each question is paired with its correct set of choices and answer key,
    accommodating variations in formatting.

    Parameters:
    - dump (str): The text dump containing questions, choices, and answer keys.

    Returns:
    - List of tuples containing the question, choices (as a list), and answer key text for each successfully extracted QA pair.
    """
    dump = dump.replace("*", "")  # Remove asterisks used for bold formatting

    # Updated pattern to match the specific structure of the text dump more closely
    complete_qa_pattern = re.compile(
        r"Question:\s*([^\n]+)\s*"
        r"- A\)\s*([^\n]+)\s*- B\)\s*([^\n]+)\s*- C\)\s*([^\n]+)\s*- D\)\s*([^\n]+)\s*- E\)\s*([^\n]+)\s*"
        r"(?:Correct )?Answer:\s*([A-E])",
        re.DOTALL | re.IGNORECASE,
    )

    # Search for complete Q&A blocks
    complete_qa_matches = complete_qa_pattern.findall(dump)

    # Process each match to format the output
    output = []
    for match in complete_qa_matches:
        question = match[0].strip()
        choices = [choice.strip() for choice in match[1:6]]
        correct_answer_marker = match[6].strip().upper()
        correct_answer_index = ord(correct_answer_marker) - ord("A")
        correct_answer_text = choices[correct_answer_index]

        output.append((question, choices, correct_answer_marker, correct_answer_text))

    return output
