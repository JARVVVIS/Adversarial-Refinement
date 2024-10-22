import groq
from requests.exceptions import RequestException
import random
import time
import os

groq_client = groq.Client(api_key=os.getenv("GROQ_API_KEY"))


def get_groq_response(
    prompt,
    model_id="llama-3.1-70b-versatile",
    temperature=0.1,
    top_p=1,
    max_tokens=1024,
    stop=None,
):
    chat_completion = groq_client.chat.completions.create(
        messages=[
            {"role": "system", "content": "you are a helpful assistant."},
            {
                "role": "user",
                "content": prompt,
            },
        ],
        model=model_id,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        stop=stop,
        stream=False,
    )

    response = chat_completion.choices[0].message.content
    return response


def get_groq_response_with_retry(
    prompt, rate_limiter, max_retries=10, model_id="llama-3.1-70b-versatile"
):
    for attempt in range(max_retries):
        try:
            rate_limiter.wait()
            response = get_groq_response(prompt, model_id=model_id)
            return response
        except (RequestException, groq.InternalServerError) as e:
            if (
                isinstance(e, RequestException)
                and e.response is not None
                and e.response.status_code == 429
            ):
                error_type = "Rate limit"
            elif isinstance(e, groq.InternalServerError) and str(e).startswith(
                "Error code: 503"
            ):
                error_type = "Service Unavailable"
            else:
                raise

            wait_time = min(
                2**attempt + random.uniform(0, 1), 60
            )  # Max wait time of 60 seconds
            print(
                f"{error_type} error encountered. Waiting for {wait_time:.2f} seconds before retrying..."
            )
            time.sleep(wait_time)

    raise Exception("Max retries reached. Unable to get response from Groq API.")


class RateLimiter:
    def __init__(self, requests_per_minute, requests_per_day):
        self.requests_per_minute = requests_per_minute
        self.requests_per_day = requests_per_day
        self.minute_counter = 0
        self.day_counter = 0
        self.last_reset_time = time.time()
        self.day_start_time = time.time()

    def wait(self):
        current_time = time.time()

        # Reset minute counter if a minute has passed
        if current_time - self.last_reset_time >= 60:
            self.minute_counter = 0
            self.last_reset_time = current_time

        # Reset day counter if a day has passed
        if current_time - self.day_start_time >= 86400:  # 86400 seconds in a day
            self.day_counter = 0
            self.day_start_time = current_time

        # Check if we've hit the rate limits
        if (
            self.minute_counter >= self.requests_per_minute
            or self.day_counter >= self.requests_per_day
        ):
            sleep_time = max(60 - (current_time - self.last_reset_time), 0)
            time.sleep(sleep_time)
            self.minute_counter = 0
            self.last_reset_time = time.time()

        self.minute_counter += 1
        self.day_counter += 1


basic_groq_prompt = """USER: <prompt>Please answer the question that follows. The question will have five possible answers labeled A, B, C, D, and E, please try to provide the most probable answer in your opinion. Your output should be just one of A,B,C,D,E and the rationale of why you thinking the answer is that by just looking at the question.

**Output Format:**
    **Answer:** <Option_key>
    **Rationale:** <Your_reasoning>

Question: {question}

Note: Follow the output format strictly. Only answer with the option key (A, B, C, D, E) and the rationale of why you thinking the answer is that by just looking at the question
ASSISTANT:"""


def format_question_and_options(question, options):
    """
    Formats a question and a list of options into a single string with options labeled A, B, C, etc.

    Parameters:
    - question (str): The question to be formatted.
    - options (list of str): The options for the question.

    Returns:
    - str: The formatted question and options.
    """
    formatted_string = f"{question}\n"
    option_labels = [
        chr(ord("A") + i) for i in range(len(options))
    ]  # Generate option labels dynamically

    for label, option in zip(option_labels, options):
        formatted_string += f"- {label}) {option}\n"

    return formatted_string


def get_groq_prompt(data, prefix=""):
    options = data[f"{prefix}choices"]
    formatted_question = format_question_and_options(data[f"{prefix}question"], options)
    prompt = basic_groq_prompt.format(question=formatted_question)
    return prompt
