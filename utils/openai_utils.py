import os
from openai import OpenAI

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")


def openai_completion(client, model_id, conversation_log, temperature=0):

    response = client.chat.completions.create(
        model=model_id, messages=conversation_log, temperature=temperature
    )

    conversation_log.append(
        {
            "role": response.choices[0].message.role,
            "content": response.choices[0].message.content.strip(),
        }
    )
    return conversation_log
