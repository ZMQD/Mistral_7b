from huggingface_hub import InferenceClient

client = InferenceClient("mistralai/Mistral-7B-Instruct-v0.1")


def format_prompt(message, history):
    prompt = "<s>"
    for user_prompt, bot_response in history:
        prompt += f"[INST] {user_prompt} [/INST]"
        prompt += f" {bot_response}</s> "
    prompt += f"[INST] {message} [/INST]"
    return prompt


def generate_full_text(
    prompt, history, temperature=0.9, max_new_tokens=1024, top_p=0.95, repetition_penalty=1.0,
):
    temperature = float(temperature)
    if temperature < 1e-2:
        temperature = 1e-2
    top_p = float(top_p)

    generate_kwargs = dict(
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        do_sample=True,
        seed=42,
    )

    formatted_prompt = format_prompt(prompt, history)

    stream = client.text_generation(
        formatted_prompt,
        **generate_kwargs,
        stream=True,
        details=True,
        return_full_text=True,  # Request full text
    )

    full_text = ""
    for response in stream:
        full_text += response.token.text

    return full_text

history = [("What's your name?", "I am a language model."), ("How are you?", "I'm doing well.")]

prompt = input("Type Your Prompt Here : ")
generated_text = generate_full_text(prompt, history)

print(generated_text)
