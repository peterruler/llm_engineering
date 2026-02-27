"""Interactive website summarizer using day1 prompt functions with day2 Ollama client."""

from openai import OpenAI

from scraper import fetch_website_contents

OLLAMA_BASE_URL = "http://localhost:11434/v1"
MODEL = "llama3.2"

system_prompt = """
You are a snarky assistant that analyzes the contents of a website,
and provides a short, snarky, humorous summary, ignoring text that might be navigation related.
Respond in markdown. Do not wrap the markdown in a code block - respond just with the markdown.
"""

user_prompt_prefix = """
Here are the contents of a website.
Provide a short summary of this website.
If it includes news or announcements, then summarize these too.

"""


def messages_for(website):
    """Create the message payload for the model."""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt_prefix + website},
    ]


def summarize(url, client):
    """Fetch website content and generate a summary."""
    website = fetch_website_contents(url)
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages_for(website),
    )
    return response.choices[0].message.content


def main():
    """Interactive prompt for webpage URLs."""
    ollama = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")

    print("Website summarizer (Ollama).")
    print("Enter a webpage URL to summarize, or type 'quit' to exit.\n")

    while True:
        url = input("Webpage URL: ").strip()
        if not url:
            continue
        if url.lower() in {"q", "quit", "exit"}:
            print("Goodbye.")
            break

        try:
            print("\nSummarizing...\n")
            print(summarize(url, ollama))
            print("\n" + "-" * 80 + "\n")
        except Exception as exc:
            print(f"Failed to summarize {url}: {exc}\n")


if __name__ == "__main__":
    main()
