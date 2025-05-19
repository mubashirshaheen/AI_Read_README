import os

from langchain_ollama import OllamaLLM
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Config
README_DIR = "readmes"
# Load all available README files
readme_files = [f for f in os.listdir(README_DIR) if f.lower().endswith(".md")]
if not readme_files:
    print("No README files found in the 'readmes/' folder.")
    exit(1)

read_me_files = {"github-download.md": "Github Download"}

# Show options to user
print("Available README files:")
for idx, filename in enumerate(readme_files, start=1):
    print(f"{idx}. {read_me_files[filename]}")

# Ask for user selection
while True:
    try:
        selection = int(input("Select a README by number: "))
        if 1 <= selection <= len(readme_files):
            break
        else:
            print("Invalid number. Try again.")
    except ValueError:
        print("Please enter a number.")

selected_file = readme_files[selection - 1]
readme_path = os.path.join(README_DIR, selected_file)

# Load model with streaming
llm = OllamaLLM(
    model="llama3",
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()]
)

# Load github-download.md once
with open(readme_path, 'r') as file:
    content = file.read()

# Readme Prompt
summary_prompt = (
    "You are an expert technical writer. Read the github-download.md content below and generate a professional, step-by-step guide "
    "that explains how to use this project in very simple and clear language. Use bullet points and numbered steps. Avoid jargon.\n\n"
    "README:\n"
    "```markdown\n" + content + "\n```\n"
    "Provide the explanation below in 5 steps. 1. What is Tool? 2. System Requirements 3. Provide link  From where to download requirements and other stuff? "
                                "4. How to install? 5. Warning and Errors that can come?:\n"
)

print(f"\n🔍 Summarising {readme_files[selection - 1]}...\n")
llm.invoke(summary_prompt)

# Base system prompt
system_prompt = (
    "You are an assistant that answers questions about a github-download.md file. "
    "You must only answer questions related to the content of the README below. "
    "If a question is outside the scope of the README, just say you don't have enough information.\n\n"
    "README content (between triple backticks):\n"
    "```\n" + content + "\n```\n"
    "When the user asks a question, answer clearly using bullet points if multiple items are mentioned."
)

# Prompt loop
while True:
    user_prompt = input("\nAsk a question about the README (or 'exit'): ").strip()
    if user_prompt.lower() in ("exit", "quit"):
        break

    full_prompt = system_prompt + "\nUser question: " + user_prompt + "\nAnswer:"
    print("\nAnswer:\n")
    llm.invoke(full_prompt)  # Output will stream live