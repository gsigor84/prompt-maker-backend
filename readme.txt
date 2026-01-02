PROMPT MASTER - TECHNICAL OVERVIEW
==================================

This project is a hybrid AI tool that serves as both a CLI application and a backend web API for generating high-quality LLM prompts.

1. ARCHITECTURE
---------------
The system is built on a modular "Agentic Pipeline" architecture.

[Logic Layer]
   |
   +-- promptforge_core.py: The brain. Contains the `PromptForgeOrchestrator` class.
       It manages a multi-step state machine (Collect -> Analyze -> Propose -> Draft -> Refine).
       It uses OpenAI to perform "thought steps" at each stage.

[Service Layer]
   |
   +-- prompt_quality.py: A standalone service that uses a "Critic" agent to score generated prompts
       on a 0-10 scale and provide actionable feedback.
   |
   +-- run_store.py: A simple JSONL persistence layer to log every run's state to `data/runs.jsonl`.

[Interface Layer]
   |
   +-- main.py: The unified entry point.
       - Uses `click` to handle CLI commands.
       - Uses `rich` for formatted terminal output.
       - Can launch the Pipeline directly (CLI mode) or start the Web Server.
   |
   +-- api.py: The FastAPI web application.
       - Exposes REST endpoints (e.g., `/api/prompt/full`) for the frontend.
       - Can be mounted by `main.py` via uvicorn.

2. THE PIPELINE (How it thinks)
--------------------------------
When you run a request, `PromptForgeOrchestrator.run()` executes these steps:

1. COLLECT: Captures your raw input.
2. ANALYZE: Lifts requirements and identifies "Missing Info".
   - *Interactive Mode*: If enabled, it PAUSES here to ask the user clarifying questions using the `interaction_callback`.
3. PROPOSE: Generates 3 strategic approaches (e.g., "Socratic", "Role-based") and selects the best one.
4. DRAFT: Writes the initial prompt using the chosen strategy.
5. REFINE: A second agent pass reviews the draft for clarity and edge cases.
6. FORMAT: Assembles the final output into the standardized block format.

3. MODES OF OPERATION
---------------------
A. Fast Mode (CLI)
   Command: python main.py "Your Task"
   - Runs the pipeline linearly without stopping.

B. Builder Mode (CLI)
   Command: python main.py -b "Your Task"
   - Runs the pipeline but enables the interactive callback.
   - The CLI listens for the callback, prompts the user via stdin, and feeds answers back to the pipeline.

C. Server Mode (Web)
   Command: python main.py serve
   - Starts a Uvicorn server hosting `api.py`.
   - Useful for hosting the backend on Render/Fly.io.

4. KEY CONFIGURATION
--------------------
- config.py: Controls global settings like the OpenAI Model (`gpt-4o-mini`), timeouts, and strictness.
- .env: Must contain your `OPENAI_API_KEY`.

5. OUTPUT
---------
All pipeline runs are saved to `data/runs.jsonl` for debugging and history.
The CLI output is printed to stdout using `rich` for readability.
The API returns JSON responses.
