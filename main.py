# main.py
"""
Prompt Master CLI & Server Entry Point.
"""

import os
import uvicorn
import click
from rich.console import Console

# ✅ Export app for uvicorn workers
from api import app as fastapi_app
from promptforge_core import build_orchestrator
from prompt_quality import PromptQualityService

# ✅ NEW: Mount the Agentic Flow router
from agentic_flow.router import router as agent_router
fastapi_app.include_router(agent_router)

# CORS is handled in api.py at the top level of the FastAPI lifecycle.


console = Console()

@click.group(invoke_without_command=True)
@click.pass_context
@click.option("-b", "--builder", is_flag=True, help="Run in Interactive Builder Mode")
@click.argument("task_input", required=False)
def cli(ctx, builder, task_input):
    """
    Prompt Master: The AI Pipeline for perfect prompts.

    Usage:
      python main.py "Your simple task"  (Fast Mode)
      python main.py -b "Complex task"   (Interactive Builder)
      python main.py serve               (Start Web Server)
    """
    # 1. If 'serve' was passed, it might be captured as task_input
    if task_input == "serve":
        ctx.invoke(serve)
        return

    # 2. If a subcommand was properly invoked (rare with greedy args), let it run
    if ctx.invoked_subcommand is not None:
        return

    # 3. If no task provided, show help
    if not task_input:
        click.echo(ctx.get_help())
        return

    # 4. Run pipeline
    run_pipeline(task_input, interactive=builder)



@cli.command()
def serve():
    """Start the production web server."""
    port = int(os.getenv("PORT", "8000"))
    use_reload = os.getenv("PIPELINE_MODE") == "dev"
    
    console.print(f"[bold green]Starting Server on port {port}...[/bold green]")
    uvicorn.run(
        "main:fastapi_app",
        host="0.0.0.0",
        port=port,
        reload=use_reload,
        log_level="info",
    )

def run_pipeline(task: str, interactive: bool):
    """Runs the prompt generation pipeline."""
    orchestrator = build_orchestrator()
    quality_service = PromptQualityService(model="gpt-4o-mini")

    console.print(f"\n[bold blue]🚀 Starting Prompt Master...[/bold blue]")
    console.print(f"Task: {task}")
    if interactive:
        console.print("[yellow]Mode: Interactive Builder[/yellow]")
    else:
        console.print("[green]Mode: Fast[/green]")

    # Interaction Handler
    def ask_questions(missing_info: list[str]) -> str:
        console.print("\n[bold yellow]🤔 I need more details to build the perfect prompt:[/bold yellow]")
        answers = []
        for question in missing_info:
            ans = click.prompt(f" - {question}", type=str)
            answers.append(f"Q: {question}\nA: {ans}")
        return "\n".join(answers)

    # Run!
    try:
        final_result = orchestrator.run(
            user_task=task,
            interaction_callback=ask_questions if interactive else None
        )
    except Exception as e:
        console.print(f"\n[bold red]❌ Error during generation:[/bold red] {e}")
        return

    # Output
    console.print("\n" + "="*60)
    console.print("[bold green]✨ GENERATED PROMPT ✨[/bold green]")
    console.print("="*60)
    print(final_result.prompt)
    console.print("="*60)

    # Quality Check
    console.print("\n[bold magenta]🔍 Evaluating Quality...[/bold magenta]")
    score_result = quality_service.score_prompt(final_result.prompt, task)
    
    score_color = "green" if score_result.score >= 8 else "yellow" if score_result.score >= 5 else "red"
    console.print(f"Score: [{score_color}]{score_result.score}/10[/{score_color}]")
    console.print(f"Critique: {score_result.critique}")
    console.print("\nDone.")

# Expose app for Uvicorn (e.g., uvicorn main:app)
app = fastapi_app

if __name__ == "__main__":
    cli()
