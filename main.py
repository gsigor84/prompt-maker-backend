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
from agentic_flow.thinking_partner_router import router as thinking_router

fastapi_app.include_router(agent_router)
fastapi_app.include_router(thinking_router)


# CORS is handled in api.py at the top level of the FastAPI lifecycle.


console = Console()

@click.group(invoke_without_command=True)
@click.pass_context
@click.option("-b", "--builder", is_flag=True, help="Run in Interactive Builder Mode")
@click.argument("args", nargs=-1)
def cli(ctx, builder, args):
    """
    Prompt Master: The AI Pipeline for perfect prompts.

    Usage:
      python main.py "Your simple task"  (Fast Mode)
      python main.py -b "Complex task"   (Interactive Builder)
      python main.py serve               (Start Web Server)
      python main.py thinking "Query"    (Reframe & Audit)
    """
    # 1. If a subcommand was properly invoked (rare with greedy args), let it run
    if ctx.invoked_subcommand is not None:
        return

    # 2. Check if the first argument is a known subcommand
    if args and args[0] in cli.list_commands(ctx):
        cmd_name = args[0]
        if cmd_name == "serve":
            ctx.invoke(serve)
        elif cmd_name == "thinking":
            query = args[1] if len(args) > 1 else ""
            ctx.invoke(thinking, query=query)
        return


    # 3. Defaut to pipeline if no command found and args exist
    if not args:
        click.echo(ctx.get_help())
        return

    task_input = " ".join(args)
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

@cli.command()
@click.argument("query")
def thinking(query: str):
    """
    Reframe and Audit a query (Thinking Partner Mode).
    """
    from thinking_partner import ThinkingPartnerService
    from rich.panel import Panel
    
    console.print(Panel(f"[bold cyan]🔍 Analyzing Query:[/bold cyan]\n{query}", title="Thinking Partner"))
    
    with console.status("[bold yellow]Thinking...") as status:
        service = ThinkingPartnerService()
        result = service.analyze(query)
        
    console.print("\n[bold red]DIAGNOSIS:[/bold red]")
    console.print(f"[white]{result.diagnosis.rationale}[/white]")
    for issue in result.diagnosis.issues:
        console.print(f" • [dim]{issue}[/dim]")
        
    console.print("\n[bold green]REFRAME SUGGESTIONS:[/bold green]")
    for i, s in enumerate(result.suggestions, 1):
        console.print(f"\n[bold gold1]{i}. {s.type}[/bold gold1]")
        console.print(f"[italic]{s.educational_note}[/italic]")
        console.print(f"[white]'{s.content}'[/white]")
        
    console.print("\n[bold blue]TIPS:[/bold blue]")
    for tip in result.tips:
        console.print(f" • {tip}")


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
