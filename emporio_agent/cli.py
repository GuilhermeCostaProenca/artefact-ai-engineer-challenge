"""Terminal chat interface.

Usage:
    python -m emporio_agent [--show-tools]

Commands inside the chat: /nova (new conversation), /salvar [arquivo.md], /ferramentas (toggle tool trace), /sair
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path

import anthropic
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from .agent import AgentReply, StoreAgent
from .config import load_settings
from .data_store import DataStore

console = Console()


class Transcript:
    """Collects the conversation so it can be saved as a Markdown example."""

    def __init__(self):
        self.lines: list[str] = []

    def add_user(self, text: str):
        self.lines.append(f"**Cliente:** {text}\n")

    def add_agent(self, reply: AgentReply):
        for call in reply.tool_calls:
            args = json.dumps(call.input, ensure_ascii=False)
            result = json.dumps(call.result, ensure_ascii=False, indent=2)
            status = " (erro)" if call.is_error else ""
            self.lines.append(
                f"<details><summary>🔧 <code>{call.name}({args})</code>{status}</summary>\n\n"
                f"```json\n{result}\n```\n</details>\n"
            )
        self.lines.append(f"**Dó (Empório da Música):** {reply.text}\n")

    def save(self, path: Path, header: str) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{header}\n\n" + "\n".join(self.lines), encoding="utf-8")
        return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Empório da Música — assistente de atendimento")
    parser.add_argument("--show-tools", action="store_true", help="mostra as chamadas de ferramentas")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    settings = load_settings()
    store = DataStore(settings.data_dir)
    if not (os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN")):
        console.print("[red]ANTHROPIC_API_KEY não configurada.[/red] Copie .env.example para .env e preencha a chave.")
        return 1
    client = anthropic.Anthropic()

    agent = StoreAgent(client, settings, store)
    transcript = Transcript()
    show_tools = args.show_tools

    console.print(Panel.fit(
        "[bold]🎸 Empório da Música[/bold] — Sua música começa aqui.\n"
        f"[dim]Modelo: {settings.model} · effort: {settings.effort} · agora: {settings.now():%d/%m/%Y %H:%M}[/dim]\n"
        "[dim]Comandos: /nova · /salvar \\[arquivo.md] · /ferramentas · /sair[/dim]",
        border_style="magenta",
    ))

    while True:
        try:
            user_text = console.input("\n[bold cyan]Você:[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            return 0
        if not user_text:
            continue

        command, _, rest = user_text.partition(" ")
        if command in ("/sair", "/exit", "/quit"):
            return 0
        if command == "/nova":
            agent = StoreAgent(client, settings, store)
            transcript = Transcript()
            console.print("[dim]Nova conversa iniciada.[/dim]")
            continue
        if command == "/ferramentas":
            show_tools = not show_tools
            console.print(f"[dim]Rastreamento de ferramentas: {'ligado' if show_tools else 'desligado'}[/dim]")
            continue
        if command == "/salvar":
            default = settings.runtime_dir / "transcripts" / f"conversa-{datetime.now():%Y%m%d-%H%M%S}.md"
            path = Path(rest.strip()) if rest.strip() else default
            header = f"# Conversa — Empório da Música\n\n_Modelo: {settings.model} · data simulada: {settings.now():%d/%m/%Y %H:%M}_"
            console.print(f"[dim]Salvo em {transcript.save(path, header)}[/dim]")
            continue

        transcript.add_user(user_text)
        with console.status("[dim]Dó está digitando...[/dim]", spinner="dots"):
            reply = agent.reply(user_text)
        transcript.add_agent(reply)

        if show_tools:
            for call in reply.tool_calls:
                marker = "[red]✗[/red]" if call.is_error else "[green]✓[/green]"
                console.print(f"  {marker} [dim]{call.name}({json.dumps(call.input, ensure_ascii=False)})[/dim]")
        console.print("[bold magenta]Dó:[/bold magenta]")
        console.print(Markdown(reply.text))


if __name__ == "__main__":
    raise SystemExit(main())
