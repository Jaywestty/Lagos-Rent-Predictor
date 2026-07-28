import typer
from loguru import logger

import uuid
from typing import Optional

from entities import EntityExtractionError, PropertyEntities, QueryType, extract_entities
from query_engine import run_lookup, run_affordability, run_comparison
from response import build_lookup_response, build_affordability_response, build_comparison_response
from session import load_session_entities, save_session_entities

logger.add("logs/cli.log", rotation="10 MB", retention="14 days", level="INFO")

app = typer.Typer()

def _run_and_print(entities: PropertyEntities) -> None:
    if entities.query_type == QueryType.LOOKUP:
        listings = run_lookup(entities)
        result = build_lookup_response(entities, listings)
        typer.echo(result["message"])
        for item in result["results"]:
            typer.echo(f"- {item['title']} | {item['price']} | {item['area']} | {item['url']}")

    elif entities.query_type == QueryType.AFFORDABILITY:
        aff_result = run_affordability(entities)
        result = build_affordability_response(entities, aff_result)
        typer.echo(result["message"])
        for area in result["areas"]:
            flag = " (sparse)" if area["sparse"] else ""
            typer.echo(f"\n{area['area']}{flag} — {area['matched_count']} match(es)")
            for item in area["results"]:
                typer.echo(f"  - {item['title']} | {item['price']} | {item['url']}")
        if result["stretch_options"]:
            typer.echo("\nSlightly above budget:")
            for area in result["stretch_options"]:
                typer.echo(f"\n{area['area']}")
                for item in area["results"]:
                    typer.echo(f"  - {item['title']} | {item['price']} | {item['url']}")

    elif entities.query_type == QueryType.COMPARISON:
        comp_result = run_comparison(entities)
        result = build_comparison_response(entities, comp_result)
        typer.echo(result["message"])
        for option in result["options"]:
            typer.echo(f"\n{option['label']} — {option['matched_count']} match(es)")
            if option.get("caveat"):
                typer.echo(f"  [note: {option['caveat']}]")
            for item in option["results"]:
                typer.echo(f"  - {item['title']} | {item['price']} | {item['area']} | {item['url']}")


@app.command()
def query(text: str):
    try:
        entities = extract_entities(text)
    except EntityExtractionError as exc:
        logger.error("Entity extraction failed: {}", exc)
        typer.echo(f"Could not understand that query: {exc}")
        raise typer.Exit(code=1)

    try:
        _run_and_print(entities)
    except Exception as exc:
        logger.error("Query failed: {}", exc)
        typer.echo("Search failed. Try again.")
        raise typer.Exit(code=1)
    

@app.command()
def chat(session_id: Optional[str] = None):
    session_id = session_id or str(uuid.uuid4())
    typer.echo(f"Session: {session_id}")
    typer.echo("Type your query, or 'exit' to quit.")

    while True:
        text = typer.prompt(">")
        if text.strip().lower() in ("exit", "quit"):
            break

        previous_entities = load_session_entities(session_id)

        try:
            entities = extract_entities(text, previous_entities=previous_entities)
        except EntityExtractionError as exc:
            logger.error("Entity extraction failed: {}", exc)
            typer.echo(f"Could not understand that: {exc}")
            continue

        try:
            _run_and_print(entities)
        except Exception as exc:
            logger.error("Query failed: {}", exc)
            typer.echo("Search failed. Try again.")
            continue

        save_session_entities(session_id, entities)

@app.command()
def serve(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    import uvicorn
    uvicorn.run("main:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    app()