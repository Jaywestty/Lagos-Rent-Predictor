import typer
from loguru import logger

from entities import EntityExtractionError, QueryType, extract_entities
from query_engine import run_lookup
from response import build_lookup_response

logger.add("logs/cli.log", rotation="10 MB", retention="14 days", level="INFO")

app = typer.Typer()


@app.command()
def query(text: str):
    try:
        entities = extract_entities(text)
    except EntityExtractionError as exc:
        logger.error("Entity extraction failed: {}", exc)
        typer.echo(f"Could not understand that query: {exc}")
        raise typer.Exit(code=1)

    if entities.query_type != QueryType.LOOKUP:
        typer.echo(f"Query type '{entities.query_type.value}' is not yet supported.")
        raise typer.Exit(code=1)

    try:
        listings = run_lookup(entities)
    except Exception as exc:
        logger.error("Lookup query failed: {}", exc)
        typer.echo("Search failed. Try again.")
        raise typer.Exit(code=1)

    result = build_lookup_response(entities, listings)
    typer.echo(result["message"])
    for item in result["results"]:
        typer.echo(f"- {item['title']} | {item['price']} | {item['area']} | {item['url']}")


@app.command()
def serve(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    import uvicorn
    uvicorn.run("main:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    app()