# Orientation: Lagos Rent Assistant current state

This document describes the system as found on branch phase-1-cleanup before cleanup. Every bullet is labelled verified (seen in code or git) or inferred (deduced from structure or runtime notes).

## 1. Top level layout

- Verified: Entry points are entrypoints/api.py (FastAPI) and entrypoints/cli.py (Typer CLI). No other entry point is wired.
- Verified: Top level Python files are config.py (env config), db.py (SQLAlchemy engine), plus legacy root files app1.py and train.py (removed in this phase, see docs/cleanup.md).
- Verified: Packages are advice/, conversation/, eval/, observability/, pipeline/, query/, scraping/. SQL schema is sql/schema.sql. Local data is data/ (scraper CSV, checkpoints, known area JSON). docker-compose.yml defines only Postgres.
- Verified: No frontend directory, no package.json, no Dockerfile, no pyproject.toml, no .python-version, no uv.lock, no CI directory exist in the working tree at orientation time.
- Verified: README.md still describes the old Streamlit ML predictor, not the conversational assistant. STATUS_REPORT.md exists in the working tree but is gitignored and untracked.
- Verified: docs/ did not exist before this phase.

## 2. Import graph from entry points

- Verified: entrypoints/api.py imports query.entities (EntityExtractionError, QueryType, extract_entities), query.query_engine (run_lookup, run_affordability, run_comparison), query.response (build_lookup_response, build_affordability_response, build_comparison_response), conversation.session (load_session_entities, save_session_entities), advice.advice_engine (answer_advice_query), observability.tracer (start_trace, trace_stage), plus fastapi, loguru, pydantic.
- Verified: entrypoints/cli.py imports the same query, conversation, advice and observability modules, plus typer, sys and uuid. It adds utf8 reconfigure for stdout and stderr in the working tree (uncommitted change at orientation time).
- Verified: query.entities imports config.GROQ_MODEL, observability.tracer.record_call, groq, loguru, pydantic, python-dotenv. The working tree imports GROQ_MODEL from config; HEAD defined it separately in both files.
- Verified: query.query_engine imports db.get_engine, query.entities.PropertyEntities and QueryType, scraping.location_normalizer.normalize_location, observability.tracer.record_call.
- Verified: conversation.session imports db.get_engine and query.entities.PropertyEntities.
- Verified: advice.advice_engine imports advice.search_client, config keys, observability.tracer.record_call, groq.
- Verified: advice.search_client imports config.TAVILY_API_KEY, tavily.TavilyClient, observability.tracer.record_call.
- Verified: observability.tracer imports db.get_engine.
- Verified: db.py imports sqlalchemy, loguru and python-dotenv. config.py imports os and python-dotenv.
- Verified: pipeline/load_data.py imports db.get_engine and scraping.location_normalizer.normalize_location. scraping/scraper.py imports only requests, beautifulsoup4, stdlib, and writes to data/raw. Nothing in entrypoints imports scraping.scraper or pipeline.load_data; they are run as standalone scripts.
- Verified: Legacy imports streamlit, joblib, sklearn and xgboost appear only in app1.py, train.py and the notebook. No active module imports them.

## 3. Actual request flow for a question

- Verified: POST /query accepts a JSON body QueryRequest with fields query (string, required) and session_id (optional string). GET /health returns status ok. This is the complete API surface.
- Verified: query_listings calls start_trace, loads previous entities from conversation_sessions when session_id is present, then calls extract_entities with previous entities for carry over.
- Verified: extract_entities calls Groq chat completion with a system prompt that asks for JSON with fields query_type, beds, area, budget_ngn, budget_period, budget_period_was_explicit, property_type, listing_type, comparison_options and resolved_query. It retries up to 2 attempts on JSON or validation error, then raises EntityExtractionError which becomes HTTP 422.
- Verified: query_type lookup runs run_lookup which filters listings by is_duplicate_of IS NULL plus optional area, beds and listing_type (default rent, with early return of empty list when no filter at all is present). Affordability runs run_affordability which requires budget_ngn. Comparison runs run_comparison which reuses build_lookup_query per option. Advice calls answer_advice_query with resolved_query.
- Verified: lookup orders by annualized price ascending, limit 10. Affordability orders by annualized price ascending, limit 30, with stretch factor 1.15. Comparison uses limit 5 per option.
- Verified: advice path calls Tavily advanced search for up to 5 results, then Groq synthesis with a grounded prompt. Search or synthesis failures return friendly error answers with empty sources, not HTTP errors.
- Verified: after a successful run the API saves updated entities to conversation_sessions when session_id is present, and returns a dict built by response.py. Trace stages extraction, query_execution or advice_synthesis are persisted to the traces table by the tracer context manager.
- Verified: CLI query and chat commands mirror the API steps without HTTP. chat loops with a persistent session_id, loads previous entities per turn, and saves after each successful turn.
- Inferred: The system assumes one question equals one LLM extraction call followed by one SQL query set, except advice which adds two external calls (Tavily then Groq).

## 4. Current response shape

- Verified: Lookup success returns query_type, matched_count, message and results list where each result has title, price (formatted NGN string with period suffix when present), area, subarea, bedrooms and url.
- Verified: Lookup empty returns query_type, matched_count zero, message No matching listings found for that search. and empty results.
- Verified: Affordability returns query_type, budget_ngn, matched_count, message, areas grouped by area with matched_count, sparse flag and up to 5 results per area, and stretch_options. When budget is missing it returns a clarification style message asking for a budget. When nothing is found it returns matched_count zero and empty lists.
- Verified: Comparison returns query_type, message and options where each option has label, matched_count, caveat, cheapest formatted price or null, and results. When no valid options exist it returns empty options. Region terms (mainland, island) add a caveat that region filtering is unavailable.
- Verified: Advice returns query_type advice, answer string and sources list with title and url.
- Verified: No status enum, no confidence, no percentiles, no data_as_of, no request_id or latency, no streaming, no idempotency key and no clarification object exist today. Affordability sparse is the only low data signal (sparse when an area has fewer than 3 rows, threshold SPARSE_AREA_THRESHOLD).

## 5. Database tables and columns

- Verified: listings has id serial primary key, source text not null, listing_type text not null, url text unique not null, title text, price_raw text, price_ngn numeric, price_period text, location_raw text, area text, subarea text, bedrooms integer, bathrooms integer, toilets integer, parking integer, property_type text, is_duplicate_of integer references listings id, price_anomaly_flag boolean default false, scraped_at timestamptz, created_at timestamptz default now. Indexes on area plus bedrooms plus listing_type, on price_ngn and on url.
- Verified: location_aliases has raw_variant text primary key, normalized_area text not null, normalized_subarea text, needs_review boolean default false, created_at timestamptz default now.
- Verified: conversation_sessions has session_id text primary key, last_entities jsonb not null, updated_at timestamptz default now, with index on updated_at.
- Verified: traces has id serial primary key, trace_id text not null, stage text not null, query_type text, duration_ms numeric, call_counts jsonb, success boolean not null, error text, created_at timestamptz default now, with indexes on trace_id and created_at.
- Verified: No question_events, scraper_runs, area_freshness or admin_users tables exist yet. No Alembic setup exists. pipeline/load_data.py applies sql/schema.sql with apply_schema and then upserts listings by url.
- Inferred: Postgres is the single source of truth for answers. Listing rows are never deleted by load_data; stale rows remain after rescrape, matching the STATUS_REPORT note about phantom rows.

## 6. Annual normalisation

- Verified: query/query_engine.py defines _annualized_price_expr as (CASE WHEN price_period = 'month' THEN price_ngn * 12 ELSE price_ngn END) and uses it for ordering in build_lookup_query and for filtering and ordering in build_affordability_query. Non month rows (year, null) are treated as already annual.
- Verified: query/response.py format_price prints the stored price_ngn with a period suffix when price_period is present. It does not annualize for display.
- Verified: scraping/scraper.py _clean_price uses PRICE_RE for naira sign plus digits and PERIOD_RE for yr, year, mo or month anywhere in the truncated page text. It maps yr and year to year and mo or month to month. _infer_rent_period maps per annum and monthly phrases to year or month. Comments in the file state PropertyPro often omits period on the primary listing and warn against loosening period regex.
- Verified: pipeline/load_data.py flags anomalies: rent_missing_price_period when listing_type is rent and price_period is null; implausible monthly rent when rent monthly price exceeds 10,000,000 NGN; implausible annual rent when rent yearly price is below 300,000 NGN; property_type_equals_area_bug when property_type equals lagos (case insensitive); missing_location when location_raw is null.
- Verified: query/entities.py defaults budget_period to year unless the user explicitly states a monthly signal, and budget_period_was_explicit tracks explicitness. Affordability compares annual_price to budget directly; monthly budgets are not multiplied into annual amounts in query_engine (inferred gap to confirm in Phase 4).
- Inferred: Annual figures in the product depend on scraper period detection and can silently mix monthly and yearly rows when period is null, because null is treated as annual. Phase 4 must exclude invalid prices and never silently mix periods.

## 7. Property types

- Verified: query/entities.py prompt allows specific structural types such as flat, duplex, self contain, mini flat, terrace, bungalow, detached house, semi-detached and penthouse. Generic words apartment, house, place, property, home, accommodation and somewhere are treated as no property type.
- Verified: _sanitize_property_type drops generic terms from both the main entities and comparison options.
- Verified: scraper parse_npc_url derives property_type from the URL segment before lagos, parse_propertypro_url derives it from slug patterns including self-contain. Shortlet URLs are excluded from regular rent output.
- Verified: pipeline/load_data.py GENERIC_PROPERTY_TYPE_VALUES contains only lagos and flags equality as a bug. No allowlist validation exists. pipeline/backfill_property_type.py and backfill_property_type_db.py are one off corrections that import parse_npc_url from lagos_property_scraper (wrong module path) and are not invoked by the active app.
- Inferred: Property type values are free text from two sources plus LLM output, so the four breakdown types (self_contain, one_bedroom, two_bedroom, three_bedroom) required in Phase 4 will need explicit mapping from existing beds and property_type columns.

## 8. What is logged today

- Verified: observability/tracer.py creates a uuid trace id per request, counts per service calls (groq, tavily, db) via record_call, and writes one row per trace_stage to traces with trace_id, stage, query_type, duration_ms, call_counts JSON, success and error.
- Verified: entrypoints/api.py writes loguru logs/app.log with rotation 10 MB and 14 day retention at info level. entrypoints/cli.py writes logs/cli.log the same way.
- Verified: conversation/session.py logs session expiry and unparseable stored entities at info and warning levels.
- Verified: query code logs extraction results, lookup counts, affordability counts and comparison skips at info and warning levels.
- Verified: scraping/scraper.py uses stdlib logging to stdout with rate limit and retry messages. It writes checkpoint JSON files under data/raw per site, category and type.
- Verified: No analytics question log, no IP capture, no scraper_runs table and no request latency field exist today. Latency is only per stage duration in traces.
- Inferred: Logs and traces are local files and local Postgres rows only. No dashboard or export reads them in the repo today.

## 9. Contradictions with STATUS_REPORT.md

- Verified: STATUS_REPORT.md dated 2026-09-22 states that observability is entirely uncommitted and untracked. Git now tracks observability/tracer.py, so that statement is stale.
- Verified: STATUS_REPORT.md states that no .env.example exists. The repo now tracks .env.example (committed in e0f5527 and extended in the working tree), so that statement is stale.
- Verified: STATUS_REPORT.md states that GROQ_MODEL is hard coded twice and that llama-3.3-70b-versatile returns 404. The working tree at orientation time already changed both sites to env driven openai/gpt-oss-120b, so the report is stale unless those six working tree edits are discarded. This phase preserves those edits.
- Verified: STATUS_REPORT.md states that requirements.txt is incomplete and polluted with legacy ML deps. HEAD still has that old content, but the working tree pins runtime deps, adds sqlalchemy, psycopg2-binary, python-dotenv, rapidfuzz, requests, beautifulsoup4 and drops sklearn, xgboost, streamlit and joblib. The report describes HEAD, not the working tree.
- Verified: STATUS_REPORT.md states that CLI advice crashes on Windows due to unicode encoding at entrypoints/cli.py. The working tree adds sys.stdout and sys.stderr reconfigure to utf-8, which addresses that crash. HEAD still has the bug.
- Verified: Remaining STATUS_REPORT findings still accurate at orientation time: README is stale, frontend and CI are absent, build_lookup_query duplicates the listing_type filter block, dedup_review_flag is computed then dropped before insert, _clean_price uses page wide regex, affiliation evals are thin and LLM dependent, and stale rows are never retired.
- Verified: STATUS_REPORT.md itself is not in git. git ls-files does not list it, git log shows no history for it, and .gitignore lists STATUS_REPORT.md. Its date and audit claims therefore cannot be verified from history.

## 10. Environment and secrets

- Verified: .env exists locally and is gitignored. git ls-files does not list .env, and git log shows no commits touching .env. Never commit it and never print its values.
- Verified: .env.example lists POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DB, POSTGRES_HOST, POSTGRES_PORT, GROQ_API_KEY, TAVILY_API_KEY and GROQ_MODEL with empty values except GROQ_MODEL default.
- Verified: config.py get_env reads env vars with optional defaults. db.py defaults to lagos_rent credentials and localhost 5432 when env is unset.
- Verified: .venv, logs, __pycache__ and eval/reports are ignored but present locally. No tracked secrets were found, but rotation advice is kept in docs/cleanup.md as a precaution because .env history was checked only via log (inferred safe).
