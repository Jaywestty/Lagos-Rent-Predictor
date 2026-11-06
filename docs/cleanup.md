# Cleanup log for Phase 1

This log records what was removed, what was kept and what still needs a decision. Verification used the import graph from entrypoints/api.py and entrypoints/cli.py plus grep across tracked Python files.

## Removed with git rm

- app1.py (tracked): Old Streamlit predictor. Imports streamlit, joblib and Regressor.pkl. No active module imports it. Removed to preserve history via git rm.
- train.py (tracked): Old training script. Imports sklearn, xgboost and joblib and reads lagos-rent 3.csv. No active module imports it. Removed to preserve history via git rm.
- Regressor.pkl (tracked): Pickled model artifact. Only loaded by app1.py. No active module loads it. Removed to preserve history via git rm.
- lagos_house_rent_prediction.ipynb (tracked): Notebook with sklearn analysis. No active module executes it. Removed to preserve history via git rm.
- Lagos Rent Dashboard.pbix (tracked): Binary Power BI file. No code reference. Removed to preserve history via git rm.
- lagos-rent 3.csv (tracked): Training CSV read only by train.py. Not used by pipeline/load_data.py which reads data/raw/lagos_properties.csv. Removed to preserve history via git rm.

All six files remain in git history and can be restored with git show and git checkout if needed.

## Kept and why

- data/: Contains scraper output, checkpoints, known area JSON and quality report. Left untouched per phase instructions. data/raw/lagos_properties.csv is the pipeline input. Inferred: Do not delete local data files in this phase.
- eval/: Golden queries and eval scripts for extraction and advice. Left untouched per phase instructions. Phase 4 must keep existing evals runnable.
- pipeline/: load_data.py is live. backfill_property_type.py, backfill_property_type_db.py and check_dedup.py are one off or debug scripts with a broken import to lagos_property_scraper, but the phase rule says not to touch pipeline. They are listed as needing decision below.
- scraping/: scraper.py, location_normalizer.py, derive_known_areas.py and generate_admin_suffixes.py support data collection and normalization. Left untouched per phase instructions.
- sql/: schema.sql defines the live tables. Left untouched per phase instructions.
- .env: Local secret file. It is gitignored and untracked. It was not touched, not printed and not committed.
- STATUS_REPORT.md: Present in the working tree but gitignored and untracked. Kept in place for now. Phase 3 will decide to keep or fold into README. Deleting it in Phase 1 would violate the do not touch spirit and lose audit context.
- .env.example: Tracked and edited in the working tree to add GROQ_MODEL. Kept. The edit fixes the dead model issue noted in the status report.
- config.py, query/entities.py, entrypoints/cli.py, requirements.txt, .gitignore: Working tree edits were preserved. Config makes the Groq model env driven, entities removes the duplicate hard coded model, CLI adds utf8 handling, requirements pins runtime deps and adds missing DB deps while dropping legacy ML deps, gitignore adds .venv. See orientation doc for details.
- docker-compose.yml: Kept as is. Phase 2 may update it if requirements handling changes, but Phase 1 did not change it.
- README.md: Kept as is for now. Phase 3 will rewrite it.

## Gitignore changes

- Added missing entries node_modules, dist, .vite and coverage to .gitignore so future frontend and test artifacts stay untracked.
- Verified that .env, logs/, __pycache__/ and .venv/ are already ignored and are not tracked. git ls-files shows none of them. No untracking with git rm --cached was needed.
- git status --ignored confirms .env, .venv, __pycache__ folders, logs/ and STATUS_REPORT.md are ignored in the working tree.
- No .env file was ever committed (git log for .env is empty), so no key rotation is required because of history. Rotation is still advised if the .env file was ever shared outside this machine, because the file holds live Groq and Tavily keys.

## Needs decision

- pipeline backfill scripts: backfill_property_type.py and backfill_property_type_db.py import parse_npc_url from lagos_property_scraper which does not exist as a module (the real parser is scraping.scraper.parse_npc_url). They appear already applied. Decide to delete, archive or fix in a later phase. Not removed here because the phase rule forbids touching pipeline.
- pipeline/check_dedup.py: Ad hoc debug viewer that prints 10 duplicate pairs. Decide to move to scripts or delete later.
- pipeline/load_data.py stray import: from numpy.ma import anomalies at line 6 shadows the local anomalies dict. Harmless at runtime but misleading. Fix in Phase 2 or Phase 4 planning.
- query duplicate filter: build_lookup_query repeats the listing_type filter block. No behavior change, but should be deduped when the answer contract lands in Phase 4.
- STATUS_REPORT.md ignore: The file is ignored, so future edits will not be committed. Decide in Phase 3 to untrack or delete or track it. Orientation doc documents the contradiction.
- requirements pinning: The working tree pins exact versions such as fastapi 0.141.1 and pandas 3.0.6. Phase 2 will replace this with pyproject.toml and uv.lock, which will re-resolve versions on Python 3.12. Approve that swap.
- Untracked large local files: __pycache__ folders, logs and .venv exist locally but are ignored. They were left on disk. Confirm they should stay ignored.

## Verification done

- Import graph built by reading entrypoints/api.py and entrypoints/cli.py and listing imports of every tracked Python file outside .venv. Legacy files have zero inbound imports from the active graph.
- grep for app1, train, Regressor, lagos-rent, lagos_house and Dashboard shows hits only inside the legacy files themselves.
- git ls-files confirms the six legacy files were tracked and that no secret, log or pycache file is tracked.
- git status --ignored confirms ignore behavior.
