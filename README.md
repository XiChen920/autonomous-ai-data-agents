# Autonomous AI Agents for Data Analysis and Visualization

A two-agent system for querying SQLite databases with natural language and generating company-style visualizations. It supports Streamlit UI, CLI, user-based database access control, database onboarding, SQL evaluation, and run logging.

## Quick Start

Run the Streamlit UI:

```powershell
venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

Run the CLI in offline mode:

```powershell
venv\Scripts\python.exe cli_app.py --user alice --db chinook --question "Show total sales by country" --mode offline --chart bar --limit 5 --preview-rows 5
```

Run the CLI in online OpenAI mode:

```powershell
venv\Scripts\python.exe cli_app.py --user alice --db chinook --question "Which countries have the highest average invoice value?" --mode online --chart bar --limit 10
```

Test user restriction:

```powershell
venv\Scripts\python.exe cli_app.py --user bob --db chinook --question "Show total sales by country" --mode offline
```

Add a new database:

```powershell
venv\Scripts\python.exe cli_app.py --add-database --db-name custom_sales --db-path data/custom_sales.sqlite --description "Custom sales analytics database" --grant-user admin
```

Run automated tests:

```powershell
venv\Scripts\python.exe -m pytest -v
```

Run SQL generation evaluation:

```powershell
venv\Scripts\python.exe sql_generation_evaluator.py --mode offline
venv\Scripts\python.exe sql_generation_evaluator.py --mode online --limit 5
```

Save feedback for a previous run:

```powershell
venv\Scripts\python.exe cli_app.py --feedback-run-id your_run_id_here --feedback-rating incorrect --feedback-comment "The result used the wrong grouping."
```

Run OpenAI diagnostics:

```powershell
venv\Scripts\python.exe openai_diagnostics.py --list-models
```

## Architecture

```text
Streamlit UI / CLI
  -> AccessControl
  -> DatabaseRegistry
  -> AgentOrchestrator
  -> DataAnalysisAgent
       semantic schema retrieval
       question relevance check
       OpenAI tool-calling SQL loop or offline SQL templates
       SQL safety check
       read-only SQLite query
  -> DataVisualizationAgent
       company-style chart generation
       chart recommendation when forced chart type is not ideal
       PNG and CSV output
  -> AnalysisLogger
       run logs and user feedback
```

## Main Features

- Two connected agents: `DataAnalysisAgent` and `DataVisualizationAgent`.
- Streamlit UI and CLI share the same backend pipeline.
- User permissions restrict which databases each user can access.
- Admin users can add new SQLite databases through UI, CLI, or YAML config.
- Online mode uses OpenAI for flexible natural-language questions.
- Offline mode uses fixed SQL templates for stable demos and repeatable tests.
- SQL safety layer blocks destructive SQL and uses read-only SQLite connections.
- Visualization uses company style from `config/style.yaml`.
- If a user forces a non-ideal chart type, the chart is still rendered and the system recommends a better chart.
- Every run can be logged with SQL, error, latency, chart type, and user feedback.

## Agentic SQL Generation

Online mode uses a tool-calling loop instead of one-shot SQL generation.

The Analysis Agent can:

- inspect retrieved schema context,
- generate SQL,
- validate SQL,
- run SQL safely,
- observe execution errors,
- repair the query,
- return final SQL only after a successful preflight run.

The loop is implemented without `previous_response_id`, so it also works in OpenAI Zero Data Retention environments.

## Semantic Schema Retrieval

Before SQL generation, the system builds a metadata index from:

- database names and descriptions,
- table names,
- column names,
- generated column descriptions,
- sample questions.

The user question is embedded as well. The system retrieves the most relevant database, table, column, and sample-question context using vector similarity, then passes only the relevant schema context to the LLM. This makes the design more scalable for many databases.

## Evaluation

The SQL generation evaluator checks:

- expected columns,
- minimum row count,
- expected SQL fragments,
- golden SQL,
- golden result DataFrame,
- numeric tolerance value comparison,
- execution accuracy,
- result accuracy,
- invalid SQL rate,
- repair success rate.

Example:

```powershell
venv\Scripts\python.exe sql_generation_evaluator.py --mode online --limit 5
```

## Logging and Feedback

Each pipeline run is logged to:

```text
outputs/logs/analysis_runs.json
```

Each record includes:

- user,
- database,
- question,
- generated SQL,
- SQL source,
- error message,
- row count,
- chart type,
- latency,
- user feedback.

This makes it possible to trace root causes when a user reports an incorrect result.

## Default Databases

`chinook`

- Music store database.
- Good for invoices, sales, customers, artists, genres, and monthly trends.

`northwind`

- Sales and orders database.
- Good for products, revenue, customers, employees, categories, and stock.

`sakila`

- DVD rental database.
- Good for films, rentals, payments, actors, stores, and categories.

## Key Files

```text
config/
  databases.yaml
  users.yaml
  style.yaml

src/
  agents/
    analysis_agent.py
    visualization_agent.py
    orchestrator.py
  auth/
    access_control.py
  cli/
    cli_app.py
  db/
    connector.py
    registry.py
    schema_reader.py
    sql_guard.py
  observability/
    analysis_logger.py
  retrieval/
    schema_metadata_index.py
  tools/
    openai_diagnostics.py
    sql_generation_evaluator.py
    sql_generation_cases.yaml
  ui/
    streamlit_app.py
  visualization/
    chart_factory.py
    company_style.py

tests/
  test_foundation.py
  test_analysis_agent.py
  test_visualization_agent.py
  test_semantic_schema_retrieval.py
  test_sql_generation_evaluator.py
  test_observability.py
```

## Demo Flow

1. Start Streamlit.
2. Log in as `bob` to show restricted access.
3. Run a `sakila` sample question.
4. Show agent status, SQL, chart, data preview, and download files.
5. Force a non-ideal chart type and show the recommended chart message.
6. Switch to `admin`.
7. Add or update a user/database permission.
8. Run an invalid question to show guardrails.
9. Show automated tests or the SQL evaluator.

## Notes

- New databases do not automatically get offline sample questions.
- For newly added databases, use online mode and type a custom question.
- `Row limit` controls SQL result size.
- `Preview rows` only controls how many rows are displayed in the UI/CLI preview.
