# Autonomous AI Agents for Data Analysis and Visualization

## Part 1. Quick Start and Commands

This project is a two-agent system for SQLite data analysis and company-style visualization.
It can be run from either a Streamlit UI or a command-line interface.

Run the commands below from the project root.

### 1. Start the Streamlit UI

```powershell
venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

### 2. Run the CLI in Offline Mode

```powershell
venv\Scripts\python.exe cli_app.py --user alice --db chinook --question "Show total sales by country" --mode offline --chart bar --limit 5 --preview-rows 5
```

### 3. Run the CLI in Online OpenAI Mode

Online mode assumes `.env` already contains a valid `OPENAI_API_KEY`.

```powershell
venv\Scripts\python.exe cli_app.py --user alice --db chinook --question "Which countries have the highest average invoice value?" --mode online --chart bar --limit 10 --preview-rows 10
```

### 4. Test User Access Restriction

```powershell
venv\Scripts\python.exe cli_app.py --user bob --db chinook --question "Show total sales by country" --mode offline
```

Expected result:

```text
Error: User 'bob' is not allowed to access database 'chinook'.
```

### 5. Add a New Database from CLI

```powershell
venv\Scripts\python.exe cli_app.py --add-database --db-name custom_sales --db-path data/custom_sales.sqlite --description "Custom sales analytics database" --grant-user admin
```

Grant multiple users:

```powershell
venv\Scripts\python.exe cli_app.py --add-database --db-name custom_sales --db-path data/custom_sales.sqlite --description "Custom sales analytics database" --grant-user admin --grant-user alice
```

Run a new database with online mode:

```powershell
venv\Scripts\python.exe cli_app.py --user admin --db custom_sales --question "Show total amount by region" --mode online --limit 10
```

### 6. Run OpenAI Diagnostics

```powershell
venv\Scripts\python.exe openai_diagnostics.py
```

List available models:

```powershell
venv\Scripts\python.exe openai_diagnostics.py --list-models
```

Only check completion:

```powershell
venv\Scripts\python.exe openai_diagnostics.py --check-completion
```

### 7. Run Automated Tests

Show test names:

```powershell
venv\Scripts\python.exe -m pytest -v
```

Run one test file:

```powershell
venv\Scripts\python.exe -m pytest tests\test_analysis_agent.py -v
```

### 8. Run SQL Generation Evaluation

Offline starter eval:

```powershell
venv\Scripts\python.exe sql_generation_evaluator.py --mode offline
```

Online OpenAI eval:

```powershell
venv\Scripts\python.exe sql_generation_evaluator.py --mode online
```

Use a smaller returned row limit:

```powershell
venv\Scripts\python.exe sql_generation_evaluator.py --mode online --limit 5
```

### 9. Save Feedback from CLI

Use the `Run id` printed by a previous CLI analysis:

```powershell
venv\Scripts\python.exe cli_app.py --feedback-run-id your_run_id_here --feedback-rating incorrect --feedback-comment "The result used the wrong grouping."
```

### 10. Stop the Streamlit Server

Press:

```text
Ctrl + C
```

## Part 2. Project Explanation

### 1. Project Overview

This project implements a controllable two-agent workflow for analyzing SQLite databases and generating visual outputs. Users can run it through Streamlit for a live demo or through the CLI for scripted execution.

The system is designed around four assignment requirements:

- User controllability
- User access restriction
- Easy database integration
- Automated evaluation examples

### 2. Requirement Mapping

| Assignment requirement | Project implementation |
| --- | --- |
| Two distinct AI agents | `DataAnalysisAgent` and `DataVisualizationAgent` |
| Agents connected | `AgentOrchestrator` runs analysis first, then visualization |
| Simple UI or CLI | `streamlit_app.py` and `cli_app.py` |
| User controllability | User chooses database, question, mode, chart type, row limit, and preview rows |
| User restriction | `config/users.yaml` controls allowed databases per user |
| Easy integration | New databases can be added through Streamlit, CLI, or YAML config |
| Automated testing | `pytest` tests in `tests/` |
| Live demonstration | Streamlit app at `http://localhost:8501` |

### 3. Architecture

```text
User
  -> Streamlit UI or CLI
  -> AccessControl
       checks whether the user can access the selected database
  -> DatabaseRegistry
       resolves the logical database name to a SQLite file
  -> DataAnalysisAgent
       builds a metadata index from database/table/column/sample-question text
       embeds the user question and retrieves relevant schema context
       validates question relevance
       generates SQL through an OpenAI tool-calling loop or offline templates
       checks SQL safety
       queries SQLite
       returns AnalysisResult
  -> DataVisualizationAgent
       receives AnalysisResult.dataframe
       creates a chart using company style
       saves chart PNG and CSV
  -> User receives SQL, summary, table preview, chart, and downloadable outputs
```

### 4. Main Workflow

```text
1. User selects username.
2. System loads allowed databases for that user.
3. User selects database, mode, chart type, row limit, and question.
4. Orchestrator checks access permission.
5. Data Analysis Agent retrieves relevant schema metadata before SQL generation.
6. Data Analysis Agent creates safe SQL and returns a DataFrame.
7. Data Visualization Agent creates a company-style chart.
8. UI or CLI displays the result.
```

### 5. The Two Agents

`DataAnalysisAgent`

- Reads SQLite schema.
- Builds a metadata index from database descriptions, tables, columns, generated column descriptions, and sample questions.
- Embeds the user question and retrieves the most relevant database/table/column context.
- Checks whether the question looks like a database-analysis request.
- Uses an OpenAI tool-calling loop in online mode to inspect schema, test SQL, and repair execution errors.
- Uses exact predefined SQL templates in offline mode.
- Runs SQL safety checks before querying the database.
- Returns SQL, summary, row count, SQL source, and a pandas DataFrame.

`DataVisualizationAgent`

- Receives the DataFrame from the analysis agent.
- Chooses or applies the requested chart type.
- Uses `config/style.yaml` for company colors, size, font, and grid style.
- Saves chart images and CSV outputs under `outputs/charts/`.

### 6. Online Mode vs Offline Mode

`Online OpenAI`

- Calls the OpenAI API.
- Gives the model local tools for retrieved schema inspection, SQL validation, and safe SQL execution.
- Feeds SQL execution errors back to the model so it can repair the query before returning final SQL.
- Supports flexible custom questions.
- Required for newly added databases because they have no fixed sample SQL templates.

`Offline fallback`

- Does not call OpenAI.
- Only accepts predefined sample questions.
- Maps each sample question to a fixed SQL template.
- Useful for stable demos and repeatable tests.

The assignment does not require both modes. They are included to balance AI flexibility with demo reliability.

### 7. UI Controls

`Database`

- Selects the SQLite database.
- Options are filtered by the logged-in user's permission.

`Mode`

- `Online OpenAI` for flexible questions.
- `Offline fallback` for predefined sample questions.

`Chart`

- `auto`: system chooses a chart.
- `bar`: category comparison.
- `line`: trend over time.
- `scatter`: numeric relationship.
- `table`: raw table-style result.

`Row limit`

- Controls the maximum rows returned by the SQL query.
- It limits final returned rows, not necessarily the internal database scan or aggregation.

`Preview rows`

- Controls how many rows are displayed in the UI preview.
- It does not change SQL, chart generation, or CSV output.

### 8. Databases

The default configured databases are:

`chinook`

- Music store database.
- Good for sales, invoices, artists, genres, customers, and monthly trends.

Example questions:

```text
Show total sales by country
Show artist sales ranking
Show monthly sales trend
```

`northwind`

- Sales and orders database.
- Good for products, revenue, customers, employees, and stock.

Example questions:

```text
Show top products by revenue
Show revenue by category
Show products with low stock
```

`sakila`

- DVD rental database.
- Good for films, rentals, payments, stores, actors, and categories.

Example questions:

```text
Show revenue by category
Show film rental count
Show top customers by payment amount
```

### 9. User Restriction

User permissions are stored in:

```text
config/users.yaml
```

Example:

```yaml
users:
  alice:
    allowed_databases:
      - chinook
      - northwind
  bob:
    allowed_databases:
      - sakila
  admin:
    allowed_databases:
      - chinook
      - northwind
      - sakila
```

The UI only shows databases the current user can access. The orchestrator also checks permission before running any analysis.

### 10. Easy Database Integration

Database definitions are stored in:

```text
config/databases.yaml
```

Example:

```yaml
databases:
  my_new_database:
    path: data/my_new_database.db
    description: My new sample database
```

There are three ways to add a database:

- Streamlit admin database management panel
- CLI `--add-database` command
- Manual edit of `config/databases.yaml` and `config/users.yaml`

Newly added databases do not automatically appear in the sample-question selector. They should be used with typed custom questions in online mode unless fixed SQL templates are manually added.

### 11. SQL Safety

The system includes several safety checks:

- SQLite connections are read-only.
- Users can only query authorized databases.
- Generated SQL must pass `sql_guard.py`.
- SQL generation receives retrieved schema context instead of blindly using every table and column.
- Online SQL is preflighted through the tool-calling loop before final execution.
- Only `SELECT` and `WITH` style read queries are allowed.
- Dangerous keywords such as `DROP`, `DELETE`, `UPDATE`, `INSERT`, `ALTER`, and `CREATE` are blocked.
- A row limit is added to prevent very large accidental result sets.
- Irrelevant questions are rejected before SQL generation.

### 12. Logging and Feedback

Each pipeline run is logged to:

```text
outputs/logs/analysis_runs.json
```

Each log record stores:

- User
- Database
- Question
- Generated SQL
- SQL source
- Error message
- Row count
- Chart type
- Latency
- User feedback

The Streamlit UI can save feedback after a result is shown. The CLI can also save feedback with `--feedback-run-id`. This makes it possible to trace root causes when a user says a result is wrong.

### 13. Testing and Evaluation

The automated tests cover:

- User access control
- Adding and updating users
- Adding new database integrations
- Database registry path resolution
- SQLite schema reading
- SQL safety checks
- Semantic schema metadata retrieval
- Analysis Agent offline behavior
- Invalid question rejection
- Online tool-calling SQL repair behavior
- SQL-generation execution accuracy and result accuracy checks
- Golden SQL and golden DataFrame comparison with numeric tolerance
- Analysis run logging and user feedback updates
- Visualization Agent chart generation
- End-to-end pipeline
- New user plus new database access
- Starter SQL-generation evaluation harness

The SQL-generation evaluator is intentionally a starter benchmark. It checks execution success, result accuracy, expected result columns, minimum row count, important SQL fragments, golden SQL output, golden DataFrame values, invalid SQL rate, and repair success rate. It is still not a large-scale accuracy evaluation for OpenAI-generated SQL.

### 14. Project Structure

```text
autonomous-ai-data-agents/
  cli_app.py
  streamlit_app.py
  openai_diagnostics.py
  sql_generation_evaluator.py
  README.md
  requirements.txt
  .env.example
  .gitignore

  config/
    databases.yaml
    users.yaml
    style.yaml

  data/
    chinook.db
    northwind_small.sqlite
    sakila.db

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

    utils/
      config_loader.py

    visualization/
      chart_factory.py
      company_style.py

  tests/
    test_foundation.py
    test_analysis_agent.py
    test_observability.py
    test_semantic_schema_retrieval.py
    test_visualization_agent.py
    test_sql_generation_evaluator.py

  outputs/
    charts/
    logs/
    reports/
```

### 15. Recommended Live Demo Flow

1. Start Streamlit.
2. Log in as `bob`.
3. Show that only `sakila` is available.
4. Run `Show revenue by category`.
5. Show that the Data Analysis Agent finishes first.
6. Show that the Data Visualization Agent receives the result and creates the chart.
7. Show SQL, summary, table preview, PNG chart, and CSV download.
8. Click `Clear result`.
9. Log in as `admin`.
10. Add a new user and assign permissions.
11. Test an invalid question such as `Tell me a joke about pizza`.
12. Optionally show CLI or automated tests.

### 16. Troubleshooting

If Streamlit asks for an email on first launch, press Enter to skip it.

If the UI is blank, stop the server with `Ctrl + C` and restart it.

If online mode fails, run:

```powershell
venv\Scripts\python.exe openai_diagnostics.py --list-models
```

Then update `OPENAI_MODEL` in `.env` to a model available to the API key.
