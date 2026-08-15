# Autonomous AI Agents for Data Analysis and Visualization

## 1. Project Overview

This project implements a two-agent data analysis system for SQLite databases.
Users can control the system through either a Streamlit UI or a CLI.

Start the Streamlit UI:

```powershell
cd C:\Users\win11\PycharmProjects\autonomous-ai-data-agents
venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

Then open:

```text
http://localhost:8501
```

Run the CLI:

```powershell
cd C:\Users\win11\PycharmProjects\autonomous-ai-data-agents
venv\Scripts\python.exe cli_app.py --user alice --db chinook --question "Show total sales by country" --offline --limit 5
```

The system contains two main agents:

1. Data Analysis Agent
   - Reads the selected database schema.
   - Checks whether the user question is relevant to the database.
   - Generates SQL using either OpenAI or offline fallback templates.
   - Validates that the SQL is safe.
   - Runs the query against SQLite and returns a pandas DataFrame.

2. Data Visualization Agent
   - Receives the analysis result.
   - Selects or applies the requested chart type.
   - Applies the company chart style.
   - Saves a PNG chart and CSV output.

The project is designed around controllability, user restrictions, simple database integration, and automated testing.

## 2. Requirement Mapping

| Assignment requirement | How this project satisfies it |
| --- | --- |
| Two distinct AI agents | `DataAnalysisAgent` and `DataVisualizationAgent` |
| Agents connected | `AgentOrchestrator` runs analysis first, then visualization |
| Simple UI or CLI | Streamlit UI launched by `streamlit_app.py`; CLI launched by `cli_app.py` |
| User controllability | Users choose database, question, mode, chart type, row limit, and preview rows |
| User restriction | `config/users.yaml` limits which databases each user can access |
| Easy integration | New databases can be added through `config/databases.yaml` |
| Automated testing | `pytest` tests in the `tests/` folder |
| Live demonstration | Streamlit app at `http://localhost:8501` |

## 3. Architecture

```text
User: Streamlit UI or CLI
  -> AccessControl
  -> DatabaseRegistry
  -> DataAnalysisAgent
       -> SchemaReader
       -> OpenAI SQL generation or offline fallback SQL
       -> SQLGuard
       -> SQLiteConnector
  -> DataVisualizationAgent
       -> ChartFactory
       -> CompanyStyle
  -> outputs/charts/*.png and outputs/charts/*.csv
```

## 4. Project Structure

```text
autonomous-ai-data-agents/
  cli_app.py
  streamlit_app.py
  openai_diagnostics.py
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

    cli/
      cli_app.py

    auth/
      access_control.py

    db/
      connector.py
      registry.py
      schema_reader.py
      sql_guard.py

    ui/
      streamlit_app.py

    tools/
      openai_diagnostics.py

    utils/
      config_loader.py

    visualization/
      chart_factory.py
      company_style.py

  tests/
    test_foundation.py
    test_analysis_agent.py
    test_visualization_agent.py

  outputs/
    charts/
    reports/
```

## 5. Databases

The project uses three sample SQLite databases.

### Chinook

Music store database.

Good for:

- Sales by country
- Invoice count by country
- Artist sales ranking
- Genre track count
- Monthly sales trend

Example questions:

```text
Show total sales by country
Show artist sales ranking
Show monthly sales trend
```

### Northwind

Orders, products, customers, employees, and suppliers database.

Good for:

- Product revenue
- Revenue by customer
- Revenue by category
- Low stock products
- Employee order performance

Example questions:

```text
Show top products by revenue
Show revenue by category
Show products with low stock
```

### Sakila

DVD rental database.

Good for:

- Revenue by film category
- Film rental count
- Customer payment ranking
- Store rentals
- Actor film count

Example questions:

```text
Show revenue by category
Show film rental count
Show top customers by payment amount
```

## 6. Streamlit UI

Start the app:

```powershell
cd C:\Users\win11\PycharmProjects\autonomous-ai-data-agents
venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

Then open:

```text
http://localhost:8501
```

### Login / User Workspace

The UI starts with a username input.

Configured users are stored in:

```text
config/users.yaml
```

Example:

```text
alice -> chinook, northwind
bob -> sakila
admin -> chinook, northwind, sakila
```

After login, the database selector only shows databases that the user is allowed to access.

### Admin User Management

Only `admin` sees the user management panel.

Admin can:

- Add a new user.
- Choose allowed databases.
- Update an existing user's permissions.
- See whether the save action created, updated, or changed nothing.

Example messages:

```text
New user 'charlie' loaded successfully. Permissions: chinook, sakila.
Updated user 'charlie'. Changes: added northwind; removed sakila.
No changes for user 'charlie'. Permissions remain: chinook, sakila.
```

## 7. UI Controls

### Database

Selects which SQLite database the agents will use.

The options are filtered by the logged-in user's permissions.

### Mode

`Online OpenAI`

- Calls the OpenAI API.
- The model receives the database schema and user question.
- The model generates SQL.
- Best for flexible, new, natural-language questions.

`Offline fallback`

- Does not call OpenAI.
- Uses predefined SQL templates in `analysis_agent.py`.
- Best for stable testing and live demo safety.

The assignment does not require both online and offline modes. They are included to balance AI flexibility with demo reliability.

### Chart

Controls the visualization type:

- `auto`: the system chooses a chart type.
- `bar`: good for category comparisons.
- `line`: good for trends over time.
- `scatter`: good for numeric relationships.
- `table`: good for raw tabular results.

If a forced chart type does not fit the data shape, the chart factory falls back safely instead of crashing.

### Row Limit

Controls how many rows the SQL query returns.

Example:

```sql
LIMIT 10
```

Important: the database may still scan and aggregate all relevant rows internally. The row limit controls the final returned result size, not necessarily the amount of data considered during aggregation.

### Preview Rows

Controls how many rows are displayed in the UI data preview.

It does not change the SQL query, chart, or CSV output.

Example:

```text
Row limit = 50
Preview rows = 10
```

The system queries up to 50 rows, but the UI preview shows 10.

## 8. CLI Usage

Run a stable offline demo:

```powershell
venv\Scripts\python.exe cli_app.py --user alice --db chinook --question "Show total sales by country" --offline --limit 5
```

Run online OpenAI mode:

```powershell
venv\Scripts\python.exe cli_app.py --user alice --db chinook --question "Which countries have the highest average invoice value?" --limit 10
```

Test access denial:

```powershell
venv\Scripts\python.exe cli_app.py --user bob --db chinook --question "Show total sales by country" --offline
```

Expected result:

```text
Error: User 'bob' is not allowed to access database 'chinook'.
```

## 9. OpenAI Configuration

Create a `.env` file:

```env
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-5-mini
```

Run OpenAI diagnostics:

```powershell
venv\Scripts\python.exe openai_diagnostics.py
```

Only list available models:

```powershell
venv\Scripts\python.exe openai_diagnostics.py --list-models
```

Only check whether the configured model can respond:

```powershell
venv\Scripts\python.exe openai_diagnostics.py --check-completion
```

## 10. Testing

Run all automated tests:

```powershell
venv\Scripts\python.exe -m pytest -q
```

Current expected result:

```text
15 passed
```

The tests cover:

- User access control
- Adding and updating user permissions
- Database registry path resolution
- SQLite schema reading
- SQL safety checks
- Analysis Agent fallback behavior
- Invalid question rejection
- Visualization Agent chart generation
- End-to-end analysis and visualization pipeline

## 11. Adding a New Database

1. Put the SQLite file in `data/`.

Example:

```text
data/my_new_database.db
```

2. Add it to `config/databases.yaml`.

```yaml
databases:
  my_new_database:
    path: data/my_new_database.db
    description: My new sample database
```

3. Give a user permission in `config/users.yaml` or through the admin UI.

```yaml
users:
  alice:
    allowed_databases:
      - my_new_database
```

4. Use Online OpenAI mode for flexible questions, or add offline fallback SQL templates to `analysis_agent.py` for stable demo examples.

## 12. Security and Safety

The project includes several safety layers:

- Users can only see authorized databases.
- The orchestrator checks access before running analysis.
- SQLite connections are read-only.
- Agent-generated SQL must pass `sql_guard.py`.
- Mutating SQL keywords are blocked.
- Row limits prevent large accidental result sets.
- Irrelevant user questions are rejected before SQL generation.

## 13. Recommended Live Demo Flow

1. Log in as `bob`.
2. Show that only `sakila` is available.
3. Run:

```text
Show revenue by category
```

4. Show the two agent status section:

```text
Data Analysis Agent: analysis task done
Data Visualization Agent: visualization task done
```

5. Show chart, SQL, data preview, and download buttons.
6. Click `Clear result`.
7. Log in as `admin`.
8. Add a new user and assign permissions.
9. Test an invalid question:

```text
Tell me a joke about pizza
```

Expected result: the system rejects it as unrelated to database analysis.

## 14. Troubleshooting

If Streamlit asks for an email on first launch, press Enter to skip it.

If `localhost:8501` is blank, stop the server with `Ctrl + C` and restart:

```powershell
venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

If port 8501 is busy:

```powershell
venv\Scripts\python.exe -m streamlit run streamlit_app.py --server.port 8502
```

If online mode fails, switch to `Offline fallback` for a stable demo.

If a model access error appears, run:

```powershell
venv\Scripts\python.exe openai_diagnostics.py --list-models
```

Then update `OPENAI_MODEL` in `.env`.
