"""Streamlit UI for controlling the two-agent data analysis workflow.

The UI provides a simple login-like workspace, user-specific database access,
admin user management, analysis controls, and result downloads.
"""

import os
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.analysis_agent import (
    AnalysisAgentError,
    DataAnalysisAgent,
    is_supported_fallback_question,
)
from src.agents.orchestrator import AgentOrchestrator
from src.agents.visualization_agent import DataVisualizationAgent
from src.auth.access_control import (
    AccessControl,
    AccessControlError,
    InvalidUserConfigError,
    UnknownUserError,
)
from src.db.connector import DatabaseQueryError
from src.db.registry import DatabaseRegistry, DatabaseRegistryError
from src.db.sql_guard import UnsafeSQLError
from src.visualization.chart_factory import ChartCreationError


EXAMPLE_QUESTIONS = {
    "chinook": [
        "Show total sales by country",
        "Show invoice count by country",
        "Show artist sales ranking",
        "Show top genres by track count",
        "Show customer count by country",
        "Show monthly sales trend",
        "Tell me a joke about pizza",
    ],
    "northwind": [
        "Show top products by revenue",
        "Show order count by country",
        "Show revenue by category",
        "Show revenue by customer",
        "Show products with low stock",
        "Show orders by employee",
        "What is the weather tomorrow?",
    ],
    "sakila": [
        "Show revenue by category",
        "Show film rental count",
        "Show top customers by payment amount",
        "Show rentals by store",
        "Show actor film count",
        "Show revenue by country",
        "Tell me a joke about pizza",
    ],
}

CUSTOM_QUESTION_LABEL = "None / custom question"


# Creates the access-control service for the UI.
def load_access_control() -> AccessControl:
    return AccessControl()


# Creates the database registry service for the UI.
def load_database_registry() -> DatabaseRegistry:
    return DatabaseRegistry()


# Returns predefined example questions for built-in demo databases.
def get_example_questions(database: str) -> list[str]:
    return EXAMPLE_QUESTIONS.get(database, [])


# Builds the orchestrator with UI-selected row limit and mode.
def build_orchestrator(row_limit: int, use_openai: bool) -> AgentOrchestrator:
    analysis_agent = DataAnalysisAgent(
        row_limit=row_limit,
        use_openai=use_openai,
    )
    visualization_agent = DataVisualizationAgent()
    return AgentOrchestrator(
        analysis_agent=analysis_agent,
        visualization_agent=visualization_agent,
    )


# Computes the databases visible to the logged-in user.
def get_allowed_database_options(
    access_control: AccessControl,
    database_registry: DatabaseRegistry,
    user: str,
) -> list[str]:
    allowed = set(access_control.allowed_databases(user))
    configured = set(database_registry.list_databases())
    return sorted(allowed.intersection(configured))


# Resets question controls when the selected database changes.
def reset_question_state_if_needed(database: str, examples: list[str]) -> None:
    if st.session_state.get("active_question_database") == database:
        return

    selected = examples[0] if examples else CUSTOM_QUESTION_LABEL
    st.session_state["active_question_database"] = database
    st.session_state["selected_example"] = selected
    st.session_state["question_text"] = selected if selected != CUSTOM_QUESTION_LABEL else ""


# Copies the selected example question into the editable text area.
def sync_question_from_example() -> None:
    selected = st.session_state.get("selected_example", CUSTOM_QUESTION_LABEL)
    if selected == CUSTOM_QUESTION_LABEL:
        st.session_state["question_text"] = ""
    else:
        st.session_state["question_text"] = selected


# Switches the example selector to custom mode when the text is edited.
def sync_example_from_question() -> None:
    question_text = st.session_state.get("question_text", "").strip()
    selected = st.session_state.get("selected_example", CUSTOM_QUESTION_LABEL)

    if not question_text:
        st.session_state["selected_example"] = CUSTOM_QUESTION_LABEL
    elif selected != CUSTOM_QUESTION_LABEL and question_text != selected:
        st.session_state["selected_example"] = CUSTOM_QUESTION_LABEL


# Injects lightweight CSS for spacing and agent status text.
def apply_page_style() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 4.25rem;
            padding-bottom: 2rem;
            max-width: 1180px;
            font-size: 1.06rem;
        }
        section[data-testid="stSidebar"] > div:first-child {
            padding-top: 3.5rem;
        }
        section[data-testid="stSidebar"] {
            min-width: 340px !important;
            width: 340px !important;
        }
        section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
            font-size: 1.08rem !important;
            font-weight: 700;
        }
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] span,
        section[data-testid="stSidebar"] label {
            font-size: 1.02rem !important;
        }
        section[data-testid="stSidebar"] button {
            font-size: 1.02rem !important;
            min-height: 2.65rem;
        }
        section[data-testid="stSidebar"] input,
        section[data-testid="stSidebar"] textarea,
        section[data-testid="stSidebar"] [data-baseweb="select"] * {
            font-size: 1.02rem !important;
        }
        .block-container [data-testid="stMarkdownContainer"] p,
        .block-container textarea,
        .block-container input,
        .block-container button {
            font-size: 1.04rem;
        }
        .block-container h1 {
            font-size: 2.45rem;
        }
        .block-container h2 {
            font-size: 1.85rem;
        }
        .block-container h3 {
            font-size: 1.45rem;
        }
        .result-meta {
            color: #52616F;
            font-size: 1rem;
            margin: 0.25rem 0 1rem 0;
        }
        .agent-step {
            margin: 0.5rem 0;
            padding: 0.1rem 0;
        }
        .agent-step-done {
            margin-top: 0.15rem;
        }
        .agent-step-title {
            color: #1F77B4;
            font-weight: 700;
            font-size: 1.06rem;
            margin-bottom: 0.05rem;
        }
        .agent-step-done .agent-step-title {
            color: #2D8A4E;
        }
        .agent-step-body {
            color: #52616F;
            font-size: 1rem;
            line-height: 1.35;
        }
        .agent-all-done {
            color: #2D8A4E;
            font-weight: 700;
            font-size: 1.06rem;
            margin-top: 0.75rem;
        }
        code {
            white-space: pre-wrap;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# Renders the username entry screen and stores the logged-in user.
def render_login(access_control: AccessControl) -> str | None:
    st.title("Autonomous Data Agents")
    st.subheader("User access")

    with st.form("login-form"):
        username = st.text_input("Username", placeholder="alice, bob, or admin")
        submitted = st.form_submit_button("Enter workspace", type="primary")

    if not submitted:
        demo_users = ", ".join(access_control.list_users())
        st.caption(f"Demo users: {demo_users}")
        return None

    username = username.strip()
    if not username:
        st.error("Please enter a username.")
        return None

    try:
        access_control.allowed_databases(username)
    except UnknownUserError:
        demo_users = ", ".join(access_control.list_users())
        st.error(f"Unknown user '{username}'. Available demo users: {demo_users}.")
        return None

    st.session_state["logged_in_user"] = username
    st.rerun()
    return None


# Renders the top workspace title, switch-user button, and permission summary.
def render_workspace_header(user: str, access_control: AccessControl) -> None:
    header_columns = st.columns([3, 1])
    header_columns[0].title("Autonomous Data Agents")
    header_columns[0].caption(f"Workspace for user: {user}")

    if header_columns[1].button("Switch user", use_container_width=True):
        st.session_state.pop("logged_in_user", None)
        st.session_state.pop("analysis_output", None)
        st.rerun()

    allowed = ", ".join(access_control.allowed_databases(user))
    st.info(f"Authorized databases for {user}: {allowed}")


# Renders the admin-only panel for adding or updating database integrations.
def render_database_management(
    access_control: AccessControl,
    database_registry: DatabaseRegistry,
    current_user: str,
) -> None:
    st.divider()
    st.subheader("Database management")

    notice = st.session_state.pop("database_admin_notice", None)
    if notice:
        notice_type, notice_text = notice
        if notice_type == "success":
            st.success(notice_text)
        elif notice_type == "info":
            st.info(notice_text)
        else:
            st.error(notice_text)

    if current_user != "admin":
        st.caption("Only admin can add or update database integrations.")
        return

    with st.expander("Add or update database"):
        database_name = st.text_input(
            "Database name",
            placeholder="for example custom_sales",
            key="new-database-name",
        )
        database_path = st.text_input(
            "SQLite file path",
            placeholder="for example data/custom_sales.sqlite",
            key="new-database-path",
        )
        database_description = st.text_input(
            "Description",
            placeholder="for example Custom sales analytics database",
            key="new-database-description",
        )
        grant_users = st.multiselect(
            "Grant access to users",
            access_control.list_users(),
            default=[current_user],
            key="new-database-grant-users",
        )

        if st.button("Save database integration", use_container_width=True):
            try:
                update_result = database_registry.add_or_update_database(
                    database_name,
                    database_path,
                    database_description,
                )
                grant_messages = []
                for username in grant_users:
                    grant_result = access_control.grant_database_to_user(
                        username,
                        update_result.database_name,
                    )
                    grant_messages.append(
                        f"{grant_result.username}: {grant_result.status}"
                    )
            except (AccessControlError, DatabaseRegistryError) as exc:
                st.error(str(exc))
            else:
                if update_result.status == "created":
                    message = (
                        f"New database '{update_result.database_name}' loaded successfully. "
                        f"Path: {update_result.current_database['path']}."
                    )
                    notice_type = "success"
                elif update_result.status == "updated":
                    changed = ", ".join(update_result.changed_fields)
                    message = (
                        f"Updated database '{update_result.database_name}'. "
                        f"Changed fields: {changed}."
                    )
                    notice_type = "success"
                else:
                    message = f"No changes for database '{update_result.database_name}'."
                    notice_type = "info"

                if grant_messages:
                    message += f" User grants: {'; '.join(grant_messages)}."
                else:
                    message += " No user permissions were changed."

                message += (
                    " Newly added databases do not have sample questions; "
                    "use Online OpenAI mode and type a custom question."
                )
                st.session_state["database_admin_notice"] = (notice_type, message)
                st.rerun()


# Renders the admin-only panel for creating or updating user permissions.
def render_user_management(
    access_control: AccessControl,
    database_registry: DatabaseRegistry,
    current_user: str,
) -> None:
    st.divider()
    st.subheader("User management")

    if current_user != "admin":
        st.caption("Only admin can create or update user database permissions.")
        return

    with st.expander("Add or update user"):
        new_username = st.text_input(
            "Username",
            placeholder="for example charlie",
            key="new-username",
        )
        selected_databases = st.multiselect(
            "Allowed databases",
            database_registry.list_databases(),
            key="new-user-databases",
        )

        if st.button("Save user permissions", use_container_width=True):
            try:
                update_result = access_control.add_or_update_user(
                    new_username,
                    selected_databases,
                )
            except InvalidUserConfigError as exc:
                st.error(str(exc))
            else:
                permissions = ", ".join(update_result.current_databases)
                if update_result.status == "created":
                    st.success(
                        f"New user '{update_result.username}' loaded successfully. "
                        f"Permissions: {permissions}."
                    )
                elif update_result.status == "updated":
                    changes = []
                    if update_result.added_databases:
                        changes.append(f"added {', '.join(update_result.added_databases)}")
                    if update_result.removed_databases:
                        changes.append(f"removed {', '.join(update_result.removed_databases)}")
                    change_text = "; ".join(changes)
                    st.success(
                        f"Updated user '{update_result.username}'. "
                        f"Changes: {change_text}. Current permissions: {permissions}."
                    )
                else:
                    st.info(
                        f"No changes for user '{update_result.username}'. "
                        f"Permissions remain: {permissions}."
                    )


# Renders one visible status line for an agent task.
def render_agent_step(title: str, body: str, done: bool = False) -> None:
    state_class = " agent-step-done" if done else ""
    check = "✓ " if done else ""
    st.markdown(
        (
            f'<div class="agent-step{state_class}">'
            f'<div class="agent-step-title">{check}{title}</div>'
            f'<div class="agent-step-body">{body}</div>'
            f"</div>"
        ),
        unsafe_allow_html=True,
    )


# Renders the final success line after both agents finish.
def render_all_done() -> None:
    st.markdown(
        '<div class="agent-all-done">✓ All done. Data Analysis Agent and Data Visualization Agent completed the analysis workflow.</div>',
        unsafe_allow_html=True,
    )


# Chooses the analysis status text for online, fallback, or offline mode.
def analysis_working_step(use_openai: bool | None = None, sql_source: str | None = None) -> tuple[str, str]:
    if use_openai is False or sql_source == "fallback":
        return (
            "Data Analysis Agent: offline analysis working",
            "Checking user access, retrieving relevant schema metadata, matching the sample question to an offline SQL template, validating SQL safety, and querying SQLite.",
        )

    if use_openai is True and not os.getenv("OPENAI_API_KEY"):
        return (
            "Data Analysis Agent: fallback analysis working",
            "Checking user access and retrieving relevant schema metadata. OPENAI_API_KEY is missing, so the agent will use the offline sample-question template.",
        )

    return (
        "Data Analysis Agent: online analysis working",
        "Checking user access, retrieving relevant schema metadata, using OpenAI tool calls to test and repair SQL, validating SQL safety, and querying SQLite.",
    )


# Chooses the completed analysis status text from the actual SQL source.
def analysis_done_step(analysis) -> tuple[str, str]:
    if analysis.sql_source == "openai":
        return (
            "Data Analysis Agent: online analysis done",
            f"Retrieved relevant schema, generated and preflighted SQL with OpenAI tools, validated it, queried SQLite, and returned {analysis.row_count} rows.",
        )

    if analysis.sql_source == "fallback":
        return (
            "Data Analysis Agent: offline analysis done",
            f"Retrieved relevant schema, matched the question to a predefined SQL template, validated it, queried SQLite, and returned {analysis.row_count} rows.",
        )

    return (
        "Data Analysis Agent: analysis task done",
        f"Returned {analysis.row_count} rows using {analysis.sql_source} SQL.",
    )


# Re-renders the persisted agent timeline after a result is locked.
def render_agent_history(analysis, visualization) -> None:
    with st.status(
        "Both agents finished the analysis workflow",
        state="complete",
        expanded=True,
    ):
        analysis_working_title, analysis_working_body = analysis_working_step(
            sql_source=analysis.sql_source
        )
        analysis_done_title, analysis_done_body = analysis_done_step(analysis)
        render_agent_step(
            analysis_working_title,
            analysis_working_body,
        )
        render_agent_step(
            analysis_done_title,
            analysis_done_body,
            done=True,
        )
        render_agent_step(
            "Data Visualization Agent: working",
            "Selecting the chart format, applying company style, and saving PNG/CSV outputs.",
        )
        render_agent_step(
            "Data Visualization Agent: visualization task done",
            f"Created a {visualization.chart_type} chart and saved output files.",
            done=True,
        )
        render_all_done()


# Renders summary, chart, data preview, SQL, and download links.
def render_pipeline_result(result, analysis, visualization, preview_rows: int) -> None:
    chart_label = visualization.chart_type if visualization else "none"
    st.markdown(
        (
            f'<div class="result-meta">'
            f"Rows: {analysis.row_count} | "
            f"SQL source: {analysis.sql_source} | "
            f"Chart: {chart_label} | "
            f"Database: {result.database}"
            f"</div>"
        ),
        unsafe_allow_html=True,
    )

    st.subheader("Summary")
    st.write(analysis.summary)

    if analysis.retrieved_tables:
        with st.expander("Retrieved schema context", expanded=False):
            st.write(f"Tables: {', '.join(analysis.retrieved_tables)}")
            if analysis.retrieved_columns:
                st.write(f"Columns: {', '.join(analysis.retrieved_columns[:20])}")

    if visualization is not None:
        st.subheader("Visualization")
        st.image(str(visualization.chart_path), use_container_width=True)

        with open(visualization.chart_path, "rb") as chart_file:
            st.download_button(
                "Download chart",
                data=chart_file,
                file_name=visualization.chart_path.name,
                mime="image/png",
            )

    data_tab, sql_tab, files_tab = st.tabs(["Data", "SQL", "Files"])

    with data_tab:
        st.dataframe(
            analysis.dataframe.head(preview_rows),
            use_container_width=True,
            hide_index=True,
        )

    with sql_tab:
        st.code(analysis.sql, language="sql")

    with files_tab:
        if visualization is not None:
            st.write(f"Chart path: `{visualization.chart_path}`")
            st.write(f"CSV path: `{visualization.data_path}`")

            with open(visualization.data_path, "rb") as data_file:
                st.download_button(
                    "Download CSV",
                    data=data_file,
                    file_name=visualization.data_path.name,
                    mime="text/csv",
                )


# Runs the Streamlit app from login through analysis and visualization.
def main() -> None:
    load_dotenv()

    st.set_page_config(
        page_title="Autonomous Data Agents",
        page_icon="A",
        layout="wide",
    )
    apply_page_style()

    access_control = load_access_control()
    database_registry = load_database_registry()

    if "logged_in_user" not in st.session_state:
        st.session_state["logged_in_user"] = None

    user = st.session_state["logged_in_user"]
    if user is None:
        render_login(access_control)
        return

    render_workspace_header(user, access_control)
    result_locked = st.session_state.get("analysis_output") is not None

    with st.sidebar:
        st.header("Controls")
        allowed_databases = get_allowed_database_options(
            access_control,
            database_registry,
            user,
        )

        if not allowed_databases:
            st.error("No databases available for this user.")
            return

        database = st.selectbox("Database", allowed_databases, disabled=result_locked)
        database_info = database_registry.get_database(database)
        st.caption(database_info.get("description", ""))

        mode = st.radio(
            "Mode",
            ["Online OpenAI", "Offline fallback"],
            horizontal=False,
            disabled=result_locked,
        )
        use_openai = mode == "Online OpenAI"

        if use_openai and not os.getenv("OPENAI_API_KEY"):
            st.warning("OPENAI_API_KEY is not set; fallback SQL will be used.")

        chart_type = st.selectbox(
            "Chart",
            ["auto", "bar", "line", "scatter", "table"],
            disabled=result_locked,
        )
        row_limit = st.slider(
            "Row limit",
            min_value=1,
            max_value=100,
            value=10,
            disabled=result_locked,
        )
        preview_rows = st.slider("Preview rows", min_value=1, max_value=50, value=10)
        render_database_management(access_control, database_registry, user)
        render_user_management(access_control, database_registry, user)

    examples = get_example_questions(database)
    reset_question_state_if_needed(database, examples)

    if examples:
        example_options = [CUSTOM_QUESTION_LABEL] + examples
        st.selectbox(
            "Example question",
            example_options,
            key="selected_example",
            on_change=sync_question_from_example,
            disabled=result_locked,
        )
    else:
        st.selectbox(
            "Example question",
            [CUSTOM_QUESTION_LABEL],
            key="selected_example",
            disabled=True,
        )
        st.caption(
            "No predefined sample questions are available for this database. "
            "Type a custom question and use Online OpenAI mode."
        )

    st.text_area(
        "Question",
        height=90,
        key="question_text",
        on_change=sync_example_from_question,
        disabled=result_locked,
    )
    current_question = st.session_state.get("question_text", "").strip()
    offline_without_examples = not use_openai and not examples
    offline_custom_question = (
        not use_openai
        and bool(current_question)
        and not is_supported_fallback_question(database, current_question)
    )
    online_without_key_for_new_database = (
        use_openai
        and not os.getenv("OPENAI_API_KEY")
        and not examples
    )

    if offline_without_examples and not result_locked:
        st.warning(
            "Offline fallback is only available for built-in sample databases with predefined questions. "
            "For newly added databases, switch Mode to Online OpenAI and type a custom question."
        )
    elif offline_custom_question and not result_locked:
        st.warning(
            "Offline fallback only supports predefined analysis sample questions. "
            "Please choose a supported analysis sample, or switch Mode to Online OpenAI for custom questions."
        )
    elif online_without_key_for_new_database and not result_locked:
        st.warning(
            "This database has no offline SQL templates. Set OPENAI_API_KEY before running custom questions in Online OpenAI mode."
        )

    if result_locked:
        if st.button("Clear result", type="primary"):
            st.session_state.pop("analysis_output", None)
            st.rerun()

        saved_output = st.session_state["analysis_output"]
        render_agent_history(saved_output["analysis"], saved_output["visualization"])
        render_pipeline_result(
            saved_output["result"],
            saved_output["analysis"],
            saved_output["visualization"],
            preview_rows=preview_rows,
        )
        return

    submitted = st.button(
        "Run analysis",
        type="primary",
        disabled=offline_without_examples
        or offline_custom_question
        or online_without_key_for_new_database,
    )

    if not submitted:
        st.info("Ready for analysis.")
        return

    question = st.session_state.get("question_text", "").strip()

    if not question:
        st.error("Question is required.")
        return

    if not use_openai and (
        not examples or not is_supported_fallback_question(database, question)
    ):
        st.error(
            "Offline fallback only supports predefined analysis sample questions. "
            "For newly added databases, switch Mode to Online OpenAI and type a custom question."
        )
        return

    orchestrator = build_orchestrator(
        row_limit=row_limit,
        use_openai=use_openai,
    )

    try:
        with st.status("Running agents...", expanded=True) as agent_status:
            analysis_working_title, analysis_working_body = analysis_working_step(
                use_openai=use_openai
            )
            render_agent_step(
                analysis_working_title,
                analysis_working_body,
            )
            result = orchestrator.run_analysis(
                user=user,
                database=database,
                question=question,
            )
            analysis = result.analysis
            analysis_done_title, analysis_done_body = analysis_done_step(analysis)
            render_agent_step(
                analysis_done_title,
                analysis_done_body,
                done=True,
            )

            render_agent_step(
                "Data Visualization Agent: working",
                "Selecting the chart format, applying company style, and saving PNG/CSV outputs.",
            )
            visualization = orchestrator.visualization_agent.visualize(
                analysis,
                chart_type=chart_type,
            )
            render_agent_step(
                "Data Visualization Agent: visualization task done",
                f"Created a {visualization.chart_type} chart and saved output files.",
                done=True,
            )
            render_all_done()
            agent_status.update(
                label="Both agents finished the analysis workflow",
                state="complete",
                expanded=True,
            )
    except (
        AccessControlError,
        AnalysisAgentError,
        DatabaseRegistryError,
        DatabaseQueryError,
        UnsafeSQLError,
        ChartCreationError,
    ) as exc:
        st.error(str(exc))
        return

    st.session_state["analysis_output"] = {
        "result": result,
        "analysis": analysis,
        "visualization": visualization,
    }
    st.rerun()


if __name__ == "__main__":
    main()
