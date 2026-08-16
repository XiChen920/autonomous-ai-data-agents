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


def load_access_control() -> AccessControl:
    return AccessControl()


def load_database_registry() -> DatabaseRegistry:
    return DatabaseRegistry()


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


def get_allowed_database_options(
    access_control: AccessControl,
    database_registry: DatabaseRegistry,
    user: str,
) -> list[str]:
    allowed = set(access_control.allowed_databases(user))
    configured = set(database_registry.list_databases())
    return sorted(allowed.intersection(configured))


def reset_question_state_if_needed(database: str, examples: list[str]) -> None:
    if st.session_state.get("active_question_database") == database:
        return

    selected = examples[0] if examples else CUSTOM_QUESTION_LABEL
    st.session_state["active_question_database"] = database
    st.session_state["selected_example"] = selected
    st.session_state["question_text"] = selected if selected != CUSTOM_QUESTION_LABEL else ""


def sync_question_from_example() -> None:
    selected = st.session_state.get("selected_example", CUSTOM_QUESTION_LABEL)
    if selected == CUSTOM_QUESTION_LABEL:
        st.session_state["question_text"] = ""
    else:
        st.session_state["question_text"] = selected


def sync_example_from_question() -> None:
    question_text = st.session_state.get("question_text", "").strip()
    selected = st.session_state.get("selected_example", CUSTOM_QUESTION_LABEL)

    if not question_text:
        st.session_state["selected_example"] = CUSTOM_QUESTION_LABEL
    elif selected != CUSTOM_QUESTION_LABEL and question_text != selected:
        st.session_state["selected_example"] = CUSTOM_QUESTION_LABEL


def apply_page_style() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 4.25rem;
            padding-bottom: 2rem;
            max-width: 1180px;
        }
        section[data-testid="stSidebar"] > div:first-child {
            padding-top: 3.5rem;
        }
        .result-meta {
            color: #52616F;
            font-size: 0.92rem;
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
            margin-bottom: 0.05rem;
        }
        .agent-step-done .agent-step-title {
            color: #2D8A4E;
        }
        .agent-step-body {
            color: #52616F;
            font-size: 0.92rem;
            line-height: 1.35;
        }
        .agent-all-done {
            color: #2D8A4E;
            font-weight: 700;
            margin-top: 0.75rem;
        }
        code {
            white-space: pre-wrap;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


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


def render_all_done() -> None:
    st.markdown(
        '<div class="agent-all-done">✓ All done. Data Analysis Agent and Data Visualization Agent completed the analysis workflow.</div>',
        unsafe_allow_html=True,
    )


def analysis_working_step(use_openai: bool | None = None, sql_source: str | None = None) -> tuple[str, str]:
    if use_openai is False or sql_source == "fallback":
        return (
            "Data Analysis Agent: offline analysis working",
            "Checking user access, reading database schema, matching the sample question to an offline SQL template, validating SQL safety, and querying SQLite.",
        )

    if use_openai is True and not os.getenv("OPENAI_API_KEY"):
        return (
            "Data Analysis Agent: fallback analysis working",
            "Checking user access and reading database schema. OPENAI_API_KEY is missing, so the agent will use the offline sample-question template.",
        )

    return (
        "Data Analysis Agent: online analysis working",
        "Checking user access, reading database schema, asking OpenAI to generate SQL, validating SQL safety, and querying SQLite.",
    )


def analysis_done_step(analysis) -> tuple[str, str]:
    if analysis.sql_source == "openai":
        return (
            "Data Analysis Agent: online analysis done",
            f"Generated SQL with OpenAI, validated it, queried SQLite, and returned {analysis.row_count} rows.",
        )

    if analysis.sql_source == "fallback":
        return (
            "Data Analysis Agent: offline analysis done",
            f"Matched the question to a predefined SQL template, validated it, queried SQLite, and returned {analysis.row_count} rows.",
        )

    return (
        "Data Analysis Agent: analysis task done",
        f"Returned {analysis.row_count} rows using {analysis.sql_source} SQL.",
    )


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
        render_user_management(access_control, database_registry, user)

    examples = EXAMPLE_QUESTIONS.get(database, [])
    reset_question_state_if_needed(database, examples)
    example_options = [CUSTOM_QUESTION_LABEL] + examples

    st.selectbox(
        "Example question",
        example_options,
        key="selected_example",
        on_change=sync_question_from_example,
        disabled=result_locked,
    )
    st.text_area(
        "Question",
        height=90,
        key="question_text",
        on_change=sync_example_from_question,
        disabled=result_locked,
    )
    current_question = st.session_state.get("question_text", "").strip()
    offline_custom_question = (
        not use_openai
        and bool(current_question)
        and not is_supported_fallback_question(database, current_question)
    )

    if offline_custom_question and not result_locked:
        st.warning(
            "Offline fallback only supports predefined analysis sample questions. "
            "Please choose a supported analysis sample, or switch Mode to Online OpenAI for custom questions."
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
        disabled=offline_custom_question,
    )

    if not submitted:
        st.info("Ready for analysis.")
        return

    question = st.session_state.get("question_text", "").strip()

    if not question:
        st.error("Question is required.")
        return

    if not use_openai and not is_supported_fallback_question(database, question):
        st.error(
            "Offline fallback only supports predefined analysis sample questions. "
            "Please choose a supported analysis sample, or switch Mode to Online OpenAI for custom questions."
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
