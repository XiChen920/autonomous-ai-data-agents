"""Root launcher for the Streamlit UI.

Streamlit executes this file directly, while the actual UI implementation lives
in ``src/ui/streamlit_app.py``.
"""

from src.ui.streamlit_app import main


if __name__ == "__main__":
    main()

