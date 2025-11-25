#style.py
# This file holds the QSS (Qt Style Sheet) for the application.
# It enforces a uniform blue color scheme and consistent font sizes.

# --- UNIFORM FONT SIZES ---
FONT_SIZE_BASE = "15px"
FONT_SIZE_HEADER_SMALL = "17px"
FONT_SIZE_HEADER_LARGE = "25px"

# --- UNIFORM COLOR PALETTE ---
COLOR_BACKGROUND = "#2C3E50"      # Dark Slate Blue
COLOR_BACKGROUND_LIGHT = "#34495E"  # Medium Slate
COLOR_PRIMARY_BLUE = "#3498DB"      # Bright Blue
COLOR_PRIMARY_HOVER = "#5DADE2"      # Lighter Blue (Hover)
COLOR_PRIMARY_DISABLED = "#566573"  # Muted Gray-Blue (Disabled)
COLOR_TEXT = "#FFFFFF"            # White
COLOR_TEXT_DISABLED = "#95A5A6"     # Gray (Disabled Text)
COLOR_BLACK = "#000000"           # For video background

STYLESHEET = f"""
    /* Set default font for all widgets */
    QWidget {{
        background-color: {COLOR_BACKGROUND};
        color: {COLOR_TEXT};
        font-family: Arial, Helvetica, sans-serif;
        font-size: {FONT_SIZE_BASE};
    }}

    /* Main Title Label */
    #TitleLabel {{
        font-size: {FONT_SIZE_HEADER_LARGE};
        font-weight: bold;
    }}

    /* Video feed titles */
    #FeedTitleLabel {{
        font-size: {FONT_SIZE_HEADER_SMALL};
        font-weight: bold;
    }}

    /* Video feed display labels */
    #VideoFeedLabel {{
        background-color: {COLOR_BLACK};
        border: 2px solid {COLOR_BACKGROUND_LIGHT};
        border-radius: 8px;
    }}

    /* Control Bar Container */
    #ControlBar {{
        background-color: {COLOR_BACKGROUND_LIGHT};
        border-radius: 10px;
        padding: 10px;
    }}

    /* All Buttons & Dropdowns */
    QPushButton, QComboBox {{
        font-weight: bold;
        color: {COLOR_TEXT};
        background-color: {COLOR_PRIMARY_BLUE};
        padding: 10px 20px;
        border-radius: 8px;
        min-height: 30px; /* Set a minimum height */
    }}

    QPushButton:hover, QComboBox:hover {{
        background-color: {COLOR_PRIMARY_HOVER};
    }}

    QPushButton:disabled, QComboBox:disabled {{
        background-color: {COLOR_PRIMARY_DISABLED};
        color: {COLOR_TEXT_DISABLED};
    }}

    /* Dropdown specific */
    QComboBox::drop-down {{ border: none; }}
    QComboBox QAbstractItemView {{
        background-color: {COLOR_BACKGROUND_LIGHT};
        color: {COLOR_TEXT};
        selection-background-color: {COLOR_PRIMARY_BLUE};
    }}

    /* Status Bar */
    QStatusBar {{
        font-size: {FONT_SIZE_BASE};
        font-weight: bold;
    }}
    
    /* Recording/Pause Indicators (Simple white text) */
    #RecordingIndicator, #PauseLabel {{
        font-size: 24px;
        font-weight: bold;
        color: {COLOR_TEXT};
        background-color: rgba(0, 0, 0, 0.5); /* Semi-transparent black bg */
        border-radius: 5px;
        padding: 10px;
    }}
"""

