from __future__ import annotations

import datetime

import reflex as rx
import reflex_webcam as webcam


# --- STATE MANAGEMENT ---


class State(rx.State):
    """The application state."""

    # Status options: "safe", "warning", "critical"
    status: str = "safe"
    confidence: int = 98
    is_streaming: bool = True

    # Logs
    logs: list[dict] = [
        {"time": "10:42:05", "msg": "System initialized"},
        {"time": "10:42:10", "msg": "Camera feed connected"},
    ]

    # Risk factors (only show when not safe)
    risk_factors: list[str] = []

    # Snapshot data from webcam
    snapshot: str = ""
    snapshot_timestamp: str = ""

    # ------- CORE STATE HELPERS -------

    def toggle_stream(self):
        self.is_streaming = not self.is_streaming
        self.add_log("Stream " + ("resumed" if self.is_streaming else "paused"))

    def add_log(self, message: str):
        now = datetime.datetime.now().strftime("%H:%M:%S")
        self.logs.insert(0, {"time": now, "msg": message})
        # Keep log reasonably short
        if len(self.logs) > 20:
            self.logs.pop()

    # ------- SNAPSHOT HANDLER (PHASE 1 INTEGRATION) -------

    def handle_snapshot(self, img_data_uri: str):
        """Called when a snapshot is captured from the webcam."""
        self.snapshot = img_data_uri
        self.snapshot_timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.add_log("Snapshot captured")

    # ------- SIMULATION LOGIC (for testing UI only) -------

    def simulate_critical(self):
        self.status = "critical"
        self.confidence = 92
        self.risk_factors = ["Stoop Detected", "Bad Posture Persistence (>1s)"]
        self.add_log("CRITICAL: Stoop detected")

    def simulate_warning(self):
        self.status = "warning"
        self.confidence = 78
        self.risk_factors = ["Knees not bent enough"]
        self.add_log("WARNING: Check form")

    def simulate_safe(self):
        self.status = "safe"
        self.confidence = 98
        self.risk_factors = []
        self.add_log("Status returned to Safe")


# --- UI COMPONENTS ---


def status_card():
    """The big colored card showing the current AI prediction."""
    return rx.box(
        rx.vstack(
            rx.icon(
                rx.cond(
                    State.status == "safe",
                    "check-circle",
                    rx.cond(State.status == "warning", "alert-triangle", "siren"),
                ),
                size=48,
                color="white",
            ),
            rx.text(
                rx.cond(
                    State.status == "safe",
                    "SAFE POSTURE",
                    rx.cond(State.status == "warning", "WARNING", "CRITICAL RISK"),
                ),
                font_size="2em",
                font_weight="bold",
                color="white",
            ),
            rx.text(
                f"Confidence: {State.confidence}%",
                color="white",
                opacity=0.8,
            ),
            align_items="center",
            spacing="2",
        ),
        # Dynamic Background Color based on Status
        bg=rx.cond(
            State.status == "safe",
            "green",
            rx.cond(State.status == "warning", "orange", "red"),
        ),
        padding="2em",
        border_radius="lg",
        width="100%",
        box_shadow="lg",
        transition="all 0.3s ease",
    )


def risk_factors_panel():
    """Displays specific issues if they exist."""
    return rx.vstack(
        rx.text(
            "ACTIVE RISK FACTORS",
            font_weight="bold",
            color="gray.400",
            font_size="sm",
        ),
        # Show this if safe
        rx.cond(
            State.status == "safe",
            rx.text(
                "No ergonomic risks detected.",
                color="green.400",
                font_size="sm",
            ),
        ),
        # Show list of risks if not safe
        rx.foreach(
            State.risk_factors,
            lambda risk: rx.hstack(
                rx.icon("alert-circle", color="red", size=16),
                rx.text(risk, color="white", font_size="sm"),
                bg="rgba(255, 0, 0, 0.1)",
                padding="0.5em",
                border_radius="md",
                width="100%",
                align_items="center",
            ),
        ),
        width="100%",
        padding_y="1em",
        spacing="2",
    )


def log_panel():
    """Scrollable log of recent events."""
    return rx.vstack(
        rx.text(
            "RECENT EVENT LOG",
            font_weight="bold",
            color="gray.400",
            font_size="sm",
        ),
        rx.scroll_area(
            rx.vstack(
                rx.foreach(
                    State.logs,
                    lambda log: rx.hstack(
                        rx.text(
                            log["time"],
                            color="gray.500",
                            font_family="monospace",
                            font_size="xs",
                        ),
                        rx.text(
                            log["msg"],
                            color="gray.300",
                            font_size="xs",
                        ),
                        spacing="2",
                    ),
                ),
                width="100%",
                spacing="1",
            ),
            type="always",
            scrollbars="vertical",
            style={"height": "150px"},
        ),
        width="100%",
        padding_top="1em",
    )


def snapshot_preview():
    """Shows last snapshot captured from webcam."""
    return rx.cond(
        State.snapshot != "",
        rx.vstack(
            rx.text(
                "LAST SNAPSHOT",
                font_weight="bold",
                color="gray.400",
                font_size="sm",
            ),
            rx.image(src=State.snapshot, border_radius="md"),
            rx.text(
                State.snapshot_timestamp,
                color="gray.500",
                font_size="xs",
            ),
            spacing="1",
        ),
        rx.box(),  # empty if no snapshot yet
    )


def webcam_feed():
    """
    Left panel: Live webcam feed with dynamic status border.
    This is where we integrate the real webcam component (Phase 1).
    """
    return rx.box(
        rx.vstack(
            rx.hstack(
                rx.text(
                    "LIVE FEED - CAM 01",
                    color="white",
                    font_weight="bold",
                    font_size="sm",
                ),
                rx.spacer(),
                rx.badge("LIVE", color_scheme="red", variant="solid"),
                width="100%",
                padding="0.5em",
            ),
            # VIDEO CONTAINER
            rx.box(
                webcam.webcam(
                    id="admin_cam",
                    audio=False,
                    mirrored=False,
                ),
                # Dynamic border color based on posture status
                border=rx.cond(
                    State.status == "safe",
                    "4px solid #16a34a",  # Green
                    rx.cond(
                        State.status == "warning",
                        "4px solid #f97316",  # Orange
                        "6px solid #dc2626",  # Red (thicker)
                    ),
                ),
                box_shadow=rx.cond(
                    State.status == "critical",
                    "0 0 20px #dc2626",  # Glow on critical
                    "none",
                ),
                position="center",
                width="100%",
                border_radius="md",
                overflow="hidden",
            ),
            spacing="0",
            bg="black",
            border_radius="lg",
            overflow="hidden",
        ),
        width="100%",
        height="100%",
    )


def controls_panel():
    """Stream toggle + snapshot button."""
    return rx.hstack(
        rx.button(
            rx.cond(State.is_streaming, "Stop Stream", "Start Stream"),
            on_click=State.toggle_stream,
            color_scheme=rx.cond(State.is_streaming, "red", "green"),
            width="50%",
        ),
        rx.button(
            "Snapshot",
            color_scheme="gray",
            variant="outline",
            width="50%",
            # Phase 1: capture a single frame from the webcam
            on_click=webcam.upload_screenshot(
                webcam_id="admin_cam",
                handler=State.handle_snapshot,  # type: ignore
            ),
        ),
        width="100%",
        padding_y="1em",
    )


def debug_controls():
    """Buttons to manually trigger states for testing the UI."""
    return rx.hstack(
        rx.button(
            "Sim Safe",
            on_click=State.simulate_safe,
            size="1",
            color_scheme="green",
        ),
        rx.button(
            "Sim Warn",
            on_click=State.simulate_warning,
            size="1",
            color_scheme="orange",
        ),
        rx.button(
            "Sim Crit",
            on_click=State.simulate_critical,
            size="1",
            color_scheme="red",
        ),
        position="fixed",
        bottom="10px",
        left="10px",
        opacity="0.5",
    )


# --- MAIN PAGE LAYOUT ---


def index():
    return rx.box(
        # Top Header
        rx.hstack(
            rx.heading(
                "Real-Time Industrial Posture Monitoring",
                size="5",
                color="white",
            ),
            rx.spacer(),
            rx.text("Admin Dashboard", color="gray.400"),
            width="100%",
            padding="1em",
            border_bottom="1px solid #333",
            bg="gray.900",
        ),
        # Main Content Grid
        rx.hstack(
            # LEFT COLUMN (Video Feed) - Takes 70% width
            rx.box(
                webcam_feed(),
                width="70%",
                padding="2em",
            ),
            # RIGHT COLUMN (Status & Controls) - Takes 30% width
            rx.vstack(
                status_card(),
                risk_factors_panel(),
                controls_panel(),
                rx.divider(border_color="gray.700"),
                log_panel(),
                snapshot_preview(),
                width="30%",
                height="100%",
                bg="gray.800",
                padding="2em",
                border_left="1px solid #333",
            ),
            width="100%",
            height="calc(100vh - 70px)",  # Full height minus header
            align_items="stretch",
            spacing="0",
        ),
        debug_controls(),  # Remove this in production
        bg="gray.900",
        height="100vh",
        width="100vw",
    )


# --- APP CONFIG ---


app = rx.App(theme=rx.theme(appearance="dark"))
app.add_page(index, title="Industrial Safety Admin Dashboard")
