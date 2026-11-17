"""Textual theme and stylesheet for Orion AI."""

from __future__ import annotations

from textual.theme import Theme

ORION_THEME = Theme(
    name="orion",
    primary="#a78bfa",
    secondary="#22d3ee",
    accent="#f472b6",
    foreground="#e5e7eb",
    background="#08080c",
    success="#34d399",
    warning="#fbbf24",
    error="#f87171",
    surface="#101018",
    panel="#15151f",
    dark=True,
)

ORION_CSS = """
Screen {
    background: #08080c;
    color: #e5e7eb;
}

#topbar {
    height: 3;
    padding: 0 2;
    background: #0d0d14;
    border-bottom: solid #1e1e2b;
    content-align: left middle;
}

#body {
    height: 1fr;
}

#main {
    width: 1fr;
    height: 1fr;
}

#messages {
    height: 1fr;
    padding: 1 2;
    scrollbar-size-vertical: 1;
    scrollbar-background: #0d0d14;
    scrollbar-color: #2a2a3a;
}

#sidebar {
    width: 38;
    height: 1fr;
    padding: 1 1;
    background: #0b0b11;
    border-left: solid #1e1e2b;
}

#sidebar-content {
    height: auto;
    width: 100%;
}

#statusbar {
    height: 1;
    padding: 0 2;
    background: #08080c;
    color: #6b7280;
}

#inputbar {
    height: auto;
    padding: 0 2 1 2;
    background: #08080c;
    border-top: solid #1e1e2b;
}

#mode-chip {
    width: auto;
    min-width: 11;
    height: 3;
    padding: 0 1;
    margin: 0 1 0 0;
    content-align: center middle;
    text-style: bold;
}

.mode-build {
    background: #1e1b4b;
    color: #c4b5fd;
    border: round #7c3aed;
}

.mode-plan {
    background: #0c4a6e;
    color: #bae6fd;
    border: round #0284c7;
}

#input {
    width: 1fr;
    height: 3;
    padding: 0 1;
    background: #101018;
    border: round #262635;
    color: #e5e7eb;
}

#input:focus {
    border: round #a78bfa;
    background: #12121c;
}

#input-hint {
    width: auto;
    height: 3;
    padding: 0 1;
    content-align: center middle;
    color: #4b5563;
}

MessageView {
    height: auto;
    margin-bottom: 1;
}

.msg-header {
    height: 1;
    padding: 0 1;
    text-style: bold;
    color: #a78bfa;
}

.msg-user .msg-header {
    color: #22d3ee;
}

.msg-assistant .msg-header {
    color: #a78bfa;
}

.msg-orion .msg-header {
    color: #f472b6;
}

.msg-body {
    height: auto;
    padding: 0 0 0 2;
}

.markdown-body {
    height: auto;
    padding: 0 0 0 2;
}

.markdown-body MarkdownBlock {
    margin: 0;
}

.note {
    height: auto;
    padding: 0 1;
    margin: 0 0 1 2;
    color: #9ca3af;
}

.note-warning {
    color: #fbbf24;
}

.note-error {
    color: #f87171;
}

.note-success {
    color: #34d399;
}

.welcome-logo {
    height: auto;
    padding: 0 1;
    color: #a78bfa;
}

.welcome {
    height: auto;
    padding: 0 1;
    margin-bottom: 1;
    color: #9ca3af;
}

.plan-view, .review-view, .test-view, .diff-view {
    height: auto;
    margin: 0 0 1 1;
    padding: 0 1;
    border: round #2a2a3a;
    background: #0d0d14;
}

.view-title {
    height: 1;
    text-style: bold;
    color: #a78bfa;
}

.plan-view .view-title {
    color: #e879f9;
}

.review-approve .view-title {
    color: #34d399;
}

.review-changes .view-title {
    color: #f87171;
}

.test-passed .view-title {
    color: #34d399;
}

.test-failed .view-title {
    color: #f87171;
}

.test-skipped .view-title {
    color: #fbbf24;
}

.diff-view .view-title {
    color: #22d3ee;
}

ToolCallView {
    height: auto;
    margin: 0 0 1 1;
    background: #0d0d14;
    border: round #22222f;
}

ToolCallView > .collapsible--title {
    color: #8b8b9e;
}

.tool-body {
    height: auto;
    padding: 0 1;
    color: #9ca3af;
}

#footer {
    background: #0d0d14;
    color: #6b7280;
}

ApprovalModal, CommandPalette {
    align: center middle;
    background: #000000 70%;
}

#approval-dialog, #palette-dialog {
    width: 82%;
    max-width: 110;
    height: auto;
    max-height: 80%;
    background: #101018;
    border: round #a78bfa;
    padding: 1 2;
}

.modal-title {
    height: 1;
    text-style: bold;
    color: #fbbf24;
    margin-bottom: 1;
}

.modal-detail {
    height: auto;
    max-height: 16;
    color: #d1d5db;
}

.modal-actions {
    height: auto;
    margin-top: 1;
    align-horizontal: right;
}

.modal-actions Button {
    margin-left: 1;
}

#palette-input {
    width: 100%;
    height: 3;
    background: #15151f;
    border: round #2a2a3a;
    margin-bottom: 1;
}

#palette-input:focus {
    border: round #22d3ee;
}

#palette-list {
    width: 100%;
    height: auto;
    max-height: 18;
    background: transparent;
}

#palette-list > .option-list--option {
    padding: 0 1;
}

#palette-list > .option-list--option-highlighted {
    background: #1e1b4b;
    color: #e9d5ff;
}

#sidebar .sidebar-title {
    color: #6b7280;
}
"""


__all__ = ["ORION_CSS", "ORION_THEME"]
