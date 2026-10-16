"""LAPE UI Dark and Light Theme Stylesheets"""
import streamlit as st

DARK_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

    :root {
        --bg-main: #0B0B0C;
        --bg-sidebar: #141416;
        --bg-panel: #1C1C1F;
        --bg-input: #242428;
        --border: #333338;
        --text-primary: #F2F2F3;
        --text-secondary: #A1A1AA;
        --accent: #2CB7B0;
        --accent-hover: #269f99;
        --warning: #D97706;
        --critical: #DC2626;
        --success: #10B981;
        --table-header-bg: #1C1C1F;
        --table-row-hover: #242428;
        --login-card-bg: #1C1C1F;
    }

    .stApp {
        background-color: var(--bg-main) !important;
        color: var(--text-primary) !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    }

    .main .block-container {
        padding-top: 2.2rem !important;
        padding-bottom: 2rem !important;
        max-width: 1320px;
    }

    /* Top header configuration - keep visible so collapse toggle works */
    header[data-testid="stHeader"] {
        background: transparent !important;
        visibility: visible !important;
        height: auto !important;
        z-index: 1000 !important;
    }

    /* Hide unnecessary menu & deploy button */
    #MainMenu, footer, .stDeployButton {
        display: none !important;
        visibility: hidden !important;
    }

    /* CRITICAL FIX: Sidebar Collapse & Expand Toggle controls must always be visible & clickable */
    [data-testid="collapsedControl"],
    button[data-testid="stSidebarExpandButton"],
    button[data-testid="stSidebarCollapseButton"] {
        display: flex !important;
        visibility: visible !important;
        opacity: 1 !important;
        pointer-events: auto !important;
        z-index: 999999 !important;
        color: var(--text-primary) !important;
        background-color: var(--bg-panel) !important;
        border: 1px solid var(--border) !important;
        border-radius: 6px !important;
    }
    [data-testid="collapsedControl"] svg,
    button[data-testid="stSidebarExpandButton"] svg,
    button[data-testid="stSidebarCollapseButton"] svg {
        fill: var(--text-primary) !important;
        color: var(--text-primary) !important;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: var(--bg-sidebar) !important;
        border-right: 1px solid var(--border) !important;
    }
    section[data-testid="stSidebar"] * {
        color: var(--text-secondary) !important;
    }

    /* Sidebar Brand */
    .sidebar-brand {
        padding: 8px 6px 16px 6px;
        border-bottom: 1px solid var(--border);
        margin-bottom: 14px;
    }
    .sidebar-logo {
        font-size: 24px;
        font-weight: 700;
        color: var(--accent);
        letter-spacing: -0.5px;
        line-height: 1.1;
    }
    .sidebar-brand-title {
        font-size: 13.5px;
        color: var(--text-secondary);
        margin-top: 4px;
        text-transform: uppercase;
        letter-spacing: 0.6px;
    }

    /* Sidebar Group Heading */
    .sidebar-group-heading {
        font-size: 13.5px !important;
        font-weight: 600 !important;
        color: var(--text-secondary) !important;
        text-transform: uppercase !important;
        letter-spacing: 0.8px !important;
        padding: 14px 6px 6px 6px !important;
    }

    /* Sidebar Enterprise Nav Buttons */
    section[data-testid="stSidebar"] .stButton > button {
        width: 100% !important;
        text-align: left !important;
        display: flex !important;
        align-items: center !important;
        justify-content: flex-start !important;
        padding: 10px 14px !important;
        margin-bottom: 4px !important;
        border-radius: 6px !important;
        border: none !important;
        border-left: 3px solid transparent !important;
        font-size: 16.5px !important;
        font-weight: 500 !important;
        background-color: transparent !important;
        color: var(--text-secondary) !important;
        box-shadow: none !important;
        transition: background-color 0.15s ease, border-color 0.15s ease !important;
    }
    section[data-testid="stSidebar"] .stButton > button:hover {
        background-color: var(--table-row-hover) !important;
        color: var(--text-primary) !important;
        border-left: 3px solid transparent !important;
    }
    section[data-testid="stSidebar"] .stButton > button[kind="primary"] {
        background-color: var(--bg-panel) !important;
        color: var(--text-primary) !important;
        border-left: 3px solid var(--accent) !important;
        font-weight: 600 !important;
        box-shadow: none !important;
    }

    /* Profile Card at bottom of sidebar */
    .user-profile-card {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 10px 12px;
        background: var(--bg-panel);
        border: 1px solid var(--border);
        border-radius: 6px;
        margin-top: 14px;
        margin-bottom: 8px;
    }
    .user-avatar {
        width: 32px;
        height: 32px;
        border-radius: 50%;
        background: var(--bg-input);
        color: var(--accent);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 12px;
        font-weight: 700;
        border: 1px solid var(--border);
    }
    .user-name {
        font-size: 13px;
        font-weight: 600;
        color: var(--text-primary);
        line-height: 1.2;
    }
    .user-sub {
        font-size: 11px;
        color: var(--text-secondary);
        display: flex;
        align-items: center;
        gap: 6px;
        margin-top: 2px;
    }
    .badge-demo {
        background: var(--bg-input);
        color: var(--warning);
        font-size: 9px;
        padding: 1px 5px;
        border-radius: 3px;
        font-weight: 700;
        letter-spacing: 0.4px;
        border: 1px solid var(--border);
    }

    /* Complete Form controls, widget labels, and text contrast */
    label,
    label[data-testid="stWidgetLabel"],
    label[data-testid="stWidgetLabel"] p,
    label[data-testid="stWidgetLabel"] span {
        color: var(--text-primary) !important;
        font-size: 13.5px !important;
        font-weight: 500 !important;
    }

    div[data-testid="stCheckbox"] label,
    div[data-testid="stCheckbox"] label span,
    div[data-testid="stCheckbox"] label p {
        color: var(--text-primary) !important;
        font-size: 13.5px !important;
    }

    div[data-testid="stRadio"] label,
    div[data-testid="stRadio"] label span,
    div[data-testid="stRadio"] label p,
    div[data-testid="stRadio"] div[role="radiogroup"] label {
        color: var(--text-primary) !important;
        font-size: 13.5px !important;
    }
    div[data-testid="stRadio"] label:has(input:checked) span,
    div[data-testid="stRadio"] label:has(input:checked) p {
        color: var(--accent) !important;
        font-weight: 600 !important;
    }

    .stTextInput input, .stNumberInput input, .stSelectbox select, .stTextArea textarea {
        background-color: var(--bg-input) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border) !important;
        border-radius: 6px !important;
        font-size: 14px !important;
    }
    .stTextInput input::placeholder,
    .stNumberInput input::placeholder,
    .stTextArea textarea::placeholder {
        color: var(--text-secondary) !important;
        opacity: 0.8 !important;
    }

    div[data-baseweb="select"] > div {
        background-color: var(--bg-input) !important;
        border: 1px solid var(--border) !important;
        border-radius: 6px !important;
        color: var(--text-primary) !important;
    }
    div[data-baseweb="select"] * {
        color: var(--text-primary) !important;
    }
    div[data-baseweb="popover"], div[data-baseweb="menu"], ul[role="listbox"] {
        background-color: var(--bg-panel) !important;
        border: 1px solid var(--border) !important;
    }
    li[role="option"] {
        color: var(--text-primary) !important;
        background-color: var(--bg-panel) !important;
    }
    li[role="option"]:hover, li[role="option"][aria-selected="true"] {
        background-color: var(--table-row-hover) !important;
        color: var(--accent) !important;
    }

    /* Buttons */
    .main .stButton > button,
    .main .stButton > button[kind="secondary"],
    .stButton > button:not([kind="primary"]) {
        background-color: var(--bg-input) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border) !important;
        border-radius: 6px !important;
        font-size: 13.5px !important;
        font-weight: 500 !important;
    }
    .main .stButton > button:hover,
    .main .stButton > button[kind="secondary"]:hover,
    .stButton > button:not([kind="primary"]):hover {
        background-color: var(--bg-panel) !important;
        border-color: var(--accent) !important;
        color: var(--accent) !important;
    }
    .main .stButton > button[kind="primary"], button[kind="primary"] {
        background-color: var(--accent) !important;
        color: #0B0B0C !important;
        border: none !important;
        font-weight: 600 !important;
        border-radius: 6px !important;
        font-size: 14px !important;
    }
    .main .stButton > button[kind="primary"]:hover {
        background-color: var(--accent-hover) !important;
    }

    /* General text and headings */
    h1, h2, h3, h4, h5, h6 {
        color: var(--text-primary) !important;
    }
    .stMarkdown p, .stMarkdown span {
        color: var(--text-primary);
    }
    .stCaption, div[data-testid="stCaptionContainer"] p, .stCaption p {
        color: var(--text-secondary) !important;
    }

    /* Tabs */
    button[data-baseweb="tab"] {
        color: var(--text-secondary) !important;
        font-size: 13.5px !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: var(--accent) !important;
        font-weight: 600 !important;
        border-bottom-color: var(--accent) !important;
    }

    /* Slider numbers & labels */
    div[data-testid="stSlider"] * {
        color: var(--text-primary) !important;
    }

    /* Solid panels - NO gradients, NO glassmorphism */
    .lape-panel {
        background-color: var(--bg-panel);
        border: 1px solid var(--border);
        border-radius: 6px;
        padding: 18px 20px;
        margin-bottom: 14px;
    }
    .lape-metric-label {
        font-size: 12.5px;
        color: var(--text-secondary);
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.6px;
    }
    .lape-metric-value {
        font-size: 28px;
        font-weight: 700;
        color: var(--text-primary);
        margin-top: 6px;
        line-height: 1.2;
        font-variant-numeric: tabular-nums;
    }
    .lape-metric-sub {
        font-size: 12px;
        color: var(--text-secondary);
        margin-top: 4px;
    }

    /* Tri-dimensional cards */
    .dim-exposure { background: var(--bg-panel); border: 1px solid var(--border); border-left: 3px solid var(--critical); border-radius: 6px; padding: 18px; }
    .dim-reliability { background: var(--bg-panel); border: 1px solid var(--border); border-left: 3px solid var(--accent); border-radius: 6px; padding: 18px; }
    .dim-urgency { background: var(--bg-panel); border: 1px solid var(--border); border-left: 3px solid var(--warning); border-radius: 6px; padding: 18px; }

    /* Clean Solid Tables with Tabular Numerals */
    .lape-table-wrap {
        background: var(--bg-panel);
        border: 1px solid var(--border);
        border-radius: 6px;
        overflow: hidden;
    }
    .lape-table {
        width: 100%; border-collapse: collapse; font-size: 13.5px; color: var(--text-primary);
        font-family: 'Inter', sans-serif;
    }
    .lape-table th {
        background: var(--table-header-bg); color: var(--accent); padding: 11px 14px;
        text-align: left; border-bottom: 1px solid var(--border);
        font-weight: 600; text-transform: uppercase; font-size: 11.5px; letter-spacing: 0.4px;
    }
    .lape-table td {
        padding: 9px 14px; border-bottom: 1px solid #242428; font-variant-numeric: tabular-nums;
    }
    .lape-table tr:hover { background: var(--table-row-hover); }
    .lape-table-footer { font-size: 11.5px; color: var(--text-secondary); padding: 7px 14px; }

    /* Information Callouts */
    .lape-info {
        background: var(--bg-panel); border: 1px solid var(--border); border-radius: 6px;
        padding: 13px 18px; font-size: 13.5px; color: var(--text-secondary); margin: 8px 0;
        line-height: 1.5;
    }
    .lape-info strong { color: var(--accent); }
    .lape-info code { background: var(--bg-input); color: var(--accent); padding: 2px 6px; border-radius: 4px; font-size: 12.5px; font-family: 'JetBrains Mono', monospace; }

    /* Compact Login Header & Wordmark */
    .login-brand-header {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 12px;
        margin: 8px 0 18px 0;
    }
    .login-square-logo {
        width: 36px;
        height: 36px;
        background-color: var(--accent);
        color: #0B0B0C;
        font-family: 'Inter', -apple-system, sans-serif;
        font-weight: 700;
        font-size: 20px;
        border-radius: 6px;
        display: flex;
        align-items: center;
        justify-content: center;
        line-height: 1;
        user-select: none;
    }
    .login-wordmark {
        font-family: 'Inter', -apple-system, sans-serif;
        font-size: 22px;
        font-weight: 700;
        color: var(--text-primary);
        letter-spacing: 0.5px;
        line-height: 1;
    }

    .captcha-box {
        background: var(--bg-input);
        border: 1px solid var(--border);
        border-radius: 6px;
        padding: 12px 14px;
        margin: 12px 0 16px 0;
    }
    .captcha-label {
        font-size: 12.5px;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 6px;
    }

    .demo-box {
        background: var(--bg-panel);
        border: 1px solid var(--border);
        border-radius: 6px;
        padding: 10px 14px;
        margin-top: 14px;
        font-size: 12px;
        color: var(--text-secondary);
        line-height: 1.4;
        text-align: center;
    }
    .demo-box strong { color: var(--text-primary); }
    .demo-box code {
        background: var(--bg-input);
        color: var(--accent);
        padding: 2px 6px;
        border-radius: 4px;
        font-size: 11.5px;
        font-family: 'JetBrains Mono', monospace;
    }

    /* Header Bar */
    .lape-header-bar {
        background: var(--bg-panel);
        border: 1px solid var(--border);
        border-radius: 6px;
        padding: 16px 22px;
        margin-bottom: 20px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .lape-header-title { font-size: 20px; font-weight: 700; color: var(--text-primary); }
    .lape-header-sub { font-size: 13px; color: var(--text-secondary); margin-top: 2px; }
    .lape-header-badge {
        background: var(--bg-input);
        color: var(--accent);
        border: 1px solid var(--border);
        padding: 4px 10px;
        border-radius: 4px;
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Comparison cards */
    .compare-card { background: var(--bg-panel); border: 1px solid var(--border); border-radius: 6px; padding: 20px; }
    .compare-card.winner { border-color: var(--success); }

    /* Tabular Font */
    .mono-val { font-family: 'JetBrains Mono', monospace; font-size: 13px; }
</style>
"""

LIGHT_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

    :root {
        --bg-main: #F8F9FA;
        --bg-sidebar: #F1F3F4;
        --bg-panel: #FFFFFF;
        --bg-input: #FFFFFF;
        --border: #DADCE0;
        --text-primary: #202124;
        --text-secondary: #5F6368;
        --accent: #1A73E8;
        --accent-hover: #1557B0;
        --warning: #E37400;
        --critical: #D93025;
        --success: #188038;
        --table-header-bg: #F1F3F4;
        --table-row-hover: #F8F9FA;
        --login-card-bg: #FFFFFF;
    }

    .stApp {
        background-color: var(--bg-main) !important;
        color: var(--text-primary) !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    }

    .main .block-container {
        padding-top: 2.2rem !important;
        padding-bottom: 2rem !important;
        max-width: 1320px;
    }

    /* Top header configuration - keep visible so collapse toggle works */
    header[data-testid="stHeader"] {
        background: transparent !important;
        visibility: visible !important;
        height: auto !important;
        z-index: 1000 !important;
    }

    /* Hide unnecessary menu & deploy button */
    #MainMenu, footer, .stDeployButton {
        display: none !important;
        visibility: hidden !important;
    }

    /* CRITICAL FIX: Sidebar Collapse & Expand Toggle controls must always be visible & clickable */
    [data-testid="collapsedControl"],
    button[data-testid="stSidebarExpandButton"],
    button[data-testid="stSidebarCollapseButton"] {
        display: flex !important;
        visibility: visible !important;
        opacity: 1 !important;
        pointer-events: auto !important;
        z-index: 999999 !important;
        color: var(--text-primary) !important;
        background-color: var(--bg-panel) !important;
        border: 1px solid var(--border) !important;
        border-radius: 6px !important;
    }
    [data-testid="collapsedControl"] svg,
    button[data-testid="stSidebarExpandButton"] svg,
    button[data-testid="stSidebarCollapseButton"] svg {
        fill: var(--text-primary) !important;
        color: var(--text-primary) !important;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: var(--bg-sidebar) !important;
        border-right: 1px solid var(--border) !important;
    }
    section[data-testid="stSidebar"] * {
        color: var(--text-secondary) !important;
    }

    /* Sidebar Brand */
    .sidebar-brand {
        padding: 8px 6px 16px 6px;
        border-bottom: 1px solid var(--border);
        margin-bottom: 14px;
    }
    .sidebar-logo {
        font-size: 24px;
        font-weight: 700;
        color: var(--accent);
        letter-spacing: -0.5px;
        line-height: 1.1;
    }
    .sidebar-brand-title {
        font-size: 13.5px;
        color: var(--text-secondary);
        margin-top: 4px;
        text-transform: uppercase;
        letter-spacing: 0.6px;
    }

    /* Sidebar Group Heading */
    .sidebar-group-heading {
        font-size: 13.5px !important;
        font-weight: 600 !important;
        color: var(--text-secondary) !important;
        text-transform: uppercase !important;
        letter-spacing: 0.8px !important;
        padding: 14px 6px 6px 6px !important;
    }

    /* Sidebar Enterprise Nav Buttons */
    section[data-testid="stSidebar"] .stButton > button {
        width: 100% !important;
        text-align: left !important;
        display: flex !important;
        align-items: center !important;
        justify-content: flex-start !important;
        padding: 10px 14px !important;
        margin-bottom: 4px !important;
        border-radius: 6px !important;
        border: none !important;
        border-left: 3px solid transparent !important;
        font-size: 16.5px !important;
        font-weight: 500 !important;
        background-color: transparent !important;
        color: var(--text-secondary) !important;
        box-shadow: none !important;
        transition: background-color 0.15s ease, border-color 0.15s ease !important;
    }
    section[data-testid="stSidebar"] .stButton > button:hover {
        background-color: var(--table-row-hover) !important;
        color: var(--text-primary) !important;
        border-left: 3px solid transparent !important;
    }
    section[data-testid="stSidebar"] .stButton > button[kind="primary"] {
        background-color: var(--bg-panel) !important;
        color: var(--text-primary) !important;
        border-left: 3px solid var(--accent) !important;
        font-weight: 600 !important;
        box-shadow: none !important;
    }

    /* Profile Card at bottom of sidebar */
    .user-profile-card {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 10px 12px;
        background: var(--bg-panel);
        border: 1px solid var(--border);
        border-radius: 6px;
        margin-top: 14px;
        margin-bottom: 8px;
    }
    .user-avatar {
        width: 32px;
        height: 32px;
        border-radius: 50%;
        background: var(--bg-input);
        color: var(--accent);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 12px;
        font-weight: 700;
        border: 1px solid var(--border);
    }
    .user-name {
        font-size: 13px;
        font-weight: 600;
        color: var(--text-primary);
        line-height: 1.2;
    }
    .user-sub {
        font-size: 11px;
        color: var(--text-secondary);
        display: flex;
        align-items: center;
        gap: 6px;
        margin-top: 2px;
    }
    .badge-demo {
        background: var(--bg-sidebar);
        color: var(--warning);
        font-size: 9px;
        padding: 1px 5px;
        border-radius: 3px;
        font-weight: 700;
        letter-spacing: 0.4px;
        border: 1px solid var(--border);
    }

    /* Complete Form controls, widget labels, and text contrast */
    label,
    label[data-testid="stWidgetLabel"],
    label[data-testid="stWidgetLabel"] p,
    label[data-testid="stWidgetLabel"] span {
        color: var(--text-primary) !important;
        font-size: 13.5px !important;
        font-weight: 500 !important;
    }

    div[data-testid="stCheckbox"] label,
    div[data-testid="stCheckbox"] label span,
    div[data-testid="stCheckbox"] label p {
        color: var(--text-primary) !important;
        font-size: 13.5px !important;
    }

    div[data-testid="stRadio"] label,
    div[data-testid="stRadio"] label span,
    div[data-testid="stRadio"] label p,
    div[data-testid="stRadio"] div[role="radiogroup"] label {
        color: var(--text-primary) !important;
        font-size: 13.5px !important;
    }
    div[data-testid="stRadio"] label:has(input:checked) span,
    div[data-testid="stRadio"] label:has(input:checked) p {
        color: var(--accent) !important;
        font-weight: 600 !important;
    }

    .stTextInput input, .stNumberInput input, .stSelectbox select, .stTextArea textarea {
        background-color: var(--bg-input) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border) !important;
        border-radius: 6px !important;
        font-size: 14px !important;
    }
    .stTextInput input::placeholder,
    .stNumberInput input::placeholder,
    .stTextArea textarea::placeholder {
        color: var(--text-secondary) !important;
        opacity: 0.8 !important;
    }

    div[data-baseweb="select"] > div {
        background-color: var(--bg-input) !important;
        border: 1px solid var(--border) !important;
        border-radius: 6px !important;
        color: var(--text-primary) !important;
    }
    div[data-baseweb="select"] * {
        color: var(--text-primary) !important;
    }
    div[data-baseweb="popover"], div[data-baseweb="menu"], ul[role="listbox"] {
        background-color: var(--bg-panel) !important;
        border: 1px solid var(--border) !important;
    }
    li[role="option"] {
        color: var(--text-primary) !important;
        background-color: var(--bg-panel) !important;
    }
    li[role="option"]:hover, li[role="option"][aria-selected="true"] {
        background-color: var(--table-row-hover) !important;
        color: var(--accent) !important;
    }

    /* Buttons */
    .main .stButton > button,
    .main .stButton > button[kind="secondary"],
    .stButton > button:not([kind="primary"]) {
        background-color: #FFFFFF !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border) !important;
        border-radius: 6px !important;
        font-size: 13.5px !important;
        font-weight: 500 !important;
    }
    .main .stButton > button:hover,
    .main .stButton > button[kind="secondary"]:hover,
    .stButton > button:not([kind="primary"]):hover {
        background-color: var(--bg-sidebar) !important;
        border-color: var(--accent) !important;
        color: var(--accent) !important;
    }
    .main .stButton > button[kind="primary"], button[kind="primary"] {
        background-color: var(--accent) !important;
        color: #FFFFFF !important;
        border: none !important;
        font-weight: 600 !important;
        border-radius: 6px !important;
        font-size: 14px !important;
    }
    .main .stButton > button[kind="primary"]:hover {
        background-color: var(--accent-hover) !important;
    }

    /* General text and headings */
    h1, h2, h3, h4, h5, h6 {
        color: var(--text-primary) !important;
    }
    .stMarkdown p, .stMarkdown span {
        color: var(--text-primary);
    }
    .stCaption, div[data-testid="stCaptionContainer"] p, .stCaption p {
        color: var(--text-secondary) !important;
    }

    /* Tabs */
    button[data-baseweb="tab"] {
        color: var(--text-secondary) !important;
        font-size: 13.5px !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: var(--accent) !important;
        font-weight: 600 !important;
        border-bottom-color: var(--accent) !important;
    }

    /* Slider numbers & labels */
    div[data-testid="stSlider"] * {
        color: var(--text-primary) !important;
    }

    /* Solid panels - NO gradients, NO glassmorphism */
    .lape-panel {
        background-color: var(--bg-panel);
        border: 1px solid var(--border);
        border-radius: 6px;
        padding: 18px 20px;
        margin-bottom: 14px;
    }
    .lape-metric-label {
        font-size: 12.5px;
        color: var(--text-secondary);
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.6px;
    }
    .lape-metric-value {
        font-size: 28px;
        font-weight: 700;
        color: var(--text-primary);
        margin-top: 6px;
        line-height: 1.2;
        font-variant-numeric: tabular-nums;
    }
    .lape-metric-sub {
        font-size: 12px;
        color: var(--text-secondary);
        margin-top: 4px;
    }

    /* Tri-dimensional cards */
    .dim-exposure { background: var(--bg-panel); border: 1px solid var(--border); border-left: 3px solid var(--critical); border-radius: 6px; padding: 18px; }
    .dim-reliability { background: var(--bg-panel); border: 1px solid var(--border); border-left: 3px solid var(--accent); border-radius: 6px; padding: 18px; }
    .dim-urgency { background: var(--bg-panel); border: 1px solid var(--border); border-left: 3px solid var(--warning); border-radius: 6px; padding: 18px; }

    /* Clean Solid Tables with Tabular Numerals */
    .lape-table-wrap {
        background: var(--bg-panel);
        border: 1px solid var(--border);
        border-radius: 6px;
        overflow: hidden;
    }
    .lape-table {
        width: 100%; border-collapse: collapse; font-size: 13.5px; color: var(--text-primary);
        font-family: 'Inter', sans-serif;
    }
    .lape-table th {
        background: var(--table-header-bg); color: var(--accent); padding: 11px 14px;
        text-align: left; border-bottom: 1px solid var(--border);
        font-weight: 600; text-transform: uppercase; font-size: 11.5px; letter-spacing: 0.4px;
    }
    .lape-table td {
        padding: 9px 14px; border-bottom: 1px solid #EEEEEE; font-variant-numeric: tabular-nums;
    }
    .lape-table tr:hover { background: var(--table-row-hover); }
    .lape-table-footer { font-size: 11.5px; color: var(--text-secondary); padding: 7px 14px; }

    /* Information Callouts */
    .lape-info {
        background: var(--bg-panel); border: 1px solid var(--border); border-radius: 6px;
        padding: 13px 18px; font-size: 13.5px; color: var(--text-secondary); margin: 8px 0;
        line-height: 1.5;
    }
    .lape-info strong { color: var(--accent); }
    .lape-info code { background: #F1F3F4; color: var(--accent); padding: 2px 6px; border-radius: 4px; font-size: 12.5px; font-family: 'JetBrains Mono', monospace; }

    /* Compact Login Header & Wordmark */
    .login-brand-header {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 12px;
        margin: 8px 0 18px 0;
    }
    .login-square-logo {
        width: 36px;
        height: 36px;
        background-color: var(--accent);
        color: #FFFFFF;
        font-family: 'Inter', -apple-system, sans-serif;
        font-weight: 700;
        font-size: 20px;
        border-radius: 6px;
        display: flex;
        align-items: center;
        justify-content: center;
        line-height: 1;
        user-select: none;
    }
    .login-wordmark {
        font-family: 'Inter', -apple-system, sans-serif;
        font-size: 22px;
        font-weight: 700;
        color: var(--text-primary);
        letter-spacing: 0.5px;
        line-height: 1;
    }

    .captcha-box {
        background: var(--bg-input);
        border: 1px solid var(--border);
        border-radius: 6px;
        padding: 12px 14px;
        margin: 12px 0 16px 0;
    }
    .captcha-label {
        font-size: 12.5px;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 6px;
    }

    .demo-box {
        background: var(--bg-sidebar);
        border: 1px solid var(--border);
        border-radius: 6px;
        padding: 10px 14px;
        margin-top: 14px;
        font-size: 12px;
        color: var(--text-secondary);
        line-height: 1.4;
        text-align: center;
    }
    .demo-box strong { color: var(--text-primary); }
    .demo-box code {
        background: #FFFFFF;
        color: var(--accent);
        padding: 2px 6px;
        border-radius: 4px;
        font-size: 11.5px;
        font-family: 'JetBrains Mono', monospace;
        border: 1px solid var(--border);
    }

    /* Header Bar */
    .lape-header-bar {
        background: var(--bg-panel);
        border: 1px solid var(--border);
        border-radius: 6px;
        padding: 16px 22px;
        margin-bottom: 20px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .lape-header-title { font-size: 20px; font-weight: 700; color: var(--text-primary); }
    .lape-header-sub { font-size: 13px; color: var(--text-secondary); margin-top: 2px; }
    .lape-header-badge {
        background: var(--bg-sidebar);
        color: var(--accent);
        border: 1px solid var(--border);
        padding: 4px 10px;
        border-radius: 4px;
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Comparison cards */
    .compare-card { background: var(--bg-panel); border: 1px solid var(--border); border-radius: 6px; padding: 20px; }
    .compare-card.winner { border-color: var(--success); }

    /* Tabular Font */
    .mono-val { font-family: 'JetBrains Mono', monospace; font-size: 13px; }
</style>
"""

def apply_theme():
    """Apply the active stylesheet."""
    if st.session_state.get("theme", "dark") == "light":
        st.markdown(LIGHT_CSS, unsafe_allow_html=True)
    else:
        st.markdown(DARK_CSS, unsafe_allow_html=True)
