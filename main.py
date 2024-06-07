# Import necessary libraries and modules
import streamlit as st
from datetime import datetime
from streamlit_extras.stoggle import stoggle
from utils import *

# Set the layout of the Streamlit page to 'wide' for better visualization display
st.set_page_config(
    layout="wide",
    page_title="PODSV Project FS 2024",
    page_icon="🏎️",
    menu_items={
        "About": "# Beyond the Track is a School Project \n ## HAVE FUN! 🏎️💨"
    },
)

# Path to the logo image
logo_path = "F1_logo.webp"

# Load the image and convert to base64
logo_base64 = load_logo(logo_path)

# App: ChatGPT Prompt for the Logo
# Create a black and white logo featuring a sleek, stylized Formula 1 race car.
# The car should have a dynamic and streamlined design, emphasizing speed and performance.
# The overall look should be modern and clean, suitable for use as an official logo.
# The background should be black, making the white car stand out clearly.
# Ensure that the details of the car, such as the tires and aerodynamics, are clear and visually appealing.

# Insert the logo with HTML and CSS
st.markdown(
    f"""
    <style>
    .header-container {{
        display: flex;
        align-items: center;
        justify-content: space-between;
    }}
    .header-container img {{
        width: 350px;
        height: auto;
        order: 2;
        margin-left: 20px; # margin-right
    }}
    .header-container div {{
        order: 1;
    }}
    </style>
    <div class="header-container">
        <div>
            <h1>Beyond the Track</h1>
            <h2>Decoding the Strategy and Teamwork of Formula 1</h2>
        </div>
        <img src="data:image/webp;base64,{logo_base64}" alt="Logo">
    </div>
    """,
    unsafe_allow_html=True,
)

# Add a toggle to show or hide the detailed introduction
stoggle(
    "Show detailed introduction",
    """
Welcome to "Beyond the Track," a comprehensive dashboard designed to delve deep into the intricate world of Formula 1 racing. Here, we uncover the layers of strategy, teamwork, and data analytics that drive the success of teams and drivers on the world stage.

**Explore various aspects of F1 racing:**
- **Qualifying Dynamics**: Understand how drivers and their teams perform under the pressure of Qualifying rounds.
- **Race Performance**: Dive into detailed race data to see how strategies unfold on the track.
- **Driver Comparisons**: Compare your favorite drivers through various metrics and visualizations.
- **Team Tactics**: Gain insights into how teams manage resources, including tyre strategies in order to optimize their race outcomes.

This tool aims to provide fans, analysts, and enthusiasts with a deeper understanding of the often unseen aspects of Formula 1 racing, enhancing your appreciation for this high-octane sport. Use the tabs to navigate between different data views and uncover hidden patterns that influence race day decisions and outcomes.
""",
)


# Get the current year to fetch relevant F1 event data
current_year = datetime.now().year

# Initialize or access a session state variable for user interaction modes
if "mode" not in st.session_state:
    st.session_state["mode"] = "Overview"

# Create three columns for layout, with the middle one being wider
col1, col2, col3 = st.columns([1, 3, 1])

# Fetch past F1 events for the current year and create selection options for the user
available_gps = past_events(current_year)
available_sessions = ["Qualifying", "Race"]

# Ensure session state has valid selections for GP and Session
if "gp" not in st.session_state or st.session_state["gp"] not in available_gps:
    st.session_state["gp"] = (
        available_gps[-1] if available_gps else None
    )  # Default to the last event

if "session" not in st.session_state:
    st.session_state["session"] = available_sessions[-1]  # Default to the last option

# Use two columns to display dropdowns for selecting GP and Session neatly next to each other
col_gp, col_session = st.columns([2, 2])
with col_gp:
    st.session_state["gp"] = st.selectbox(
        "GP", available_gps, index=available_gps.index(st.session_state["gp"])
    )
with col_session:
    st.session_state["session"] = st.selectbox(
        "Session",
        available_sessions,
        index=available_sessions.index(st.session_state["session"]),
    )

# Organize content into tabs for better navigation
tabs = st.tabs(["Overview🏁", "Telemetry📊", "Standings🏆", "In Depth Analysis🔍"])

# Import and use render functions from other modules based on the selected tab
with tabs[0]:
    import overview

    overview.render()

with tabs[1]:
    import telemetry

    telemetry.render()

with tabs[2]:
    import standings

    standings.render()

with tabs[3]:
    import in_depth

    in_depth.render()

# Custom CSS to adjust the font size within the tabs for better readability
css = """
<style>
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
    font-size:1.5rem;
    }
</style>
"""

# Apply the custom CSS to the app
st.markdown(css, unsafe_allow_html=True)
