# Import necessary Python libraries
import streamlit as st
import fastf1 as ff1
from datetime import datetime
from utils import *

# Cache data using Streamlit's caching to avoid reloading it upon every script execution, improving performance
@st.cache_data
def load_session(year, gp, session_type):
    """Load and cache a FastF1 session."""
    session = ff1.get_session(year, gp, session_type)
    session.load()
    return session

# Define the main rendering logic for the Streamlit app
def render():
    gp = st.session_state["gp"]  # Access the selected Grand Prix from session state
    session_type = st.session_state["session"]  # Access the session type (Qualifying or Race)

    # QUALIFYING
    if session_type == "Qualifying":
        with st.spinner(text="Preparing Qualifying Data... 📊"):
            session = load_session(datetime.now().year, gp, session_type)
            circuit_info = session.get_circuit_info()
            quali_results = get_quali_ranking(session)
            fastest_laptime = session.laps.pick_fastest()
            corners = circuit_info.corners["Number"].iloc[-1]
            round_number = session.event['RoundNumber']
            avg_trackspeed = get_avg_trackspeed(session)

            # Layout setup for Streamlit columns
            col1, col4 = st.columns([5.6, 1.6])
            with col1:
                st.header(session.event.EventName)
                st.plotly_chart(plot_track(session, circuit_info))

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(
                        f"<div style='font-size:22px; color:#3da3ff;'><b>🛣️ The Track has {corners} corners</b></div>",
                        unsafe_allow_html=True,
                    )
                with col2:
                    st.markdown(
                        f"<div style='font-size:22px; color:#3da3ff;'><b>🏎️ Average Speed: {avg_trackspeed} km/h</b></div>",
                        unsafe_allow_html=True,
                    )
                with col3:
                    st.markdown(
                        f"<div style='font-size:22px; color:#3da3ff;'><b>⏲️ Fastest Lap Time<br>{fastest_laptime.Driver} :  {format_timedelta(fastest_laptime.LapTime)}</b></div>",
                        unsafe_allow_html=True,
                    )

            # Display formatted lap times in another column
            with col4:
                quali_results = get_quali_results(session)
                st.table(quali_results)

    # RACE
    elif session_type == "Race":
        with st.spinner(text="Preparing Racing Data... 📊"):
            session = load_session(datetime.now().year, gp, session_type)
            circuit_info = session.get_circuit_info()
            round_number = session.event['RoundNumber']
            # round_number = round_number-1
            avg_trackspeed = get_avg_trackspeed(session)
            fastest_laptime = session.laps.pick_fastest()
            corners = circuit_info.corners["Number"].iloc[-1]

            # Layout setup similar to Qualifying
            col1, col4 = st.columns([5, 2])
            with col1:
                st.header(session.event.EventName)
                st.plotly_chart(plot_track(session, circuit_info))

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(
                        f"<div style='font-size:22px; color:#3da3ff;'><b>🛣️ The Track has {corners} corners</b></div>",
                        unsafe_allow_html=True,
                    )
                with col2:
                    st.markdown(
                        f"<div style='font-size:22px; color:#3da3ff;'><b>🏎️ Average Speed: {avg_trackspeed} km/h</b></div>",
                        unsafe_allow_html=True,
                    )
                with col3:
                    st.markdown(
                        f"<div style='font-size:22px; color:#3da3ff;'><b>⏲️ Fastest Lap Time<br>{fastest_laptime.Driver} :  {format_timedelta(fastest_laptime.LapTime)}</b></div>",
                        unsafe_allow_html=True,
                    )

            # Display race results in another column
            with col4:
                race_results = get_race_results_df(round_number)
                final_results = race_results[['Position', 'Driver', 'Constructor', 'Time']]
                st.markdown(final_results.to_markdown(index=False), unsafe_allow_html=True)

    else:
        st.error("No data available for the selected session type. Please select either 'Qualifying' or 'Race'.")
