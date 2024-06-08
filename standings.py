# Import  libraries 
import streamlit as st
import fastf1 as ff1
from datetime import datetime
from utils import *

# Page Title
st.title('Standings')

# Caching function to load and cache Formula 1 session data to improve performance
@st.cache_data
def load_session(year, gp, session_type):
    '''Load and cache a FastF1 session.'''
    session = ff1.get_session(year, gp, session_type)
    session.load()
    return session

# Main rendering function that generates the streamlit page based on the session type
def render():
    gp = st.session_state['gp']  # Access the selected Grand Prix from the session state
    session_type = st.session_state['session']  # Access the session type (Qualifying or Race)

    # QUALIFYING 
    if session_type == 'Qualifying':
        with st.spinner('Calculating lap times...⏲️'):
            st.subheader('Fastest Lap Time of Each Driver Over Sessions (Q1 to Q3)')

            col1, col2 = st.columns([.9,1])
            with col1:
                

                session = load_session(datetime.now().year, gp, session_type)
                st.table(qualifying_results(session))
                
            with col2:
                st.plotly_chart(qualifying_standing_plot(session))

    # RACE
    elif session_type == 'Race':
        with st.spinner(text='Almost there! Finalizing the latest standings...🔄'):
            session = load_session(datetime.now().year, gp, session_type)
            round_number = session.event['RoundNumber']


            tabs = st.tabs(['Drivers', 'Teams'])
            
            with tabs[0]:
                st.subheader('Driver Standings')
                st.table(driver_standings(2024, round_number))
                
            with tabs[1]:
                st.subheader('Team Standings')
                st.table(constructor_standings(2024, round_number))

            # Handle Further analysis   
            on = st.toggle('Further Standings Information', value=False, help='If toggled you will get more interesting data about the standings 😊')
            if on:

                st.subheader('Standings')
                tabs = st.tabs(['Drivers', 'Teams'])
                with tabs[0]:
                    st.subheader('Driver Standings through the 2024 Season')
                    new_pivot, races = get_season_results(2024, round_number)
                    st.plotly_chart(race_standings_plot(new_pivot, races))
                with tabs[1]:
                    st.subheader('Team Standings through the 2024 Season')
                    new_pivot_constructor, races_constructor = get_constructor_season_results(2024, round_number)
                    st.plotly_chart(constructors_standings(new_pivot_constructor, races_constructor))

    else:
        st.error('Invalid session type. Please select either "Qualifying" or "Race".')