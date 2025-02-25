# Import libraries 
import streamlit as st 
import fastf1 as ff1
from datetime import datetime
from utils import * 

# Caching function to load and cache Formula 1 session data to improve performance
@st.cache_data
def load_session(year, gp, session_type):
    '''Load and cache a FastF1 session.'''
    session = ff1.get_session(year, gp, session_type)
    session.load()
    return session

# Main rendering function that generates the web page based on the session type
def render():
    gp = st.session_state['gp']  # Access the selected Grand Prix from the session state
    session_type = st.session_state['session']  # Access the session type (Qualifying or Race)

    # QUALIFYING 
    if session_type == 'Qualifying':
        with st.spinner('Unveiling strategy metrics...💻'):
            session = load_session(2024, gp, session_type)
            st.title('Qulifying Tyre Strategies')
            
            # Prepare session data
            sessions, all_drivers_sorted = prepare_sessions(session)

            # Display qualifying tyre life plot
            st.plotly_chart(qualifying_tyre_life(sessions, all_drivers_sorted)) 
            
            # Display air and track temperature plot during qualifying
            st.subheader(f'Air and Track Temperature during the Qualifying')
            weather_data = session.weather_data
            st.plotly_chart(plot_temperatures(weather_data))
            
    # RACE
    elif session_type == 'Race':
        with st.spinner(text='Extracting sector performance insights...🏎️'):
            session = load_session(2024, gp, session_type)

            # Display tyre strategies plot
            st.subheader('Tyre Strategies')
            stints = get_race_stints(session=session)
            drivers = get_drivers_list(session=session)
            st.plotly_chart(tyre_strategies(drivers, stints))
            
            # Display laptime distribution plot of the point finisher
            st.subheader('Laptime Distribution of the point Finisher')
            st.plotly_chart(driver_laptime_violinplot(session))
            
            # Display gained and lost places plot
            st.subheader('Gained and Lost Places', help='In the Plot you can see the Drivers which gained and lost most places during the Race')
            st.plotly_chart(analyze_driver_positions(session))

        # Display air and track temperature plot during the race
        st.subheader(f'Air and Track Temperature during the Race')
        weather_data = session.weather_data
        st.plotly_chart(plot_temperatures(weather_data))
        
    else:
        st.warning('This plot is only available for Qualifying or Race sessions.')