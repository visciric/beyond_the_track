# Import necessary libraries for data handling and visualization
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

    all_metrics = ["Speed", "Throttle", "Brake",] 

    # QUALIFYING    
    if session_type == 'Qualifying':
        with st.spinner('Tyres are beeing changed...🛞'):
            session = load_session(datetime.now().year, gp, session_type)
            circuit_info = session.get_circuit_info()
            st.title('Drivers Comparison')

            # Multi-selection for choosing drivers to compare
            driver_list = [session.get_driver(i)['Abbreviation'] for i in session.drivers]
            selected_drivers = st.multiselect('Select two drivers:', driver_list, help = '')

            # Compare the selected drivers if two are chosen
            if len(selected_drivers) == 2:
                
                 # Add Columns
                col1, col2 = st.columns([1.9,1]) # set the column ratio
                
                with col1:
                    # Data preparation and plotting for the selected drivers                
                    st.subheader('Telemetry Comparison')
                    fastest_driver1 = session.laps.pick_driver(selected_drivers[0]).pick_fastest()
                    fastest_driver2 = session.laps.pick_driver(selected_drivers[1]).pick_fastest()
                    df1 = fastest_driver1.get_car_data().add_distance()
                    df2 = fastest_driver2.get_car_data().add_distance()
                    
                    st.plotly_chart(telemetry_quali_plots(df1, df2, selected_drivers[0], selected_drivers[1], all_metrics, circuit_info))
                                    
                with col2:
                    # Plot the Track with the Speed Differences 
                    st.plotly_chart(driver_comparison_track_plot(session, selected_drivers))
                
                    
            else:
                st.warning('Please select exactly two drivers.')
             
    # RACE   
    elif session_type == 'Race':
        
        with st.spinner(text='Race data is loading...'):
            session = load_session(datetime.now().year, gp, session_type)
            circuit_info = session.get_circuit_info()
        st.title('Comparing Drivers')

        # Multi-selection for choosing drivers to compare
        driver_list = [session.get_driver(i)['Abbreviation'] for i in session.drivers]
        selected_drivers = st.multiselect('Select drivers:', driver_list, help = "For optimal visibility, it's best to select a maximum of two drivers")
        
        # Get available lap numbers for each selected driver
        if selected_drivers:
            laps_dict = {}
            for driver in selected_drivers:
                laps = session.laps.pick_driver(driver)
                # Convert lap numbers to integer
                laps_dict[driver] = [int(lap) for lap in laps['LapNumber'].unique().tolist()]

            # Select laps for each driver
            lap_selections = {}
            for driver in selected_drivers:
                lap_numbers = st.multiselect(f"Select laps for {driver}:", laps_dict[driver], help = 'Default: Displays the fastest lap of each driver if no lap is selected.')
                lap_selections[driver] = lap_numbers or [int(session.laps.pick_driver(driver).pick_fastest()['LapNumber'])]



            # Plotting telemetry for selected laps
            if len(selected_drivers) >= 1:
                telemetry_plots = []

                # Collect telemetry data for each selected lap and driver
                for driver in lap_selections:
                    for lap_number in lap_selections[driver]:
                        lap_data = session.laps.pick_driver(driver).loc[session.laps['LapNumber'] == lap_number].get_telemetry().add_distance()
                        telemetry_plots.append((lap_data, driver, lap_number))

                if len(telemetry_plots) > 0:
                    
                    col1, col2 = st.columns([2,.85]) # set the column ratio
                    with col2:
                        
                        circuit_info = session.get_circuit_info()
                        st.plotly_chart(race_track_plot(session, circuit_info))
                    
                    with col1:
                        
                        st.plotly_chart(telemetry_race_plots(telemetry_plots, all_metrics, circuit_info))
                else:
                    st.write("No data selected for plotting.")

            else: 
                st.info('Please select your favourite Drivers 😉')
        
    else:
        st.warning('This plot is only available for Qualifying or Race sessions.')
        
        
        
#######
# Dash Implementation that we used to try but didn't work out properly 
# we would've needed more Time and knowledge about Dash to be able to implement it correctly 

                # track_telemetry_x = fastest_driver1.telemetry['X']
                # track_telemetry_y = fastest_driver1.telemetry['Y']
                # track_angle = circuit_info.rotation / 180 * np.pi
                # points_track = rotate(
                #     np.array([track_telemetry_x, track_telemetry_y]).T,
                #     angle=track_angle,
                # )

                # df_list = [fastest_driver1.telemetry, fastest_driver2.telemetry]
                # labels = [selected_drivers[0], selected_drivers[1]]
                # speed_plot = create_speed_plot(df_list, 'Distance', 'Speed', labels)

                # # Initialize and run the Dash app in a background thread
                # def run_dash():
                #     app = setup_dash_app(points_track, speed_plot)
                #     app.run_server(port=8059, debug=True, use_reloader=False)

                # threading.Thread(target=run_dash, daemon=True).start()

                # # Streamlit page setup
                # st.title("Vehicle Data Analysis")
                # st.components.v1.iframe("http://localhost:8059", height=1200)
######