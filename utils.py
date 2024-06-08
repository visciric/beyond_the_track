# Import all Packages
import base64
import numpy as np
import pandas as pd
import fastf1 as ff1
import plotly.express as px
from datetime import datetime
from fastf1.ergast import Ergast
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import fastf1.plotting

# GENERAL FUNCTIONS

def load_session(year, gp, session_type):
    """Load and cache a FastF1 session."""
    session = ff1.get_session(year, gp, session_type)
    session.load()
    return session

# Get Past events 
def past_events(year):
    today = datetime.today()
    
    # Retrieve the event schedule for the actual year
    schedule = ff1.get_event_schedule(year)

    # Filter past events (excluding pre-season testing) based on the event date
    past_events = schedule[(schedule["EventDate"] <= today) & (schedule["EventName"] != "Pre-Season Testing")]
    return list(past_events["EventName"].unique())


# Function to load the Logo and convert it to base64
def load_logo(image_path):
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")
    return base64_image


def apply_zoom(min_val, max_val, zoom_factor=0.8):
    """
    Apply a zoom factor to axis range.
    Is used to enhance the visability of the Track"""
    # Calculate the center of the range
    center = (min_val + max_val) / 2
    # Calculate the span of the range with the zoom factor applied
    range_span = (max_val - min_val) * zoom_factor / 2
    # Return the new min and max values after applying the zoom
    return center - range_span, center + range_span


def rotate(xy, *, angle):
    """
    This function is used to rotate the Race Track around the origin by a given angle.
    It's used for the right alignment

    """
    # Create the rotation matrix using the given angle
    rot_mat = np.array(
        [[np.cos(angle), np.sin(angle)], 
         [-np.sin(angle), np.cos(angle)]]
    )
    return np.matmul(xy, rot_mat)


def format_timedelta(timedelta):
    """
    Changes the Timedelta of the Qualifying Lap Times into string in the format "MM:SS.sss".

    Returns:
    str: A string representing the formatted time in "MM:SS.sss" format. 
         If the input is NaT (Not a Time), an empty string is returned.
    """
    if pd.isna(timedelta):
        return ""
    
    # Extract the minutes, seconds, and milliseconds from the timedelta
    total_seconds = int(timedelta.total_seconds())
    minutes = total_seconds // 60
    seconds = total_seconds % 60
    milliseconds = int(timedelta.microseconds / 1000)

    return f"{minutes}:{seconds:02d}.{milliseconds:03d}"

def format_lap_time(seconds):
    """
    Formats the Qualifying Standings into a string in the format "MM:SS.sss".

    Returns:
    str: A string representing the formatted time in "MM:SS.sss" format.
    """
    # Calculate the number of minutes from the total seconds
    minutes = int(seconds // 60)
    
    # Calculate the remaining seconds after converting to minutes
    seconds_remainder = int(seconds % 60)
    
    # Calculate the remaining milliseconds
    milliseconds = int((seconds - minutes * 60 - seconds_remainder) * 1000)
    
    # Return the formatted string
    return f'{minutes:02d}:{seconds_remainder:02d}.{milliseconds:03d}'



def format_to_hh_mm_ss(ms):
    """
    Changes the Race time from milliseconds into a string in the format "HH:MM:SS.sss".

    Parameters:
    ms (int): The time duration in milliseconds to format.

    Returns:
    str: A string representing the formatted time in "HH:MM:SS.sss" format.
    """
    hours = int(ms // 3600000)
    ms = int(ms % 3600000)
    minutes = int(ms // 60000)
    ms = int(ms % 60000)
    seconds = int(ms // 1000)
    milliseconds = int(ms % 1000)
    
    return f"{hours:02}:{minutes:02}:{seconds:02}.{milliseconds:03}"


def format_to_seconds(ms):
    """
    Formats the milliseconds in "SS.sss
    """
    seconds = int(ms // 1000)
    milliseconds = int(ms % 1000)
    return f"+ {seconds}.{milliseconds:03d} s"


def get_driver_ranking(session):
    '''Get driver ranking from a session.'''
    return session.groupby('Driver')['Time'].min().sort_values().index.tolist()

def get_drivers_list(session):
    """
    Gets a list with the Abbreviation of the Drivers.
    """
    drivers = session.drivers
    return [session.get_driver(driver)["Abbreviation"] for driver in drivers]


def get_avg_trackspeed(session):
    """
    Extracts the average Track Speed of the Track
    """
    fastest = session.laps.pick_fastest().get_car_data()
    avg_speed =  fastest['Speed'].mean()
    return int(round(avg_speed,0))


##########################################################################################
# OVERVIEW

def get_final_race_results(session):
    """
    Get the final race results based on the session status.

    Parameters:
    session (dict): A dictionary containing race session information. 
                    It must have a 'status' key indicating the race status 
                    and 'totalRaceTimeMillis' key if the race is finished.

    Returns:
    str or int: The total race time in milliseconds if the race is finished,
                the status '+1 Lap' or '+2 Laps' if the racer is one or two laps 
                behind, respectively, or 'DNF' (Did Not Finish) for other statuses.
    """
    # Check if the race is finished and return the total race time
    if session['status'] == 'Finished':
        return session['totalRaceTimeMillis']
    
    # Check if the racer is one lap behind and return the corresponding status
    elif session['status'] == '+1 Lap':
        return '+1 Lap'
    
    # Check if the racer is two laps behind and return the corresponding status
    elif session['status'] == '+2 Laps':
        return '+2 Laps'
    
    # Return 'DNF' for any other race status
    else:
        return 'DNF'


def get_race_results_df(round_number, season = 2024):
    """
    Retrieve and format race results for a given round and season.

    Parameters:
    round_number (int): The round number of the race.
    season (int, optional): The season year of the race. Default is 2024.

    Returns:
    pandas.DataFrame: A DataFrame containing the formatted race results.
    """
    ergast = Ergast()
    results = ergast.get_race_results(season, round_number)
    race_results = results.content[0]
    
    # Dictionary to rename specific columns
    col_to_rename = {'position': 'Position', 'constructorName': 'Constructor'}
    
    # Fill missing race times with 0
    race_results['totalRaceTimeMillis'] = race_results['totalRaceTimeMillis'].fillna(0)
    
    # Calculate the difference between the first driver's time and the other drivers
    first_drivers_time = race_results['totalRaceTimeMillis'].iloc[0]
    race_results['differenceFromFirst'] = race_results['totalRaceTimeMillis'] - first_drivers_time

    # Format the first driver's time
    race_results.iloc[0, race_results.columns.get_loc('totalRaceTimeMillis')] = format_to_hh_mm_ss(first_drivers_time)

    # Format other drivers' time differences
    for idx in range(1, len(race_results)):
        time_diff = race_results.iloc[idx]['differenceFromFirst']
        race_results.iloc[idx, race_results.columns.get_loc('totalRaceTimeMillis')] = format_to_seconds(time_diff)

    # Drop the temporary column used for time difference calculation
    race_results.drop(columns=['differenceFromFirst'], inplace=True)
    
    # Apply the get_final_race_results function to each row to get the final race time/status
    race_results['Time'] = race_results.apply(get_final_race_results, axis=1)
    
    # Create a 'Driver' column by combining the given name and family name
    race_results['Driver'] = race_results['givenName'].str[0] + ' ' + race_results['familyName']
    
    # Rename columns based on the col_to_rename dictionary
    race_results.rename(columns=col_to_rename, inplace=True)
        
    return race_results


# Function for plotting the Race Track in the overview
def plot_track(session, circuit_info):
    """
    Plot the race track using the telemetry data from the fastest lap in the session.

    Parameters:
    session (object): The session object containing lap and telemetry data.
    circuit_info (object): The circuit information object containing track rotation data.

    Returns:
    plotly.graph_objects.Figure: A Plotly figure object representing the race track.
    """
    # Pick the fastest lap from the session
    lap = session.laps.pick_fastest()
    
    # Convert the track rotation angle from degrees to radians
    track_angle = circuit_info.rotation / 180 * np.pi
    
    # Convert telemetry data to numpy arrays for easier indexing
    x = lap.telemetry["X"].to_numpy()
    y = lap.telemetry["Y"].to_numpy()

    # Rotate the telemetry points by the track angle
    points_track = rotate(
        np.array([x, y]).T,
        angle=track_angle,
    )
    
    # Create a Plotly figure
    fig = go.Figure()

    # Add the race track trace to the figure
    fig.add_trace(
        go.Scatter(
            x=points_track[:, 0],
            y=points_track[:, 1],
            mode="lines",
            hoverinfo="none",
            line=dict(color="#3da3ff", width=5),
        )
    )

    x_min, x_max = points_track[:, 0].min(), points_track[:, 0].max()
    y_min, y_max = points_track[:, 1].min(), points_track[:, 1].max()


    # Apply zoom to the x and y ranges
    x_range = apply_zoom(x_min, x_max, zoom_factor=1.2)
    y_range = apply_zoom(y_min, y_max, zoom_factor=1.2)

    # Define the length of the arrow indicating the start direction of the track
    arrow_length = 7

    # Add an annotation arrow to indicate the start direction
    fig.add_annotation(
        x=points_track[arrow_length, 0],
        y=points_track[arrow_length, 1],
        ax=points_track[0, 0],
        ay=points_track[0, 1],
        xref="x",
        yref="y",
        axref="x",
        ayref="y",
        showarrow=True,
        arrowhead=3,
        arrowsize=4,
        arrowcolor="white",
    )

    # Update layout settings for the figure
    fig.update_layout(
        title='',
        autosize=True,
        showlegend=False,
        xaxis=dict(visible=False, range=x_range),
        yaxis=dict(visible=False, range=y_range),
        height=600,
        width=800,
    )
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)

    return fig


def get_quali_ranking(session):
    '''
    Get the qualifying ranking from a session.
    Parameters:
        session (FastF1 session object): The FastF1 session object containing qualifying results.
    Returns:
        DataFrame: A DataFrame containing the sorted qualifying ranking with lap times converted to seconds.
    '''
    # Extract top 10 results from Q3 and format the DataFrame
    result_q3 = (
        session.results[:10]
        .drop(['Q1', 'Q2'], axis=1)
        .rename(columns={'Q3': 'LapTime'}))
    
    # Extract results from 11th to 15th position from Q2 and format the DataFrame
    result_q2 = (
        session.results[10:15]
        .drop(['Q1', 'Q3'], axis=1)
        .rename(columns={'Q2': 'LapTime'}))
    
    # Extract results from 16th to 20th position from Q1 and format the DataFrame
    result_q1 = (
        session.results[15:20]
        .drop(['Q2', 'Q3'], axis=1)
        .rename(columns={'Q1': 'LapTime'}))

    # Concatenate all results and sort them by position
    all_results = pd.concat([result_q1, result_q2, result_q3])
    sorted_results = all_results.sort_values(by='Position')
    
    # Convert LapTime to seconds for easier comparison
    sorted_results['LapTime'] = sorted_results['LapTime'].dt.total_seconds()
    return sorted_results


def get_quali_results(session):
    '''
    Returns the Qualifying Results in the format of 
    Driver1 -> 92.00  [sec]
    Driver2 -> 0.32   [sec]
    '''
    # Fetching and preparing the results dataframe
    results = get_quali_ranking(session)[['BroadcastName', 'LapTime']].copy()
    results.rename(columns={'BroadcastName': 'Drivers', 'LapTime': 'Time'}, inplace=True)

    # Convert first two letters of 'Drivers' to uppercase
    results['Drivers'] = results['Drivers'].apply(lambda name: name[:2].upper() + name[2:])

    # Get the time of the first driver
    first_drivers_time = results.iloc[0]['Time']

    # Calculate time differences
    results['TimeDifference'] = results['Time'] - first_drivers_time

    # Format first driver's time
    formatted_first_time = format_lap_time(first_drivers_time)
    results.iloc[0, results.columns.get_loc('Time')] = formatted_first_time

    # Handle NaN values in 'TimeDifference' by marking them as 'DNF'
    results['TimeDifference'] = results['TimeDifference'].apply(lambda x: 'DNF' if pd.isna(x) else x)


    # Format other drivers' time differences
    for idx in range(1, len(results)):
        time_diff = results.iloc[idx]['TimeDifference']
        if time_diff == 'DNF':
            results.iloc[idx, results.columns.get_loc('Time')] = 'DNF'
        else:
            seconds = int(time_diff)
            milliseconds = int((time_diff - seconds) * 1000)
            results.iloc[idx, results.columns.get_loc('Time')] = f"+{seconds}.{milliseconds:03d} s"

    # Dropping the TimeDifference column as it's no longer needed
    results.drop(columns=['TimeDifference'], inplace=True)

    return results

##########################################################################################
# TELEMETRY

# Telemetry Qualifying Plots 

def driver_comparison_track_plot(session, selected_drivers):
    """
    Plots a comparison of the fastest laps of two drivers on a racing track, highlighting sections
    where one driver is faster than the other.

    Parameters:
    - session: The racing session data containing laps and circuit information.
    - selected_drivers: A list containing the names of the two drivers to compare.

    Returns:
    - fig: A Plotly figure object representing the track with sections color-coded based on speed differences
           between the two drivers.
    """
    # Pick the fastest laps for the chosen drivers
    driver1 = session.laps.pick_driver(selected_drivers[0]).pick_fastest()
    driver2 = session.laps.pick_driver(selected_drivers[1]).pick_fastest()

    # Validate that both drivers have valid lap times
    if pd.isna(driver1.LapTime):
        raise ValueError(f"{selected_drivers[0]} has no LapTime. Please choose another driver.")
    if pd.isna(driver2.LapTime):
        raise ValueError(f"{selected_drivers[1]} has no LapTime. Please choose another driver.")

    # Get telemetry data and calculate speed differences
    driver1_tel = driver1.get_car_data().add_distance()
    driver2_tel = driver2.get_car_data().add_distance()
    speed_diff = driver1_tel['Speed'] - driver2_tel['Speed']

    # Get circuit info and rotate the track layout
    circuit_info = session.get_circuit_info()
    track_angle = circuit_info.rotation / 180 * np.pi
    track = driver1.get_pos_data().loc[:, ('X', 'Y')].to_numpy()
    rotated_track = rotate(track, angle=track_angle)

    # Initialize Plotly figure
    fig = go.Figure()

    # Plot the base track layout
    fig.add_trace(go.Scatter(
        x=rotated_track[:, 0],
        y=rotated_track[:, 1],
        mode='lines',
        line=dict(color='lightgrey', width=3),
        name='Track Base',
        showlegend=False,
        hoverinfo='none'
    ))

    
    # Track legend status for each driver
    legend_added = {selected_drivers[0]: False, selected_drivers[1]: False}

    # Plot sections of the track with speed difference color-coding
    for i in range(len(rotated_track) - 1):
        if i < len(speed_diff):
            driver_name = selected_drivers[0] if speed_diff.iloc[i] >= 0 else selected_drivers[1]
            color = '#3DA3FF' if speed_diff.iloc[i] >= 0 else '#ff0000'
            show_legend = not legend_added[driver_name]
            fig.add_trace(go.Scatter(
                x=[rotated_track[i, 0], rotated_track[i+1, 0]],
                y=[rotated_track[i, 1], rotated_track[i+1, 1]],
                mode='lines',
                line=dict(color=color, width=5),
                name=driver_name,
                showlegend=show_legend,
                hoverinfo='skip'
            ))
            if show_legend:
                legend_added[driver_name] = True
    
    # Apply zoom to the plot
    x_range = apply_zoom(rotated_track[:, 0].min(), rotated_track[:, 0].max(), zoom_factor=1.2)
    y_range = apply_zoom(rotated_track[:, 1].min(), rotated_track[:, 1].max(), zoom_factor=1.2)

    # Add an arrow to indicate the start direction of the track
    arrow_length = min(7, len(rotated_track) - 1)
    fig.add_annotation(
        x=rotated_track[arrow_length, 0],
        y=rotated_track[arrow_length, 1],
        ax=rotated_track[0, 0],
        ay=rotated_track[0, 1],
        xref="x",
        yref="y",
        axref="x",
        ayref="y",
        showarrow=True,
        arrowhead=3,
        arrowsize=4,
        arrowcolor="white")
    
    # Annotate corners with custom offset and rotation
    offset_vector = [300, 0]  # Offset length chosen arbitrarily
    for _, corner in circuit_info.corners.iterrows():
        offset_angle = corner["Angle"] / 180 * np.pi
        offset = rotate(np.array([offset_vector]), angle=offset_angle)[0]
        text_x = corner["X"] + offset[0] 
        text_y = corner["Y"] + offset[1] *1.2
        text_position = rotate(np.array([[text_x, text_y]]), angle=track_angle)[0]
        
        
        # Add the Circle Trace 
        fig.add_trace(go.Scatter(
            x=[text_position[0]],
            y=[text_position[1]],
            mode='markers',
            marker=dict(size=22, symbol='circle', color='LightSkyBlue',line=dict(color='DarkSlateGrey', width=1)), 
            showlegend=False,
            hoverinfo='none'))
        
        # Numerate the Corners
        fig.add_trace(go.Scatter(
            x=[text_position[0]],
            y=[text_position[1]],
            text=f"{corner['Number']}{corner['Letter']}",
            mode='text',
            textposition='middle center',
            showlegend=False,
            textfont=dict(size=17, color = 'black'), 
            hoverinfo = 'none'))

    # Update layout to disable zoom and hover
    fig.update_layout(
        showlegend=True,
        legend_title="Driver",
        xaxis=dict(range=x_range, showgrid=False, zeroline=False, visible=False, fixedrange=True),
        yaxis=dict(range=y_range, showgrid=False, zeroline=False, visible=False, fixedrange=True),
        width=450, 
        height=550,
        dragmode=False)

    return fig


def telemetry_quali_plots(df1, df2, driver1, driver2, selected_metrics, circuit_info):
    '''
    Create stacked subplots comparing telemetry data of two drivers with driver-specific line colors.

    Parameters:
    - df1: DataFrame containing telemetry data for driver1.
    - df2: DataFrame containing telemetry data for driver2.
    - driver1: Name of the first driver.
    - driver2: Name of the second driver.
    - selected_metrics: List of telemetry metrics to be plotted.
    - circuit_info: DataFrame containing circuit corner information.

    Returns:
    - fig: A Plotly figure object with subplots for each selected metric.
    '''
    num_metrics = len(selected_metrics)
    metric_units = {
        'Speed': 'km/h',
        'Throttle': '%',
        'Brake': '',
        'nGear': '',  
        'RPM': 'rev/min',
        'DRS': ''
    }

    # Create subplots layout
    fig = make_subplots(
        rows=num_metrics,
        cols=1,
        shared_xaxes=True,
        subplot_titles=tuple(selected_metrics))

    # Define colors for the drivers
    colors = ['#3DA3FF',  '#ff0000']

    # Loop through each selected metric and add traces to the plot
    for i, metric in enumerate(selected_metrics):
        unit = metric_units.get(metric, '')
        text_driver1 = [f'<b>Driver: {driver1}</b><br><b>{value} {unit}</b>' for value in df1[metric]]
        text_driver2 = [f'<b>Driver: {driver2}</b><br><b>{value} {unit}</b>' for value in df2[metric]]

        # Check if the metric exists in both DataFrames
        if metric in df1.columns and metric in df2.columns:
            fig.add_trace(
                go.Scatter(
                    x=df1['Distance'],
                    y=df1[metric],
                    mode='lines',
                    name=f'{driver1} {metric}',
                    line=dict(color=colors[0], width=3.2),
                    hoverinfo='text',
                    text=text_driver1,
                ),
                row=i + 1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=df2['Distance'],
                    y=df2[metric],
                    mode='lines',
                    name=f'{driver2} {metric}',
                    line=dict(color=colors[1], width=3.2),
                    hoverinfo='text',
                    text=text_driver2,
                ),
                row=i + 1,
                col=1,
            )
            # Update y-axis label with the unit
            fig.update_yaxes(title_text=f'{unit}',  
                             row=i + 1,
                             col=1)
            
            fig.update_xaxes(ticks = 'outside', 
                             ticklen = 10,
                             tickcolor = 'black')

 
                    
                    
            # Add corner annotations for each subplot
            for _, corner in circuit_info.corners.iterrows():
                txt = f"{corner['Number']}{corner['Letter']}"
                # Add a vertical line with annotation
                fig.add_vline(
                    x=corner["Distance"],
                    line=dict(color="gray", width=1, dash="dash"),
                    annotation_text=txt, 
                    annotation_position='bottom',
                    annotation_font=dict(size=12),  
                    annotation_showarrow=False, 
                    row=i + 1,
                    col=1,
                    opacity=0.25  
                )


        else:
            print(f"Metric '{metric}' not found in one of the DataFrames")

    # Update x-axis label for the last subplot
    fig.update_xaxes(title_text='Lap distance (meters)', 
                     row=num_metrics, 
                     col=1,
                         title_font =dict(size=15))
    
    # Update layout settings
    fig.update_layout(
        showlegend=False,
        height= 800, 
        width=800,
        hovermode='x unified')
    
    return fig


def race_track_plot(session, circuit_info):
    """
    Plots the race track for a given session and circuit information.

    Parameters:
    - session: The session object containing lap and telemetry data.
    - circuit_info: DataFrame containing circuit information such as rotation angle and corner details.

    The function performs the following steps:
    1. Picks the fastest lap from the session.
    2. Rotates the track based on the circuit's rotation angle.
    3. Creates a plotly figure and plots the race track line.
    4. Annotates the track with directional arrows.
    5. Annotates the corners with custom offsets and rotations, adding circles and numbering.
    6. Configures the plot layout and returns the figure.

    Returns:
    - fig: A plotly figure object containing the race track plot.
    """
    lap = session.laps.pick_fastest()
    
    track_angle = circuit_info.rotation / 180 * np.pi
    # Convert telemetry data to numpy arrays for easier indexing
    x = lap.telemetry["X"].to_numpy()
    y = lap.telemetry["Y"].to_numpy()

    # Rotate the Points in order to display them correctly
    points_track = rotate(
        np.array([x, y]).T,
        angle=track_angle,
    )
    
    # Create the Plotly Object
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=points_track[:, 0],
            y=points_track[:, 1],
            mode="lines",
            hoverinfo="none",
            line=dict(color="#3da3ff", width=5),
        )
    )

    x_min, x_max = points_track[:, 0].min(), points_track[:, 0].max()
    y_min, y_max = points_track[:, 1].min(), points_track[:, 1].max()

    x_range = apply_zoom(x_min, x_max, zoom_factor=1.2)
    y_range = apply_zoom(y_min, y_max, zoom_factor=1.2)

    arrow_length = 7

    fig.add_annotation(
        x=points_track[arrow_length, 0],
        y=points_track[arrow_length, 1],
        ax=points_track[0, 0],
        ay=points_track[0, 1],
        xref="x",
        yref="y",
        axref="x",
        ayref="y",
        showarrow=True,
        arrowhead=3,
        arrowsize=4,
        arrowcolor="white",
    )
  # Annotate corners with custom offset and rotation
    offset_vector = [300, 0]   
    for _, corner in circuit_info.corners.iterrows():
        offset_angle = corner["Angle"] / 180 * np.pi
        offset = rotate(np.array([offset_vector]), angle=offset_angle)[0]
        text_x = corner["X"] + offset[0] 
        text_y = corner["Y"] + offset[1] * 1.2
        text_position = rotate(np.array([[text_x, text_y]]), angle=track_angle)[0]
        
        # Add the Circle Trace 
        fig.add_trace(go.Scatter(
            x=[text_position[0]],
            y=[text_position[1]],
            mode='markers',
            marker=dict(size=22, symbol='circle', color='LightSkyBlue',line=dict(color='DarkSlateGrey', width=1)), 
            showlegend=False,
            hoverinfo='none'
        ))
        
        # Numerate the Corners
        fig.add_trace(go.Scatter(
            x=[text_position[0]],
            y=[text_position[1]],
            text=f"{corner['Number']}{corner['Letter']}",
            mode='text',
            textposition='middle center',
            showlegend=False,
            textfont=dict(size=17, color = 'black'), 
            hoverinfo = 'none'
        ))
    fig.update_layout(
        autosize=True,
        showlegend=False,
        xaxis=dict(visible=False, range=x_range),
        yaxis=dict(visible=False, range=y_range),
        height=550,
        width=450, 
    )
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)

    return fig


def telemetry_race_plots(telemetry_plots, selected_metrics, circuit_info):
    '''
    Create a single subplot with multiple traces for comparing telemetry data from different drivers and laps.

    Parameters:
    - telemetry_plots: List of tuples, each containing a DataFrame with telemetry data, driver name, and lap number.
    - selected_metrics: List of telemetry metrics to be plotted.

    Returns:
    - fig: A Plotly figure object with subplots for each selected metric.
    '''
    num_metrics = len(selected_metrics)
    metric_units = {
        'Speed': 'km/h',
        'Throttle': '%',
        'Brake': '',
        'nGear': '',  
        'RPM': 'rev/min',
        'DRS': ''
    }

    # Create subplots layout
    fig = make_subplots(
        rows=num_metrics,
        cols=1,
        shared_xaxes=True,
        subplot_titles=tuple(selected_metrics),
    )

    # Define colors for the traces
    # Colors have been created with davidmathlogic.com
    # The main focus has been put to the Deuteranomaly color vision deficiency
    colors = ['#3DA3FF','#ff0000', '#FF80BA', '#ff7f00', '#C4FFC2', '#FF9797', '#a65628']
    color_index = 0

    # Loop through each telemetry plot and add traces to the plot
    for df, driver, lap in telemetry_plots:
        for i, metric in enumerate(selected_metrics):
            unit = metric_units.get(metric, '')
            if metric in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df['Distance'],
                        y=df[metric],
                        mode='lines',
                        name=f'{driver} Lap {lap} {metric}',
                        line=dict(color=colors[color_index % len(colors)], width=2),
                        hoverinfo='text',
                        text=[f'<b>Driver: {driver}</b><br><b>{value} {metric_units.get(metric, "")}</b><br><b>Lap: {lap}</b>' for value in df[metric]]
                    ),
                    row=i + 1,
                    col=1
                )

        fig.update_xaxes(ticks = 'outside', 
                             ticklen = 10,
                             tickcolor = 'black')
        color_index += 1
    
    # Name the y-Axis 
    for i, metric in enumerate(selected_metrics):
        unit = metric_units.get(metric, '')
        fig.update_yaxes(title_text=unit, row=i + 1, col=1)
        
    # Add corner annotations for each subplot
    for _, corner in circuit_info.corners.iterrows():
        for j in range(num_metrics): 
            fig.add_vline(
                x=corner["Distance"],
                line=dict(color="gray", width=1, dash="dash"),
                annotation_text=f"{corner['Number']}{corner['Letter']}",
                annotation_position='bottom',
                annotation_font=dict(size=12),
                annotation_showarrow=False,
                row=j + 1,
                col=1,
                opacity=0.25
            )
            
        # Update x-axis label for the last subplot
        fig.update_xaxes(title_text='Lap distance (meters)', 
                         row=num_metrics, 
                         col=1,
                         title_font =dict(size=15))

    # Update layout settings
    fig.update_layout(
        height= 800, 
        width=870,
        hovermode='x unified',
        showlegend=False
    )
    
    return fig


################################################################################################################
# STANDINGS

# Qualifying Standings

def qualifying_results(session):
    results  = session.results
    
    for column in ['Q1', 'Q2', 'Q3']:
        results[column] = results[column].apply(format_timedelta)

    # Select the desired columns, potentially including additional columns
    results = results[['Position', 'FullName', 'TeamName', 'Q1', 'Q2', 'Q3']]
    results['TeamName'] = results['TeamName'].str.upper()
    results = results.rename(columns={'FullName': 'Name', 'TeamName': 'Team'})
    results['Position'] = results['Position'].astype(int)
    results = results.set_index('Position')
    
    results.rename(columns=str.upper, inplace=True)
    return results


def qualifying_standing_plot(session):
    q1, q2, q3 = session.laps.split_qualifying_sessions()
    
    # Convert LapTime to total seconds for each qualifying session
    q1['LapTime'], q2['LapTime'], q3['LapTime'] = (
        q1['LapTime'].dt.total_seconds(),
        q2['LapTime'].dt.total_seconds(),
        q3['LapTime'].dt.total_seconds(),
    )

    # Get all drivers who participated in any of the qualifying sessions
    all_drivers = set(q1['Driver']).union(set(q2['Driver']), set(q3['Driver']))

    # Assign a color to each driver
    driver_colors = {
        driver: ff1.plotting.driver_color(driver) for driver in all_drivers
    }

    # Initialize Plotly figure
    fig = go.Figure()

    # Loop through each driver to plot their lap times across the sessions
    for driver in all_drivers:
        driver_lap_times = [
            q[q['Driver'] == driver]['LapTime'].min() for q in [q1, q2, q3]
        ]
        driver_sessions = [1, 2, 3]

        fig.add_trace(
            go.Scatter(
                x=driver_sessions,
                y=driver_lap_times,
                mode='lines+markers',
                name=driver,
                line=dict(color=driver_colors[driver]),
                marker=dict(size=10),
                hovertemplate='Driver: <b>%{text}</b><br>'
                + 'Session:<b> %{x}</b><br>'
                + 'Lap Time:<b> %{y:.2f}s<extra></extra></b>',
                text=[driver] * len(driver_sessions),
            )
        )

    # Update layout settings for the plot
    fig.update_layout(
        xaxis=dict(
            title='Qualifying Session',
            tickmode='array',
            tickvals=[1, 2, 3],
            ticktext=['Q1', 'Q2', 'Q3'],
        ),
        yaxis=dict(title='Lap Time (s)'),
        legend_title='Drivers',
        font=dict(family='Arial', size=12),
        width=700,
        height=750,
        hovermode='closest',
        dragmode=False
    )
    
    return fig


# Race Standings 

def previous_driver_round_standings(year, current_round):
    '''
    Retrieve the driver standings from the previous round of a given season.
    If the current round is the first round, retrieve the current round standings instead.

    Parameters:
    - year: The year of the season for which standings are to be retrieved.
    - current_round: The current round of the season.

    Returns:
    - results: A DataFrame containing the driver code and their position in the standings.
    '''
    ergast = Ergast()
    
    if current_round > 1:
        # Fetch results for the previous round if it's not the first round
        previous_round_results = ergast.get_driver_standings(season=year, round=current_round - 1)
        if previous_round_results.content:
            results = previous_round_results.content[0]
            results['position'] = results.index + 1  # Assigning 'position' based on the index
            return results[['driverCode', 'position']]
    else:
        # For the first round, return current round standings instead of an empty DataFrame
        current_round_results = ergast.get_driver_standings(season=year, round=current_round)
        if current_round_results.content:
            results = current_round_results.content[0]
            results['position'] = results.index + 1
            return results[['driverCode', 'position']]
    
    # Return an empty DataFrame if no results are found
    return pd.DataFrame()


def driver_standings(year, current_round=1):
    '''
    Retrieve the driver standings for a given season and round, including position changes from the previous round.

    Parameters:
    - year: The year of the season for which standings are to be retrieved.
    - current_round: The current round of the season (default is 1).

    Returns:
    - final_standings: A DataFrame containing the position, driver name, team, points, and position change indicator.
    '''
    ergast = Ergast()
    results = ergast.get_driver_standings(year, current_round)

    # Combine given name and family name to form the full driver name
    standings = results.content[0]
    standings.loc[:, 'Driver'] = standings['givenName'] + ' ' + standings['familyName']
    standings.loc[:, 'Constructor'] = standings['constructorNames'].str[0].str.upper()

    # Fetch standings from the previous round
    previous_round_standings = previous_driver_round_standings(year, current_round)

    # Merge current standings with previous round standings
    merged_standings = pd.merge(standings, previous_round_standings, on='driverCode', how='left', suffixes=('', '_previous'))

    # Calculate position change, handling NaN if previous round data is missing
    merged_standings['PositionChange'] = merged_standings['position_previous'] - merged_standings['position']
    merged_standings['PositionChange'].fillna(0, inplace=True) 

    # Map position change to emojis
    merged_standings['Change'] = merged_standings['PositionChange'].map(lambda x: '▲ ' if x >= 1 else ('▼' if x < 0 else '−'))
    merged_standings.rename(columns={'Constructor': 'Team'}, inplace=True)

    # Select relevant columns for final standings
    final_standings = merged_standings.loc[:, ['position', 'Driver', 'Team', 'points', 'Change']]

    # Capitalize column names and set 'POSITION' as the index
    final_standings.rename(columns=str.upper, inplace=True)
    final_standings['POINTS'] = final_standings['POINTS'].astype(int)
    final_standings.set_index('POSITION', inplace=True)

    return final_standings


def previous_constructor_round_standings(year, current_round):
    '''
    Retrieve the constructor standings from the previous round of a given season.
    If the current round is the first round, retrieve the current round standings instead.

    Parameters:
    - year: The year of the season for which standings are to be retrieved.
    - current_round: The current round of the season.

    Returns:
    - results: A DataFrame containing the constructor ID and their position in the standings.
    '''
    ergast = Ergast()
    
    if current_round > 1:
        # Fetch results for the previous round if not the first round
        previous_round_results = ergast.get_constructor_standings(season=year, round=current_round - 1)
    else:
        # If it's the first round, use current round standings
        previous_round_results = ergast.get_constructor_standings(season=year, round=current_round)
        
    if previous_round_results.content:
        results = previous_round_results.content[0]
        results['position'] = results.index + 1  # Assigning 'position' based on the index
        return results[['constructorId', 'position']]
    
    # Return an empty DataFrame if no data available
    return pd.DataFrame()


def constructor_standings(year, current_round=1):
    '''
    Retrieve the constructor standings for a given season and round, including position changes from the previous round.

    Parameters:
    - year: The year of the season for which standings are to be retrieved.
    - current_round: The current round of the season (default is 1).

    Returns:
    - final_standings: A DataFrame containing the position, team name, points, and position change indicator.
    '''
    ergast = Ergast()
    results = ergast.get_constructor_standings(year, current_round)

    # Retrieve the current standings
    standings = results.content[0]
    
    # Fetch standings from the previous round
    previous_round_standings = previous_constructor_round_standings(year, current_round)

    # Merge current standings with previous round standings
    merged_standings = pd.merge(standings, previous_round_standings, on='constructorId', how='left', suffixes=('', '_previous'))

    # Calculate position change, handling NaN if previous round data is missing
    merged_standings['PositionChange'] = merged_standings['position_previous'] - merged_standings['position']
    merged_standings['PositionChange'].fillna(0, inplace=True)  # Assume no change if no previous data
    
    # Map position change to emojis
    merged_standings['Change'] = merged_standings['PositionChange'].map(lambda x: '▲' if x > 1 else ('▼' if x < 0 else '−'))
    merged_standings.rename(columns={'constructorName': 'Team'}, inplace=True)
    
    # Select relevant columns for final standings
    final_standings = merged_standings.loc[:, ['position', 'Team', 'points', 'Change']]
    
    # Capitalize column names and set 'POSITION' as the index
    final_standings.rename(columns=str.upper, inplace=True)
    final_standings['POINTS'] = final_standings['POINTS'].astype(int)
    final_standings.set_index('POSITION', inplace=True)
    
    return final_standings


# For the Driver's standings
def get_season_results(year, current_round):
    '''
    Retrieve and process the race results for a given season up to the current round.

    Parameters:
    - year: The year of the season for which results are to be retrieved.
    - current_round: The current round of the season up to which results are to be considered.

    Returns:
    - new_pivot: A DataFrame containing the cumulative points of each driver up to the current round.
    - races: A Series containing the names of the races held up to the current round.
    '''
    ergast = Ergast()
    races = ergast.get_race_schedule(year)
    results = []

    # Loop through each race in the schedule
    for rnd, race in races["raceName"].items():
        if rnd + 1 > current_round:
            break  # Stop if the round is beyond the current round

        race_results = ergast.get_race_results(season=year, round=rnd + 1)
        if race_results.content:
            temp = race_results.content[0]

            # Check for sprint results and merge with race results if available
            sprint_results = ergast.get_sprint_results(season=year, round=rnd + 1)
            if (
                sprint_results.content
                and sprint_results.description["round"][0] == rnd + 1
            ):
                temp = pd.merge(
                    temp, sprint_results.content[0], on="driverCode", how="left"
                )
                temp["points"] = temp["points_x"] + temp["points_y"]
                temp.drop(columns=["points_x", "points_y"], inplace=True)

            temp["round"] = rnd + 1
            temp["race"] = race.removesuffix(" Grand Prix")
            temp = temp[["round", "race", "driverCode", "points"]]
            results.append(temp)

    # Concatenate all results into a single DataFrame
    results = pd.concat(results)
    races = results["race"].drop_duplicates()

    # Create a pivot table with cumulative points for each driver
    pivot = (
        results.pivot_table(
            index="round", columns="driverCode", values="points", aggfunc="sum")
        .fillna(0)
        .cumsum()
    )
    # Add a row of zeros at the beginning for cumulative sum calculation
    zero_row = pd.DataFrame(0, index=[0], columns=pivot.columns)
    new_pivot = pd.concat([zero_row, pivot])

    new_pivot.reset_index(drop=True, inplace=True)

    return new_pivot, races


def race_standings_plot(new_pivot, races):
    '''
    Create a plot of race standings, showing the points of each driver across the races.

    Parameters:
    - new_pivot: A DataFrame containing the cumulative points of each driver for each race.
    - races: A Series containing the names of the races.

    Returns:
    - fig: A Plotly figure object representing the race standings with points for each driver.
    '''
    traces = []

    # Create a trace for each driver
    for column in new_pivot.columns:
        trace = go.Scatter(
            x=[str(x) for x in range(len(races) + 1)],
            y=new_pivot[column],
            mode="markers+lines",
            line=dict(width=2, color=ff1.plotting.driver_color(column)),  
            marker=dict(size=8, color=ff1.plotting.driver_color(column)), 
            name=column,
            hoverinfo="text",
            text=[
                f"Driver: <b>{column}</b><br>Points: <b>{p}</b><br>Race: <b>{r}</b>"
                for r, p in zip(['Start'] + races.tolist(), new_pivot[column])
            ],
        )
        traces.append(trace)

    # Define layout settings for the plot
    layout = go.Layout(
        xaxis=dict(
            title="Round",
            tickvals=list(range(len(races) + 1)),
            ticktext=['Start'] + [str(race) for race in races],
            tickangle=0,
            tickfont=dict(size=15),  
        ),
        yaxis=dict(
            title="Points",
        ),
        legend=dict(
            title=dict(
                text='Drivers',
                font=dict(size=20)),
            font=dict(size=18)),
        margin={"l": 40, "r": 40, "t": 40, "b": 40},
        showlegend=True,
        hovermode="closest",
        width=1200,
        height=750,
    )

    fig = go.Figure(data=traces, layout=layout)

    # Ensure fixed range for x and y axes
    fig.update_layout(xaxis_fixedrange=True, yaxis_fixedrange=True)

    return fig


# Cosntructor Standings

def get_constructor_season_results(year, round_number):
    '''
    Retrieve and process the constructor standings for a given season up to a specific round.

    Parameters:
    - year: The year of the season for which results are to be retrieved.
    - round_number: The round number up to which results are to be considered.

    Returns:
    - new_pivot: A DataFrame containing the cumulative points of each constructor up to the specified round.
    - races: A Series containing the names of the races held up to the specified round.
    '''
    ergast = Ergast()
    races = ergast.get_race_schedule(year)
    results = []

    # Loop through each race in the schedule
    for rnd, race in races['raceName'].items():
        if rnd + 1 > round_number:
            break  # Stop if the round is beyond the specified round number
        
        constructor_results = ergast.get_constructor_standings(season=year, round=rnd + 1)
        if constructor_results.content:
            temp = constructor_results.content[0]

            temp['round'] = rnd + 1
            temp['race'] = race.removesuffix(' Grand Prix')
            temp = temp[['round', 'race', 'constructorName', 'points']]
            results.append(temp)

    # Concatenate all results into a single DataFrame
    results = pd.concat(results)
    races = results['race'].drop_duplicates()

    # Create a pivot table with cumulative points for each constructor
    pivot = results.pivot_table(index='round', columns='constructorName', values='points', aggfunc='sum').fillna(0).cumsum()

    # Rename constructors for consistency
    name_change = {
        'alpine': 'Alpine F1 Team',
        'aston_martin': 'Aston Martin',
        'ferrari': 'Ferrari',
        'haas': 'Haas F1 Team',
        'mclaren': 'McLaren',
        'mercedes': 'Mercedes',
        'rb': 'RB F1 Team',
        'red_bull': 'Red Bull',
        'sauber': 'Sauber',
        'williams': 'Williams'
    }
    pivot.rename(columns=name_change, inplace=True)

    # Add a row of zeros at the beginning for cumulative sum calculation
    zero_row = pd.DataFrame(0, index=[0], columns=pivot.columns)
    new_pivot = pd.concat([zero_row, pivot])

    new_pivot.reset_index(drop=True, inplace=True)

    return new_pivot, races


def constructors_standings(new_pivot, races):
    '''
    Create a plot of constructor standings, showing the points of each team across the races.

    Parameters:
    - new_pivot: A DataFrame containing the cumulative points of each constructor for each race.
    - races: A Series containing the names of the races.

    Returns:
    - fig: A Plotly figure object representing the constructor standings with points for each team.
    '''
    traces = []

    # Create a trace for each constructor
    for column in new_pivot.columns:
        try:
            color = ff1.plotting.team_color(column)
        except KeyError:
            color = '#1634cb'  # Fallback color for teams with name inconsistencies

        trace = go.Scatter(
            x=[str(x) for x in range(len(races) + 1)],
            y=new_pivot[column],
            mode='markers+lines',
            line=dict(width=2, color=color),
            marker=dict(size=8, color=color),
            name=column,
            hoverinfo='text',
            text=[f"Team: <b>{column}</b><br>Points: <b>{p}</b><br>Race: <b>{r}</b>" for r, p in zip(['Start'] + races.tolist(), new_pivot[column])]
        )
        traces.append(trace)

    # Define layout settings for the plot
    layout = go.Layout(
        xaxis=dict(
            title='Round',
            tickvals=list(range(len(races) + 1)),
            ticktext=['Start'] + [str(race) for race in races],
            tickangle=0,
            tickfont = dict(size = 15)
        ),
        yaxis=dict(
            title='Points',
        ),
        legend=dict(
            title=dict(
                text='Team',
                font=dict(
                    size=20)),
            font=dict(
                size=18)),
        margin={'l': 40, 'r': 40, 't': 40, 'b': 40},
        showlegend=True,
        hovermode='closest',
        width=1200,
        height=750,
    )

    fig = go.Figure(data=traces, layout=layout)
    
    # Ensure fixed range for x and y axes
    fig.update_layout(
        xaxis_fixedrange=True,
        yaxis_fixedrange=True,)
    
    return fig

###################################################################
# IN-DEPTH ANALYSIS

# Qualfying 

def prepare_sessions(session):
    '''
    Prepare session data and get all drivers sorted by their rankings across qualifying sessions.

    Parameters:
    - session: The racing session data containing laps information.

    Returns:
    - sessions: A list containing DataFrames for Q1, Q2, and Q3 qualifying sessions.
    - all_drivers_sorted: A list of all drivers sorted by their rankings across the sessions.
    '''
    # Split the Qualifying Sessions 
    q1, q2, q3 = session.laps.split_qualifying_sessions()
    sessions = [q1, q2, q3]
    all_drivers_sorted = []

    # Loop through each session to get the driver ranking
    for sess in sessions:
        session_ranking = get_driver_ranking(sess)
        all_drivers_sorted.extend(
            [drv for drv in session_ranking if drv not in all_drivers_sorted]
        )

    return sessions, all_drivers_sorted


def get_race_stints(session):
    '''
    Retrieve and analyze race stints from the session data.

    Parameters:
    - session: The racing session data containing laps information.

    Returns:
    - stints: A DataFrame containing the stint information for each driver, including stint length and tyre compound used.
    '''
    # Get laps data from the session
    laps = session.laps
    
    # Select relevant columns for stints analysis
    stints = laps[["Driver", "Stint", "Compound", "LapNumber", "FreshTyre"]]
    
    # Group by driver, stint, compound, and fresh tyre status, then count the laps
    stints = (
        stints.groupby(["Driver", "Stint", "Compound", "FreshTyre"])
        .count()
        .reset_index()
    )
    
    # Rename the column 'LapNumber' to 'StintLength' to represent the length of the stint
    stints = stints.rename(columns={"LapNumber": "StintLength"})
    
    return stints


def qualifying_tyre_life(sessions, all_drivers_sorted):
    '''
    Plot tyre life of drivers during qualifying sessions using Plotly.

    Parameters:
    - sessions: A list containing DataFrames for Q1, Q2, and Q3 qualifying sessions.
    - all_drivers_sorted: A list of all drivers sorted by their rankings across the sessions.

    Returns:
    - fig: A Plotly figure object representing the tyre life of drivers with different compounds.
    '''
    driver_positions = {driver: i for i, driver in enumerate(all_drivers_sorted)}
    compound_colors = ff1.plotting.COMPOUND_COLORS  # Ensure this dictionary is available
    fig = go.Figure()

    min_time = float('inf')  # Initialize minimum time

    # Loop through each session and each driver to plot tyre life
    for sess in sessions:
        for driver in all_drivers_sorted:
            if driver in sess['Driver'].unique():
                drv_laps = sess[sess['Driver'] == driver].copy()
                drv_laps['TimeInSeconds'] = drv_laps['Time'].dt.total_seconds()
                y_position = driver_positions[driver]

                # Update the minimum time
                min_time = min(min_time, drv_laps['TimeInSeconds'].min())

                # Group laps by stint and compound to plot each segment
                for (_, compound), group in drv_laps.groupby(['Stint', 'Compound']):
                    start_time = group['TimeInSeconds'].min()
                    end_time = group['TimeInSeconds'].max() + 60
                    color = compound_colors.get(compound.upper(), 'white')
                    
                    used_status = 'USED' if not group['FreshTyre'].all() else 'NEW'
                    hover_text = f'<b>Driver: {driver}<b><br>Compound: {compound}<br> Tyre: {used_status}'
                    
                    fig.add_trace(go.Bar(
                        x=[end_time - start_time],
                        y=[y_position],
                        base=[start_time],
                        orientation='h',
                        marker=dict(color=color, pattern=dict(shape='/' if used_status == 'USED' else None)),
                        name=compound,
                        hoverinfo='text',
                        hovertext=hover_text,
                        showlegend=False
                    ))

    # Add traces for the legend
    fig.add_trace(go.Bar(
        x=[0],
        y=[0],
        marker=dict(color='white', pattern=dict(shape='/')),
        name='Used Tyres',
        showlegend=True,
        hoverinfo='skip'
    ))

    fig.add_trace(go.Bar(
        x=[0],
        y=[0],
        marker=dict(color='white'),
        name='New Tyres',
        showlegend=True,
        hoverinfo='skip'
    ))

    # Update layout settings for the plot
    fig.update_layout(
        xaxis=dict(
            title='Time [s]',
            range=[min_time, None]  # Set the minimum range to the earliest time value
        ),
        yaxis=dict(
            title='Driver',
            tickmode='array',
            tickvals=list(driver_positions.values()),
            ticktext=all_drivers_sorted,
            autorange='reversed'
        ),
        barmode='stack', 
        height=800,
        width=1200,
        legend=dict(title='Tyre Status')
    )

    return fig


def analyze_driver_positions(session, driver_col="Driver", position_col="Position"):
    '''
    Analyze gained and lost places for drivers during a race session.

    Parameters:
        session (FastF1 session object): The FastF1 session object containing lap data.
        driver_col (str, optional): The name of the column containing driver names. Defaults to "Driver".
        position_col (str, optional): The name of the column containing position data. Defaults to "Position".

    Returns:
        plotly.graph_objects.Figure: A Plotly figure showing gained and lost places for each driver.
    '''
    # Extract lap data from the session
    laps = session.laps

    # Identify unique lap numbers
    laps_list = laps["LapNumber"].unique()

    # Select data for the first, second last, and last laps
    first_lap = laps[laps["LapNumber"] == laps_list[0]]
    second_last_lap = laps[laps["LapNumber"] == laps_list[-2]]
    last_lap = laps[laps["LapNumber"] == laps_list[-1]]

    # Identify drivers in the last lap
    drivers_in_last_lap = last_lap[driver_col].unique()

    # Filter out drivers who completed the race in the second last lap
    second_last_lap_filtered = second_last_lap[
        ~second_last_lap[driver_col].isin(drivers_in_last_lap)]

    # Combine data from the second last and last laps
    combined_laps = pd.concat([second_last_lap_filtered, last_lap], ignore_index=True)

    # Extract first and last lap positions for each driver
    first_laps = first_lap[[driver_col, position_col]]
    last_laps = combined_laps[[driver_col, position_col]]

    # Merge first and last lap data based on driver names
    fist_last = pd.merge(
        first_laps, last_laps, on=driver_col, suffixes=("_FirstLap", "_LastLap"))

    # Calculate gained or lost places for each driver
    fist_last["gained_lost"] = (
        fist_last[position_col + "_FirstLap"] - fist_last[position_col + "_LastLap"])

    # Sort the data by gained or lost places
    fist_last = fist_last.sort_values(by="gained_lost", ascending=False)

    # Remove any rows with missing values
    fist_last = fist_last.dropna()

    # Create a horizontal bar plot showing gained or lost places for each driver
    fig = px.bar(
        fist_last, x="gained_lost", y=driver_col, orientation="h", text="gained_lost")

    # Customize plot appearance
    fig.update_traces(
        marker_color="#3387d2",
        textfont_size=18,
        textangle=0,
        hovertemplate="Driver: <b>%{y}</b>",)

    fig.update_layout(
        xaxis_title="Places Lost (-) / Places Gained (+)",
        yaxis_title="Driver",
        yaxis=dict(
            autorange="reversed",
            fixedrange=True,
            title_font=dict(size=14),
            tickfont=dict(size=16),),
        xaxis=dict(
            zeroline=True, zerolinewidth=3, zerolinecolor="Gray", fixedrange=True),
        barmode="relative",
        bargap=0.10,
        dragmode=False,
        width=800,
        height=600,)

    return fig

# Weather Plot
def plot_temperatures(df):
    df['Time'] = pd.to_datetime(df['Time'], unit='ns')
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df['Time'], 
        y=df['AirTemp'],
        mode='lines',
        name='Air Temperature',
        line=dict(color='#f781bf', width = 3),
        hovertemplate='Air Temperature: <b>%{y}°C</b> <br>Elapsed Time: <b>%{x}</b><extra></extra>'
    ))

    fig.add_trace(go.Scatter(
        x=df['Time'],
        y=df['TrackTemp'],
        mode='lines',
        name='Track Temperature',
        line=dict(color='#377eb8', width = 3),
        hovertemplate='Track Temperature: <b>%{y}°C</b> <br>Elapsed Time: <b>%{x}</b><extra></extra>'
    ))

    fig.update_layout(
        xaxis=dict(
            title='Time since start',
            tickformat='%H:%M', 
            title_font = dict(size =18), 
            tickfont = dict(size = 14)
        ),
        legend = dict(title=dict(text = 'Temperature Type',font=dict(size=20)), 
                      font=dict(size=18) ),
        yaxis = dict(
            title = 'Temperature (°C)', 
            title_font = dict(size= 18), 
            tickfont = dict(size = 14)),
        width=1200,
        height=800
    )

    return fig


# Race
def tyre_strategies(drivers, stints):
    '''
    Generate a plot showing tyre strategies for each driver during a race session.

    Parameters:
        drivers (list): A list of driver names.
        stints (DataFrame): A DataFrame containing tyre stints data for each driver.

    Returns:
        plotly.graph_objects.Figure: A Plotly figure showing tyre strategies for each driver.
    '''
    
    fig = make_subplots(rows=1, cols=1)
    compound_colors = ff1.plotting.COMPOUND_COLORS
    used_compounds = set(stints["Compound"])
    driver_indices = list(range(len(drivers)))

    # Reverse the y-axis to display drivers from top to bottom
    fig.update_yaxes(tickvals=driver_indices, ticktext=drivers, autorange="reversed")

    # Iterate over each driver
    for driver_index, driver in enumerate(drivers):
        # Filter stints data for the current driver
        driver_stints = stints.loc[stints["Driver"] == driver]

        # Initialize the previous stint end variable
        previous_stint_end = 0

        # Iterate over each stint of the current driver
        for idx, row in driver_stints.iterrows():
            # Determine hatch pattern based on tyre freshness
            hatch_pattern = "" if row["FreshTyre"] else "//"

            # Add a bar trace for the current stint
            fig.add_trace(
                go.Bar(
                    y=[driver_index],
                    x=[row["StintLength"]],
                    base=previous_stint_end,
                    orientation="h",
                    marker=dict(
                        color=compound_colors[row["Compound"]],
                        line=dict(
                            color="black", width=2
                        ), 
                    ),
                    showlegend=False,
                    name=row["Compound"],
                    legendgroup=row["Compound"],
                    width=0.9,
                )
            )

            # Calculate the end point of the current stint
            stint_end = previous_stint_end + row["StintLength"]
            is_last_stint = idx == driver_stints.index[-1]

            # Add lap number annotation if not the last stint
            if previous_stint_end >= 0 and not is_last_stint:
                fig.add_annotation(
                    x=stint_end,
                    y=driver_index,
                    text=f"L{stint_end}",
                    showarrow=False,
                    font=dict(size=14, color="white"),
                    bgcolor="black",
                )

            # Update the previous stint end
            previous_stint_end = stint_end

    # Create legend items for each used tyre compound
    legend_items = [
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(size=15, color=compound_colors[compound]),
            legendgroup=compound,
            showlegend=True,
            name=compound)
        for compound in used_compounds]

    # Add legend items to the figure
    for item in legend_items:
        fig.add_trace(item)

    # Customize layout
    fig.update_layout(
        barmode="stack",
        xaxis=dict(
            title="Lap Number",
            range=[0, 58],
            showgrid=False,
            gridcolor="gray",
            gridwidth=0.5,
        ),
        yaxis=dict(showgrid=False, gridcolor="gray", gridwidth=0.5, tickfont = dict(size=16)),
        legend=dict(
        title=dict(
            text="Tyre Compounds",
            font=dict(size=20)),
            font=dict(size=16)),  
        dragmode=False,
        height=600,
        width=1200)

    return fig


def driver_laptime_violinplot(session):
    '''
    Generate a violin plot showing lap time distribution for the top 10 point finishers in a race session.

    Parameters:
        session (FastF1 session object): The FastF1 session object containing lap data.

    Returns:
        plotly.graph_objects.Figure: A Plotly figure showing lap time distribution for each driver.
    '''
    # Select the top 10 point finishers
    point_finisher = session.drivers[:10]

    # Filter lap data for the selected drivers and pick only quick laps
    driver_laps = session.laps.pick_drivers(point_finisher).pick_quicklaps()

    # Reset index to ensure proper data alignment
    driver_laps = driver_laps.reset_index()

    # Get the abbreviation of each driver in finishing order
    finishing_order = [session.get_driver(i)["Abbreviation"] for i in point_finisher]

    # Define colors for each driver
    driver_colors = {
        abv: ff1.plotting.DRIVER_COLORS[driver]
        for abv, driver in ff1.plotting.DRIVER_TRANSLATE.items()
    }

    # Define colors for each compound type
    my_compound = {
        "SOFT": "#da291c",
        "MEDIUM": "#ffd12e",
        "HARD": "#efefef",
        "INTERMEDIATE": "#43b02a",
        "WET": "#0067ad",
        "UNKNOWN": "#00ffff",
        "TEST-UNKNOWN": "#434649",
    }

    # Convert lap time to seconds for easier plotting
    driver_laps["LapTime(s)"] = driver_laps["LapTime"].dt.total_seconds()

    # Generate a violin plot for lap time distribution
    fig = px.violin(
        driver_laps,
        x="Driver",
        y="LapTime(s)",
        color="Driver",
        category_orders={"Driver": finishing_order},
        color_discrete_map=driver_colors,
        violinmode="overlay",
        hover_data=None,
        hover_name=None,
        labels=None,
    )

    # Hide hover information and legend
    fig.update_traces(hoverinfo="skip", showlegend=False)

    # Add scatter plots for each compound type
    for compound, color in my_compound.items():
        filtered_data = driver_laps[driver_laps["Compound"] == compound]
        fig.add_trace(
            go.Scatter(
                x=filtered_data["Driver"],
                y=filtered_data["LapTime(s)"],
                mode="markers",
                name=f"{compound}",
                marker=dict(color=color, size=5),
                text="",
            )
        )

    # Customize layout
    fig.update_layout(
        legend_title_text="Compound",
        xaxis_title="Driver",
        yaxis_title="Lap Time [s]",
        legend=dict(x=1.05, y=1,            
                    title=dict(font=dict(size=20)),  
                    font=dict(size=18)),
        xaxis=dict(
            title="Driver",
            title_font=dict(size=18),  
            tickfont=dict(size=20)),
        yaxis=dict(
            title="Lap Time [s]",
            title_font=dict(size=18), 
            tickfont=dict(size=14)),
        width=1200,
        height=600
    )
    # Remove marker lines
    fig.update_traces(marker=dict(line=dict(width=0)))

    return fig



################################################################################
# We tried to Implement the Dash Application in order to enhance the 
# User Experience, but we did not achieve the Goal in order to spend 
# too much Time on it we decided to have this implementation for further considerations.


# import dash
# from dash import dcc, html, Input, Output
# import dash_bootstrap_components as dbc

# # Function to create a speed plot
# def create_speed_plot(df_list, dist_col, y_col, labels):
#     fig = go.Figure()
#     for i, df in enumerate(df_list):
#         fig.add_trace(
#             go.Scatter(
#                 x=df[dist_col],
#                 y=df[y_col],
#                 mode="lines",
#                 name=labels[i],
#                 hoverinfo="text",
#                 text=[
#                     f"<b>{labels[i]}</b> <br><b>Speed</b>: {speed} km/h"
#                     for speed in df[y_col]
#                 ],
#             )
#         )
#     return fig

# # Function to initialize and run the Dash app
# def setup_dash_app(points_track, speed_plot):
#     app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
#     app.layout = html.Div([
#         dcc.Graph(id='track-plot', style={'height': '600px'}),
#         dcc.Graph(id='speed-plot', figure=speed_plot, style={'height': '800px'})
#     ],  fluid=True)

#     @app.callback(
#         Output('track-plot', 'figure'),
#         [Input('speed-plot', 'hoverData')]
#     )

#     def update_track_plot(hoverData):
#         fig = go.Figure()
#         fig.add_trace(go.Scatter(
#             x=points_track[:, 0],
#             y=points_track[:, 1],
#             mode="lines",
#             name="Track",
#             line=dict(color="green", width=5),
#         ))
#         if hoverData is not None:
#             point_index = hoverData['points'][0]['pointIndex']
#             fig.add_trace(go.Scatter(
#                 x=[points_track[point_index, 0]],
#                 y=[points_track[point_index, 1]],
#                 mode='markers',
#                 marker=dict(color='red', size=10),
#                 name='Hovered Point'
#             ))
#         fig.update_layout(
#             paper_bgcolor='rgb(17,17,17)',
#             plot_bgcolor='rgb(17,17,17)',
#             showlegend=False,
#             xaxis=dict(visible=False, showticklabels=False, constrain='domain'),
#             yaxis=dict(visible=False, showticklabels=False, scaleanchor="x", scaleratio=1),
#             margin=dict(l=0, r=0, t=0, b=0)
#     )
#         return fig

#     return app
