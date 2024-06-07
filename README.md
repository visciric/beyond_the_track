# Beyond the Track 
Delve deep into the intricate world of Formula 1 racing! 

## Overview 

This Dashboard is a project work for the Data Visualization and Data Storytelling module of the Spring Semester 2024. 
With this dashboard, it is possible to dive deep into the world of Formula 1. Since a lot of data is gathered during a weekend, 
it is interesting to visualize and analyze this data. 
This dashboard is for Formula 1 enthusiasts as well as people who are new to the sport.

## Features

In this dashboard, you will find:

- Telemetry Data
- Standings of the Championship
- Tyre Management of the Teams

## Additional Features

- Interactive Graphs: Users can interact with the data to see specific details and trends.
- Historical Data: Access data from previous races to compare and analyze performance over time.

## Clone the GitHub Repository

Clone the [Beyond the Track GitHub Repository](https://github.com/visciric/beyond_the_track.git) into your working directory. Use either *GitHub Desktop* or the following command in a terminal:

```sh
git clone https://github.com/visciric/beyond_the_track.git
```

Now there is a new subdirectory named

```
C:\Daten\beyond_the_track
```

where the current version of the repository is cloned. You can always update this with *GitHub Desktop* or the `git pull` command in the terminal when new content is pushed.

## Conda Environment

We set up a virtual environment defined by the `requirements.txt` file. This ensures that you have the needed packages installed. The environment should be in the parent working directory so it can be used in all subdirectories.

Follow these steps in the terminal within the working directory. (Open the working directory in VS Code and open a new terminal there.)

### Installation
To get started with the project, follow these steps:

1. **Create and activate a Conda environment:**
    If you don't already have a Conda environment set up for your project, create and activate one:
    ```sh
    conda create --name formula1 python=3.11.9
    ```

2. **Activate the existing Conda environment:**
    If you already have a Conda environment set up for your project, activate it:
    ```sh
    conda activate formula1
    ```

3. **Install the requirements:**
    With the Conda environment activated, install the necessary packages using the "requirements.txt" file. You can use "pip" within your Conda environment:
    ```sh
    pip install -r requirements.txt
    ```
    
4. **Run the Dashboard:**
    Once the dependencies are installed, you can run the dashboard using Streamlit:
    ```sh
    streamlit run main.py
    ```
