"""
Thomas Bisgaard Astrup
"""


import xml.etree.ElementTree as ET
import os
import math
from datetime import date, datetime, timedelta #timezone
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
import contextily as ctx
from shapely.geometry import LineString


import numpy as np

DATA_FOLDER_DIRECTORY = "C:/Users/biss3/Repositories/sandbox/Scripts/data/activities_test/"
NAME = "Unknown"
YEAR = 2025
OUTPUT_FOLDER_DIRECTORY = "C:/Users/biss3/Repositories/sandbox/"
OUTPUT_FILENAME = "RunningWrapped.gif"



def long_lat_to_distance(lon1_degrees, lat1_degrees, lon2_degrees, lat2_degrees):
    if np.isnan([lon1_degrees, lat1_degrees, lon2_degrees, lat2_degrees]).any():
        return float('nan')
    lon1 = lon1_degrees*np.pi/180
    lon2 = lon2_degrees*np.pi/180
    lat1 = lat1_degrees*np.pi/180
    lat2 = lat2_degrees*np.pi/180

    earth_radius = 6371000  # metres
    delta_lat = lat2 - lat1
    delta_lon = lon2 - lon1
    d = 2*earth_radius*np.asin(
        np.sqrt(
            np.sin(delta_lat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(delta_lon/2)**2
        ))
    return d

def reduce_resolution(*arrays, tolerance=0.0005):
    """
    arrays[0], arrays[1] must be lon, lat
    remaining arrays are arbitrary per-point attributes
    """
    # Safety check
    n = len(arrays[0])
    if not all(len(a) == n for a in arrays):
        raise ValueError("All input arrays must have the same length")

    # Create N-dimensional LineString
    line = LineString(zip(*arrays))

    simplified = line.simplify(tolerance, preserve_topology=False)

    # Unzip reduced coordinates back into separate arrays
    reduced = list(zip(*simplified.coords))

    return [list(r) for r in reduced]

def read_activities(folder) -> None:
    # Loop through all files in specified folder
    def _backfill_nans(lst):
        if isinstance(lst, list):
            for i in range(len(lst) - 2, -1, -1):
                if isinstance(lst[i],float) and math.isnan(lst[i]):
                    lst[i] = lst[i + 1]
    # Activities contains all activities, each file is one activity
    activities_data = []
    previous_date = '2000-01-01T00:00:00.000Z'
    for filename in os.listdir(folder):
        if filename.lower().endswith(".tcx"):
            # Read file
            filepath = os.path.join(folder, filename)
            #print("Reading:", filepath)
            tree = ET.parse(filepath)
            root = tree.getroot()
            ns = {"tcx": "http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2"}
            activity_date = root.find(".//tcx:Activity/tcx:Id", ns).text

            # Initialilse lists to hold trackpoint data and lap data
            trackpoint_start_time = []
            trackpoint_latitude = []
            trackpoint_longitude = []
            trackpoint_altitude = []
            trackpoint_heart_rate = []
            trackpoint_distance = []
            trackpoint_cumulated_distance = []
            lap_start_time = []
            lap_total_time = []
            lap_distance = []

            # Find all laps in the activity
            for lap in root.findall(".//tcx:Lap", ns):
                # Lap data
                lp_start_time = lap.attrib.get("StartTime")
                lp_total_time = float(lap.findtext("tcx:TotalTimeSeconds", default="nan", namespaces=ns))#lap.find("tcx:TotalTimeSeconds", ns).text
                lp_distance = float(lap.findtext("tcx:DistanceMeters", default="nan", namespaces=ns))#lap.find("tcx:DistanceMeters", ns).text
                lp_track = lap.find("tcx:Track", ns)
                if lp_track is None:
                    continue
                lap_start_time.append(lp_start_time)
                lap_total_time.append(lp_total_time)
                lap_distance.append(lp_distance)

                # Find all trackpoints in the lap, each lap contains multiple trackpoint elements
                for tp in lp_track.findall("tcx:Trackpoint", ns):
                    tp_start_time = tp.find("tcx:Time", ns).text
                    tp_latitude = float(tp.findtext("tcx:Position/tcx:LatitudeDegrees", default="nan", namespaces=ns))
                    tp_longitude = float(tp.findtext("tcx:Position/tcx:LongitudeDegrees", default="nan", namespaces=ns))
                    tp_altitude = float(tp.findtext("tcx:AltitudeMeters", default="nan", namespaces=ns))
                    tp_heart_rate = float(tp.findtext("tcx:HeartRateBpm/tcx:Value", default="nan", namespaces=ns))
                    
                    # If several points are available, calculate distance travelled from last point
                    if len(trackpoint_latitude)>0 and len(trackpoint_longitude)>0:
                        tp_distance = long_lat_to_distance(trackpoint_longitude[-1],trackpoint_latitude[-1],tp_longitude,tp_latitude)
                    else:
                        tp_distance = float('nan')
                    tp_last_cumulated_distance = trackpoint_cumulated_distance[-1] if trackpoint_cumulated_distance else 0
                    tp_cumulated_step = tp_distance if not np.isnan(tp_distance) else 0
                    trackpoint_start_time.append(tp_start_time)
                    trackpoint_latitude.append(tp_latitude)
                    trackpoint_longitude.append(tp_longitude)
                    trackpoint_altitude.append(tp_altitude)
                    trackpoint_heart_rate.append(tp_heart_rate)
                    trackpoint_distance.append(tp_distance)
                    trackpoint_cumulated_distance.append(tp_cumulated_step + tp_last_cumulated_distance)

            # If the starting date of lap is the same as the previous, combine the two laps
            new_date = datetime.fromisoformat(activity_date.replace("Z", "+00:00")).date()
            
            activity_dict = {
                'date': activity_date,
                'lap': {
                    'start_time': lap_start_time,
                    'total_time': lap_total_time,
                    'distance': lap_distance
                },
                'trackpoint': {
                    'start_time': trackpoint_start_time,
                    'latitude': trackpoint_latitude,
                    'longitude': trackpoint_longitude,
                    'altitude': trackpoint_altitude,
                    'heart_rate': trackpoint_heart_rate,
                    'distance': trackpoint_distance,
                    'cumulated_distance': trackpoint_cumulated_distance
                    }
                }
            # Clean data set (fill nan values with last known value)
            for lst in activity_dict['lap'].values():
                _backfill_nans(lst)
            for lst in activity_dict['trackpoint'].values():
                _backfill_nans(lst)

            if new_date == previous_date:
                # Combine previous activity with current activity
                activities_data[-1]['lap']['start_time'].extend(activity_dict['lap']['start_time'])
                activities_data[-1]['lap']['total_time'].extend(activity_dict['lap']['total_time'])
                activities_data[-1]['lap']['distance'].extend(activity_dict['lap']['distance'])
                activities_data[-1]['trackpoint']['start_time'].extend(activity_dict['trackpoint']['start_time'])
                activities_data[-1]['trackpoint']['latitude'].extend(activity_dict['trackpoint']['latitude'])
                activities_data[-1]['trackpoint']['longitude'].extend(activity_dict['trackpoint']['longitude'])
                activities_data[-1]['trackpoint']['altitude'].extend(activity_dict['trackpoint']['altitude'])
                activities_data[-1]['trackpoint']['heart_rate'].extend(activity_dict['trackpoint']['heart_rate'])
                activities_data[-1]['trackpoint']['distance'].extend(activity_dict['trackpoint']['distance'])
                trackpoint_cumulated_distance_contd = activities_data[-1]['trackpoint']['cumulated_distance'][-1]
                trackpoint_cumulated_distance_new = [x + trackpoint_cumulated_distance_contd for x in activity_dict['trackpoint']['cumulated_distance']]
                activities_data[-1]['trackpoint']['cumulated_distance'].extend(trackpoint_cumulated_distance_new)
            else:
                # New activity
                activities_data.append(activity_dict)
            # Update previous date
            previous_date = new_date

    return activities_data

def plot_running_wrapped(activities_data: list, year: int=2025, runner_name: str="No name", keep_step: int=10):
    # ==== Prepare Data ====
    x_cal = []  # weekday number of activity date
    y_cal = []  # week number of activity date
    z_cal = []  # date string of activity date
    x_track = []  # longitude
    y_track = []  # latitude
    z_track = []  # cumulated distance
    segments_track = []
    frames = []
    update_map = []
    x_horz = 0.05  # longitudal horizon
    y_horz = 0.05  # latitudal horizon
    x_track_min = []
    x_track_max = []
    y_track_min = []
    y_track_max = []
    x_min = []
    x_max = []
    y_min = []
    y_max = []
    total_distance = 0  # Cumulated distance updated at each new activity
    for i, act in enumerate(activities_data):
        act_date = datetime.fromisoformat(act['date'].replace("Z", "+00:00")).date()
        x_cal.append(act_date.weekday())
        y_cal.append(act_date.isocalendar()[1]-date(year, 1, 1).isocalendar()[1])
        z_cal.append(act_date.strftime("%d-%b-%Y").upper())
        x_raw = act['trackpoint']['longitude']
        y_raw = act['trackpoint']['latitude']
        z_raw = act['trackpoint']['cumulated_distance']
        x, y, z = reduce_resolution(x_raw, y_raw, z_raw)
        x_act_min = min(x)
        x_act_max = max(x)
        y_act_min = min(y)
        y_act_max = max(y)
        x_track_min.append(x_act_min)
        x_track_max.append(x_act_max)
        y_track_min.append(y_act_min)
        y_track_max.append(y_act_max)
        z_track.append([z_val + total_distance for z_val in z])
        total_distance = z_track[-1][-1]
        points =  np.column_stack([x, y])
        segments = np.stack([points[:-1], points[1:]], axis=1)
        segments_track.append(segments)
        
        # Zoombox
        if i==0 or any(
            x_min[-1] > x[j] or x[j] > x_max[-1] or
            y_min[-1] > y[j] or y[j] > y_max[-1]
            for j in (0, -1)
            ):
            #or x[0]<x_min[-1] or x[0]>x_max[-1] or y[0]<y_min[-1] or y[0]>y_max[-1]:
            # If origin or destination is outside last zoom box, then 
            # search among previous zoom box, or ...
            p = i-1
            is_previous_zoom_box = False
            while i>0 and p>=0 and not is_previous_zoom_box:
                if all(
                    x_min[p] < x[j] < x_max[p] and
                    y_min[p] < y[j] < y_max[p]
                    for j in (0, -1)
                    ):
                #x[0]>x_min[p] and x[0]<x_max[p] and y[0]>y_min[p] and y[0]<y_max[p]:
                    is_previous_zoom_box = True
                else:
                    p =- 1
            # ... change to new
            if not is_previous_zoom_box:
                x_min.append(x_act_min - x_horz)
                x_max.append(x_act_max + x_horz)
                y_min.append(y_act_min - y_horz)
                y_max.append(y_act_max + y_horz)
                update_map.append(True)
            else:
                x_min.append(min(x_act_min, x_min[p]))
                x_max.append(max(x_act_max, x_max[p]))
                y_min.append(min(y_act_min, y_min[p]))
                y_max.append(max(y_act_max, y_max[p]))
                update_map.append(False)
        else:
            x_min.append(min(x_act_min, x_min[-1]))
            x_max.append(max(x_act_max, x_max[-1]))
            y_min.append(min(y_act_min, y_min[-1]))
            y_max.append(max(y_act_max, y_max[-1]))
            update_map.append(False)

        for j, (lon, lat, cumdist) in enumerate(zip(x, y, z)):
            if j % keep_step == 0:  # keep one every N'th step
                frames.append((i, j))
    
    # ==== Initialise Plot ====
    fig, axes = plt.subplots(1, 2, figsize=(10,5), gridspec_kw={"width_ratios": [1, 4]})
    ax_left, ax_right = axes
    # Trackpoints
    lc_completed_all = LineCollection([], linewidths=2, colors="gray")
    lc_completed_current = LineCollection([], linewidths=2, colors="black")
    lc_stepping_current = LineCollection([], linewidths=2, colors="red")
    ax_right.add_collection(lc_completed_all)
    ax_right.add_collection(lc_completed_current)
    ax_right.add_collection(lc_stepping_current)
    # Plot
    ax_right.set_xlim(x_min[0], x_max[0])
    ax_right.set_ylim(y_min[0], y_max[0])
    ax_right.set_aspect("equal", adjustable="box")
    # Add basemap
    ctx.add_basemap(
        ax_right,
        source=ctx.providers.CartoDB.PositronNoLabels,
        zoom=12,
        crs="EPSG:4326"
    )
    # Calender
    days_label = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    days = []
    week_numbers = []
    date_pointer = date(year, 1, 1)
    while year == date_pointer.year:
        days.append(date_pointer.weekday())
        week_numbers.append(date_pointer.isocalendar()[1]-date(year, 1, 1).isocalendar()[1])
        date_pointer = date_pointer + timedelta(days=1)
    calendar_all = ax_left.scatter(days, week_numbers, s=10, color='gray', alpha=0.1)
    calendar_completed = ax_left.scatter([], [], s=10, color='black', alpha=1)
    calendar_dot = ax_left.scatter([], [], s=10, color='red', alpha=1)
    ax_left.set_xticks(range(0,7), labels=days_label, rotation=45, fontsize=8)
    ax_left.set_yticks([1]+list(range(5,51,5))+[52], labels=[1]+list(range(5,51,5))+[52], fontsize=6)#, minor=True
    ax_left.set_frame_on(False)
    ax_left.set_ylabel("Week number")
    ax_left.tick_params(left=True, bottom=True)
    # Text in bottom right corner
    text_lr = ax_right.text(
        0.98, 0.02, "GitHub: tbisgaard",
        transform=ax_right.transAxes,
        ha="right",
        va="bottom",
        fontsize=11,
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none")
    )
    fig.suptitle("#RunningWrapped"+str(year)+" - "+runner_name.upper(), fontsize=16, fontstyle="italic")
    plt.show()


    def update(frame):
        # Update function
        # i=activity number, j=trackpoint of activity i
        i, j = frame

        # Calender:
        # - Set current day to red
        # - show current day in text
        calendar_completed.set_offsets(np.column_stack([x_cal[:i], y_cal[:i]]))
        calendar_dot.set_offsets([[x_cal[i], y_cal[i]]])
        ax_left.set_title(str(z_cal[i]))
        # Activity plot:
        # - Current line piece is red
        # - updated cumulated routes for grey colour
        # - Update x,y limits for activity
        # - Show cumulated distance travelled
        if j==0:
            segments_completed_all = np.concatenate(segments_track[:i], axis=0) if i>0 else []
            lc_completed_all.set_segments(segments_completed_all)
            # Reset
            lc_completed_current.set_segments([])
            lc_stepping_current.set_segments([])
        lc_stepping_current.set_segments(segments_track[i][j-2:j])
        lc_completed_current.set_segments(segments_track[i][:j])
        ax_right.set_xlim(x_min[i], x_max[i])
        ax_right.set_ylim(y_min[i], y_max[i])
        if update_map[i]:
            ctx.add_basemap(
                ax_right,
                source=ctx.providers.CartoDB.PositronNoLabels,
                zoom=12,
                crs="EPSG:4326"
            )
        
        ax_right.set_title(f"Total distance {z_track[i][j]/1000:.0f} km")                
        
        return [calendar_dot, calendar_completed, lc_completed_all, lc_completed_current, lc_stepping_current]
    
    ani = FuncAnimation(
        fig,
        update,
        frames=frames,
        interval=100,   # milliseconds between frames
        blit=True
    )

    ani.save(
        OUTPUT_FOLDER_DIRECTORY + OUTPUT_FILENAME,
        dpi=100,
        writer="pillow"
        )

if __name__=="__main__":
    activities_data = read_activities(DATA_FOLDER_DIRECTORY)
    plot_running_wrapped(activities_data, year=YEAR, runner_name=NAME)