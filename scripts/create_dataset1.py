# %%
import pandas as pd
import numpy as np 
import scipy as sp
import os
import argparse

# %%
parser = argparse.ArgumentParser()
parser.add_argument("-in_dir", required = True)
parser.add_argument("-out_dir", required = True)
args = parser.parse_args()

in_dir = args.in_dir
out_dir = args.out_dir


# %%
in_dir = "../Data"

# %%
def get_individual_signals(id, in_dir):
    path = os.path.join(in_dir, id)
    spo2_path = os.path.join(path, "SPO2.txt")
    thorac_path = os.path.join(path,"Thorac.txt")
    flow_path = os.path.join(path,"Flow.txt")
    #events_path = os.path.join(path, "Flow Events.txt")
    profile_path = os.path.join(path, "Sleep Profile.txt")
    
    ap_spo2 = pd.read_csv(spo2_path, sep = ';', skiprows = [0,1,2,3,4,5,6], header = None, names = ["time", "SPO2"], index_col = ["time"])
    ap_thorac = pd.read_csv(thorac_path, sep = ';',  skiprows = [0,1,2,3,4,5,6], header = None, names = ["time","thorac"],  index_col = ["time"])
    ap_flow = pd.read_csv(flow_path, sep = ';', skiprows = [0,1,2,3,4,5,6], header = None, names = ["time", "flow"], index_col = ["time"])
    ap_profile = pd.read_csv(profile_path, sep = ';', skiprows = [0,1,2,3,4,5,6], header = None, names = ["time", "profile"], index_col = ["time"])
    # %%
    ap_spo2.index = pd.to_datetime(
        ap_spo2.index.str.strip(),
        format="%d.%m.%Y %H:%M:%S,%f"   
    )
    ap_thorac.index = pd.to_datetime(
        ap_thorac.index.str.strip(),
        format="%d.%m.%Y %H:%M:%S,%f"   
    )
    ap_flow.index = pd.to_datetime(
        ap_flow.index.str.strip(),
        format="%d.%m.%Y %H:%M:%S,%f"   
    )
    ap_profile.index = pd.to_datetime(
        ap_profile.index.str.strip(),
        format="%d.%m.%Y %H:%M:%S,%f"   
    )
    
    return ap_spo2, ap_thorac, ap_flow, ap_profile

# %%
def filter_signal(ap_sig, fs):
    sig = ap_sig.iloc[:, 0].to_numpy()
    wl = 0.17/(fs/2)
    wh = 0.4/(fs/2)
    dc = np.mean(sig)
    sig = sig-dc
    b, a = sp.signal.butter(4, (wl, wh), btype = 'band')
    filt_sig = sp.signal.filtfilt(b, a, sig)
    ap_sig[ap_sig.columns[0]] = filt_sig + dc
    return ap_sig
    

# %%
ap01_spo2, ap01_thorac, ap01_flow, ap01_profile = get_individual_signals("AP01", in_dir)
ap02_spo2, ap02_thorac, ap02_flow, ap02_profile = get_individual_signals("AP02", in_dir)
ap03_spo2, ap03_thorac, ap03_flow, ap03_profile = get_individual_signals("AP03", in_dir)
ap04_spo2, ap04_thorac, ap04_flow, ap04_profile = get_individual_signals("AP04", in_dir)
ap05_spo2, ap05_thorac, ap05_flow, ap05_profile = get_individual_signals("AP05", in_dir)

# %%
filt_ap01_spo2 = filter_signal(ap01_spo2, 4)
filt_ap01_thorac = filter_signal(ap01_thorac, 32)
filt_ap01_flow = filter_signal(ap01_flow, 32)
#filt_ap01_profile = filter_signal(ap01_profile, 4)

filt_ap02_spo2 = filter_signal(ap02_spo2, 4)
filt_ap02_thorac = filter_signal(ap02_thorac, 32)
filt_ap02_flow = filter_signal(ap02_flow, 32)
#filt_ap02_profile = filter_signal(ap02_profile, 4)

filt_ap03_spo2 = filter_signal(ap03_spo2, 4)
filt_ap03_thorac = filter_signal(ap03_thorac, 32)
filt_ap03_flow = filter_signal(ap03_flow, 32)
#filt_ap03_profile = filter_signal(ap03_profile, 4)

filt_ap04_spo2 = filter_signal(ap04_spo2, 4)
filt_ap04_thorac = filter_signal(ap04_thorac, 32)
filt_ap04_flow = filter_signal(ap04_flow, 32)
#filt_ap04_profile = filter_signal(ap04_profile, 4)

filt_ap05_spo2 = filter_signal(ap05_spo2, 4)
filt_ap05_thorac = filter_signal(ap05_thorac, 32)
filt_ap05_flow = filter_signal(ap05_flow, 32)
#filt_ap05_profile = filter_signal(ap05_profile, 4)

# %%
def get_events(id, in_dir):
    path = os.path.join(in_dir, id)
    events_path = os.path.join(path, "Flow Events.txt")
    events = pd.read_csv(events_path, sep = ';', skiprows = [0,1,2,3,4,5,6], header = None, names = ["time", "x", "events", "y"])
    events["time"] = events["time"].astype(str)
    events["time"] = events["time"].str.strip()

    # %%
    del events["x"], events["y"]

    # %%
    starts = []
    ends  = []
    for interval in events["time"]:
        start,end = interval.split("-")
        end = start.split()[0] + " " + end
        starts.append(start)
        ends.append(end)


    # %%
    events['start'] = starts
    events['end'] = ends
    events['start'] = pd.to_datetime(
        events['start'].str.strip(),
        format="%d.%m.%Y %H:%M:%S,%f"
    )
    events['end'] = pd.to_datetime(
        events['end'].str.strip(),
        format="%d.%m.%Y %H:%M:%S,%f"
    )

    # %%
    del events["time"]
    return events

# %%
ap01_events = get_events("AP01", in_dir)
ap02_events = get_events("AP02", in_dir)
ap03_events = get_events("AP03", in_dir)
ap04_events = get_events("AP04", in_dir)
ap05_events = get_events("AP05", in_dir)

# %%

ap01 = pd.concat(
    [filt_ap01_thorac, filt_ap01_flow, filt_ap01_spo2, ap01_profile],
    axis=1
).ffill()


# AP02
ap02 = pd.concat(
    [filt_ap02_thorac, filt_ap02_flow, filt_ap02_spo2, ap02_profile],
    axis=1
).ffill()
ap03 = pd.concat([filt_ap03_thorac, filt_ap03_flow, filt_ap03_spo2, ap03_profile], axis=1).ffill()
ap04 = pd.concat([filt_ap04_thorac, filt_ap04_flow, filt_ap04_spo2, ap04_profile], axis=1).ffill()
ap05 = pd.concat([filt_ap05_thorac, filt_ap05_flow, filt_ap05_spo2, ap05_profile], axis=1).ffill()



# %%
def split_into_windows(df, window_size='30s'):
    """
    Splits a time-indexed DataFrame into fixed-duration windows.

    Parameters:
        df : pandas DataFrame with DatetimeIndex
        window_size : string (default '30S' for 30 seconds)

    Returns:
        List of tuples (start_time, end_time, sub_dataframe)
    """
    
    windows = []
    
    # Ensure sorted index
    df = df.sort_index()
    
    # Get window boundaries
    start = df.index.min()
    end = df.index.max()
    
    current_start = start
    inc = '15s'
    while current_start < end:
        current_end = current_start + pd.Timedelta(window_size)
        
        sub_df = df.loc[current_start:current_end]
        
        if sub_df.shape[0]==961:
            windows.append((current_start, current_end, sub_df))
        
        current_start = current_start + pd.Timedelta(inc)
    
    return windows

# %%
ap01_win = split_into_windows(ap01)
ap02_win = split_into_windows(ap02)
ap03_win = split_into_windows(ap03)
ap04_win = split_into_windows(ap04)
ap05_win = split_into_windows(ap05)

print(len(ap01_win))


# %%
ap01_event_groups = ap01_events.groupby("events")
ap02_event_groups = ap02_events.groupby("events")
ap03_event_groups = ap03_events.groupby("events")
ap04_event_groups = ap04_events.groupby("events")
ap05_event_groups = ap05_events.groupby("events")

# %%
def total_overlap(window_start, window_end, events_df):
    """
    Calculates total overlap duration (in seconds)
    between a window and all intervals in events_df.
    """

    # Compute overlap start and end for all rows
    overlap_start = events_df["start"].clip(lower=window_start)
    overlap_end   = events_df["end"].clip(upper=window_end)

    # Compute overlap duration
    overlap = (overlap_end - overlap_start)

    # Keep only positive overlaps
    overlap = overlap[overlap > pd.Timedelta(0)]

    # Return total seconds
    return overlap.sum().total_seconds()

# %%
def gen_dataset(windows, groups):
    X_breath = []
    Y_breath = []
    X_stages = []
    Y_stages = []
    for start, end , df in windows:
        Y_stages.append(df['profile'].iloc[16*32]) #choosing the forward filled sleep stage near the center of the window as the corresponding sleep stage label
        del df['profile']
        X_stages.append(df)
        X_breath.append(df)
        Y_breath.append("Normal")
        for cat, subdf in groups:
            ov = total_overlap(start, end, subdf)
            if ov>15:
                Y_breath[len(Y_breath)-1] = str(cat)
                break
        
    return X_breath, Y_breath, X_stages, Y_stages
            

# %%
ap01_Xb, ap01_Yb, ap01_Xs, ap01_Ys = gen_dataset(ap01_win, ap01_event_groups)
ap02_Xb, ap02_Yb, ap02_Xs, ap02_Ys = gen_dataset(ap02_win, ap02_event_groups)
ap03_Xb, ap03_Yb, ap03_Xs, ap03_Ys = gen_dataset(ap03_win, ap03_event_groups)
ap04_Xb, ap04_Yb, ap04_Xs, ap04_Ys = gen_dataset(ap04_win, ap04_event_groups)
ap05_Xb, ap05_Yb, ap05_Xs, ap05_Ys = gen_dataset(ap05_win, ap05_event_groups)

# %%
out_dir = ".."

# %%
out_path_breath = os.path.join(out_dir, "Dataset/breathing_dataset.csv")
out_path_stages = os.path.join(out_dir, "Dataset/sleep_stage_dataset.csv")

# %%
def write_to_csv(path, X,Y, id):
    for df, label in zip(X,Y):
        flattened = df.to_numpy().flatten()
        row_df = pd.DataFrame([[id]+list(flattened)+[label]])
        row_df.to_csv(path, index = False, header = False, mode = "a")
            
            

# %%
with open(out_path_breath, "w") as f:
    f.write("patient id, 961 by 3 dataset(features thorac, respiratory flow, spo2) flattened to a row, event label \n\n")
write_to_csv(out_path_breath, ap01_Xb, ap01_Yb,"AP01")
write_to_csv(out_path_breath, ap02_Xb, ap02_Yb, "AP02")
write_to_csv(out_path_breath, ap03_Xb, ap03_Yb, "AP03")
write_to_csv(out_path_breath, ap04_Xb, ap04_Yb, "AP04")
write_to_csv(out_path_breath, ap05_Xb, ap05_Yb, "AP05")

with open(out_path_stages, "w") as f:
    f.write("patient id, 961 by 3 dataset(features thorac, respiratory flow, spo2) flattened to a row, sleep stage label \n\n")
write_to_csv(out_path_stages, ap01_Xs, ap01_Ys, "AP01")
write_to_csv(out_path_stages, ap02_Xs, ap02_Ys, "AP02")
write_to_csv(out_path_stages, ap03_Xs, ap03_Ys, "AP03")
write_to_csv(out_path_stages, ap04_Xs, ap04_Ys, "AP04")
write_to_csv(out_path_stages, ap05_Xs, ap05_Ys, "AP05")


