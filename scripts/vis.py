# %%
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import argparse
import os

# %%
parser = argparse.ArgumentParser()
parser.add_argument("-name", required=True, help="Path to participant folder")
args = parser.parse_args()

path = args.name
id = os.path.basename(os.path.normpath(path))

spo2_path = os.path.join(path,"SPO2.txt")
thorac_path = os.path.join(path,"Thorac.txt")
flow_path = os.path.join(path,"Flow.txt")
events_path = os.path.join(path, "Flow Events.txt")

# %%
ap_spo2 = pd.read_csv(spo2_path, sep = ';', skiprows = [0,1,2,3,4,5,6], header = None, names = ["time", "SPO2"], index_col = ["time"])
ap_thorac = pd.read_csv(thorac_path, sep = ';',  skiprows = [0,1,2,3,4,5,6], header = None, names = ["time","thorac"],  index_col = ["time"])
ap_flow = pd.read_csv(flow_path, sep = ';', skiprows = [0,1,2,3,4,5,6], header = None, names = ["time", "flow"], index_col = ["time"])

# %%
ap = pd.merge(ap_thorac, ap_flow, how = 'outer',left_index = True, right_index = True)
ap = pd.merge(ap, ap_spo2, how = 'outer',left_index = True, right_index = True)
ap = ap.ffill()

# %%
ap.index = pd.to_datetime(
    ap.index.str.strip(),
    format="%d.%m.%Y %H:%M:%S,%f"
)

# %%
events = pd.read_csv(events_path, sep = ';', skiprows = [0,1,2,3,4,5,6], header = None )

# %%
events[0] = events[0].astype(str)
events[0] = events[0].str.strip()

# %%
del events[1], events[3]

# %%
starts = []
ends  = []
for interval in events[0]:
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
del events[0]
#events.head()

# %%
event_groups = events.groupby(2)

# %%
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Ensure datetime
#ap01["time"] = pd.to_datetime(ap01["time"])
#ap01 = ap01.sort_values("time").set_index("time")

window = pd.Timedelta(minutes=5)
op_file = "Visualizations/"+str(id).strip()+"_visualisations.pdf"
start = ap.index.min()
end = ap.index.max()
print(end)
current = start
with PdfPages(op_file) as pdf:
    while current < end:
        next_time = current + window
        df_window = ap.loc[current:next_time]

        if df_window.empty:
            current = next_time
            continue

        fig, axes = plt.subplots(
            3, 1,
            figsize=(12, 8),
            sharex=True
        )

        # -------- Plot 1 --------
        axes[0].plot(df_window.index, df_window["flow"])
        axes[0].set_ylabel("Nasal Flow (L/min)")
        axes[0].set_title("Nasal Flow")

        # -------- Plot 2 --------
        axes[1].plot(df_window.index, df_window["thorac"])
        axes[1].set_ylabel("Resp. Amplitude")
        axes[1].set_title("Thoracic/Abdominal Resp.")

        # -------- Plot 3 --------
        axes[2].plot(df_window.index, df_window["SPO2"])
        axes[2].set_ylabel("SpO2 (%)")
        axes[2].set_title("SpO2")

        # -------- X-axis formatting --------
        axes[2].xaxis.set_major_locator(mdates.SecondLocator(interval=10))
        axes[2].xaxis.set_major_formatter(
            mdates.DateFormatter('%d %H:%M:%S')
        )

        fig.autofmt_xdate()

        plt.suptitle(
            f"{str(id)} - {current.strftime('%Y-%m-%d %H:%M')} "
            f"to {next_time.strftime('%Y-%m-%d %H:%M')}"
        )
        window_events = events[
        (events["start"] <= next_time) &
        (events["end"] >= current)
        ]
        event_groups = window_events.groupby(2)

#event_groups = window_events.groupby("category")

# ---- Overlay on Nasal Flow plot (axes[0]) ----
        for category, df_event in event_groups:

            for _, row in df_event.iterrows():

            # Clip event to current window boundaries
                s = max(row["start"], current)
                e = min(row["end"], next_time)

                if category.lower() == "obstructive apnea":
                    color = "red"
                    alpha = 0.3
                    axes[0].axvspan(
                    s,
                    e,
                    color=color,
                    alpha=alpha
                )
                elif category.lower() == "hypopnea":
                    color = "orange"
                    alpha = 0.3
                    axes[0].axvspan(
                    s,
                    e,
                    color=color,
                    alpha=alpha
                )

        plt.tight_layout()
        
        pdf.savefig(fig)
        plt.close(fig)

        current = next_time
    
print(current)


