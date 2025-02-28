import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import json
import numpy as np
from pathlib import Path
from tkinter import Tk, Frame, Button, ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler

dataset_dir = Path("dataset_processed")

data = []
invalid_json_files = []


def calculate_angle(p1, p2, p3):
    """Calculate the angle between three points."""
    v1 = np.array([p1['x'] - p2['x'], p1['y'] - p2['y']])
    v2 = np.array([p3['x'] - p2['x'], p3['y'] - p2['y']])

    angle = np.degrees(np.arccos(np.clip(
        np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1.0, 1.0)))

    return angle

def extract_angles_from_json(json_file):
    """Extract joint angles, lateral inclination, and activity from the processed JSON file."""
    with open(json_file, 'r') as f:
        json_data = json.load(f)

    if 'frames' not in json_data:
        print(f"Key 'frames' not found in {json_file.name}. JSON data: {json_data}")
        invalid_json_files.append(json_file.name)
        return

    frames = json_data['frames']
    activity = json_data['metadata'].get('activity', 'unknown')

    for frame in frames:
        landmarks = frame['landmarks']

        left_knee_angle = calculate_angle(landmarks.get('LEFT_HIP', {}), landmarks.get('LEFT_KNEE', {}), landmarks.get('LEFT_ANKLE', {})) if all(key in landmarks for key in ['LEFT_HIP', 'LEFT_KNEE', 'LEFT_ANKLE']) else None
        right_knee_angle = calculate_angle(landmarks.get('RIGHT_HIP', {}), landmarks.get('RIGHT_KNEE', {}), landmarks.get('RIGHT_ANKLE', {})) if all(key in landmarks for key in ['RIGHT_HIP', 'RIGHT_KNEE', 'RIGHT_ANKLE']) else None

        left_shoulder = landmarks.get('LEFT_SHOULDER', {}).get('x')
        right_shoulder = landmarks.get('RIGHT_SHOULDER', {}).get('x')
        left_hip = landmarks.get('LEFT_HIP', {}).get('x')
        right_hip = landmarks.get('RIGHT_HIP', {}).get('x')

        lateral_inclination = calculate_lateral_inclination(left_shoulder, right_shoulder, left_hip, right_hip)

        angles = {
            'LEFT_KNEE_ANGLE': left_knee_angle,
            'RIGHT_KNEE_ANGLE': right_knee_angle,
            'LEFT_WRIST': landmarks.get('LEFT_WRIST', {}).get('x'),
            'RIGHT_WRIST': landmarks.get('RIGHT_WRIST', {}).get('x'),
            'LEFT_SHOULDER': left_shoulder,
            'RIGHT_SHOULDER': right_shoulder,
            'NOSE': landmarks.get('NOSE', {}).get('x'),
            'LATERAL_INCLINATION': lateral_inclination,
            'activity': activity
        }

        data.append(angles)


def calculate_velocity(p1, p2, delta_t):
    """Calculate velocity between two points given the time difference."""
    distance = np.sqrt((p1['x'] - p2['x']) ** 2 + (p1['y'] - p2['y']) ** 2)
    return distance / delta_t

def extract_motion_features(json_file):
    """Extract additional motion-related features from the JSON file."""
    with open(json_file, 'r') as f:
        json_data = json.load(f)

    if 'frames' not in json_data or len(json_data['frames']) < 2:
        print(f"Not enough frames to calculate motion features in {json_file.name}")
        return

    frames = json_data['frames']
    delta_t = 1 / 30

    for i in range(1, len(frames)):
        current_frame = frames[i]
        previous_frame = frames[i - 1]


        left_wrist_velocity = calculate_velocity(
            previous_frame['landmarks'].get('LEFT_WRIST', {}),
            current_frame['landmarks'].get('LEFT_WRIST', {}),
            delta_t
        ) if 'LEFT_WRIST' in current_frame['landmarks'] else None

        right_wrist_velocity = calculate_velocity(
            previous_frame['landmarks'].get('RIGHT_WRIST', {}),
            current_frame['landmarks'].get('RIGHT_WRIST', {}),
            delta_t
        ) if 'RIGHT_WRIST' in current_frame['landmarks'] else None


        data.append({
            'LEFT_WRIST_VELOCITY': left_wrist_velocity,
            'RIGHT_WRIST_VELOCITY': right_wrist_velocity,
            'activity': json_data['metadata'].get('activity', 'unknown')
        })

def calculate_lateral_inclination(left_shoulder, right_shoulder, left_hip, right_hip):
    """Calculate lateral inclination based on shoulder and hip positions."""
    if None in [left_shoulder, right_shoulder, left_hip, right_hip]:
        return None
    shoulder_mid = (left_shoulder + right_shoulder) / 2
    hip_mid = (left_hip + right_hip) / 2
    return shoulder_mid - hip_mid



data = []
invalid_json_files = []
for json_file in dataset_dir.glob('*.json'):

    extract_angles_from_json(json_file)


    extract_motion_features(json_file)

if invalid_json_files:
    print("\nKey 'frames' not found in {json_file.name}. JSON data: {json_data}")
    for invalid_file in invalid_json_files:
        print(invalid_file)
else:
    print("All the files have 'frames'.")


df = pd.DataFrame(data)





def plot_set_1(fig, df):
    """Create first set of plots."""
    axs = fig.subplots(1, 2)
    sns.histplot(df['LEFT_KNEE_ANGLE'], kde=True, ax=axs[0])
    axs[0].set_title('Distribution of Left Knee Angle')

    sns.histplot(df['RIGHT_KNEE_ANGLE'], kde=True, ax=axs[1])
    axs[1].set_title('Distribution of Right Knee Angle')
    fig.tight_layout()

def plot_set_2(fig, df):
    """Create second set of plots."""
    axs = fig.subplots(1, 2)
    sns.histplot(df['LEFT_WRIST'], kde=True, ax=axs[0])
    axs[0].set_title('Distribution of Left Wrist')

    sns.histplot(df['RIGHT_WRIST'], kde=True, ax=axs[1])
    axs[1].set_title('Distribution of Right Wrist')
    fig.tight_layout()

def plot_set_3(fig, df):
    """Create third set of plots."""
    axs = fig.subplots(1, 2)
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=axs[0])
    axs[0].set_title('Correlation Between Angles')

    sns.countplot(x='activity', data=df, ax=axs[1])
    axs[1].set_title('Activity Distribution')
    fig.tight_layout()

def plot_set_4(fig, df):
    """Create fourth set of plots for velocities."""
    axs = fig.subplots(1, 2)

    sns.histplot(df['LEFT_WRIST_VELOCITY'], kde=True, ax=axs[0])
    axs[0].set_title('Distribution of Left Wrist Velocity')

    sns.histplot(df['RIGHT_WRIST_VELOCITY'], kde=True, ax=axs[1])
    axs[1].set_title('Distribution of Right Wrist Velocity')
    fig.tight_layout()

def plot_set_5(fig, df):
    """Create fifth set of plots for lateral inclination and shoulder positions."""
    axs = fig.subplots(1, 3)

    sns.histplot(df['LATERAL_INCLINATION'], kde=True, ax=axs[0])
    axs[0].set_title('Distribution of Lateral Inclination')

    sns.histplot(df['LEFT_SHOULDER'], kde=True, ax=axs[1])
    axs[1].set_title('Distribution of Left Shoulder Position')

    sns.histplot(df['RIGHT_SHOULDER'], kde=True, ax=axs[2])
    axs[2].set_title('Distribution of Right Shoulder Position')

    fig.tight_layout()




def display_plots():
    root = Tk()
    root.title("Parkinson AI Plots")

    notebook = ttk.Notebook(root)
    notebook.pack(fill="both", expand=True)

    tab1 = Frame(notebook)
    tab2 = Frame(notebook)
    tab3 = Frame(notebook)
    tab4 = Frame(notebook)
    tab5 = Frame(notebook)

    notebook.add(tab1, text="Knee Angles")
    notebook.add(tab2, text="Wrist Angles")
    notebook.add(tab3, text="Correlation & Activity")
    notebook.add(tab4, text="Wrist Velocities")
    notebook.add(tab5, text="Lateral Inclination & Shoulders")

    fig1 = plt.Figure(figsize=(10, 8), dpi=100)
    fig2 = plt.Figure(figsize=(10, 8), dpi=100)
    fig3 = plt.Figure(figsize=(10, 8), dpi=100)
    fig4 = plt.Figure(figsize=(10, 8), dpi=100)
    fig5 = plt.Figure(figsize=(10, 8), dpi=100)


    canvas1 = FigureCanvasTkAgg(fig1, master=tab1)
    canvas1.draw()
    canvas1.get_tk_widget().pack(fill="both", expand=True)
    plot_set_1(fig1, df)

    canvas2 = FigureCanvasTkAgg(fig2, master=tab2)
    canvas2.draw()
    canvas2.get_tk_widget().pack(fill="both", expand=True)
    plot_set_2(fig2, df)

    canvas3 = FigureCanvasTkAgg(fig3, master=tab3)
    canvas3.draw()
    canvas3.get_tk_widget().pack(fill="both", expand=True)
    plot_set_3(fig3, df)

    canvas4 = FigureCanvasTkAgg(fig4, master=tab4)
    canvas4.draw()
    canvas4.get_tk_widget().pack(fill="both", expand=True)
    plot_set_4(fig4, df)

    canvas5 = FigureCanvasTkAgg(fig5, master=tab5)
    canvas5.draw()
    canvas5.get_tk_widget().pack(fill="both", expand=True)
    plot_set_5(fig5, df)


    root.mainloop()


display_plots()