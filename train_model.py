import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import json
import numpy as np
from pathlib import Path
from tkinter import Tk, Frame, Button
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

dataset_dir = Path("dataset_processed")

data = []

def calculate_angle(p1, p2, p3):
    """Calculate the angle between three points."""
    v1 = np.array([p1['x'] - p2['x'], p1['y'] - p2['y']])
    v2 = np.array([p3['x'] - p2['x'], p3['y'] - p2['y']])
    
    angle = np.degrees(np.arccos(np.clip(
        np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1.0, 1.0)))
    
    return angle

def extract_angles_from_json(json_file):
    """Extract joint angles and activity from the processed JSON file."""
    with open(json_file, 'r') as f:
        json_data = json.load(f)
    
  
    if 'frames' not in json_data:
        print(f"Key 'frames' not found in {json_file.name}. JSON data: {json_data}")
        return
    
    frames = json_data['frames']
    activity = json_data['metadata'].get('activity', 'unknown') 
    
    for frame in frames:
        landmarks = frame['landmarks']
        
      
        if all(key in landmarks for key in ['LEFT_HIP', 'LEFT_KNEE', 'LEFT_ANKLE']):
            left_knee_angle = calculate_angle(landmarks['LEFT_HIP'], landmarks['LEFT_KNEE'], landmarks['LEFT_ANKLE'])
        else:
            left_knee_angle = None
        
        if all(key in landmarks for key in ['RIGHT_HIP', 'RIGHT_KNEE', 'RIGHT_ANKLE']):
            right_knee_angle = calculate_angle(landmarks['RIGHT_HIP'], landmarks['RIGHT_KNEE'], landmarks['RIGHT_ANKLE'])
        else:
            right_knee_angle = None
        
        angles = {
            'LEFT_KNEE_ANGLE': left_knee_angle,
            'RIGHT_KNEE_ANGLE': right_knee_angle,
            'activity': activity
        }
        
        data.append(angles)  




for json_file in dataset_dir.glob('*.json'):
    extract_angles_from_json(json_file)


df = pd.DataFrame(data)


def plot_if_exists(ax, df, column_name, title):
    """Plot the histogram of a column if it exists in the DataFrame."""
    if column_name in df.columns:
        sns.histplot(df[column_name], kde=True, ax=ax)
        ax.set_title(title)
    else:
        ax.set_title(f"Column {column_name} does not exist in the DataFrame. Skipping this plot.")



def create_plots(fig, df):
    """Generate all plots on the given figure."""
    axs = fig.subplots(3, 2) 
    
 
    plot_if_exists(axs[0, 0], df, 'LEFT_KNEE_ANGLE', 'Distribution of Left Knee Angle')
    plot_if_exists(axs[0, 1], df, 'RIGHT_KNEE_ANGLE', 'Distribution of Right Knee Angle')
    plot_if_exists(axs[1, 0], df, 'LEFT_ELBOW_ANGLE', 'Distribution of Left Elbow Angle')
    plot_if_exists(axs[1, 1], df, 'RIGHT_ELBOW_ANGLE', 'Distribution of Right Elbow Angle')
    
  
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=axs[2, 0])
    axs[2, 0].set_title('Correlation Between Angles')
    
 
    sns.countplot(x='activity', data=df, ax=axs[2, 1])
    axs[2, 1].set_title('Activity Distribution')
    

    fig.tight_layout()



def display_plots():
    root = Tk()
    root.title("Parkinson AI Plots")
    
   
    frame = Frame(root)
    frame.pack(fill="both", expand=True)
    

    fig = plt.Figure(figsize=(10, 8), dpi=100)
    canvas = FigureCanvasTkAgg(fig, master=frame) 
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True)
    

    create_plots(fig, df)
    

    button = Button(root, text="Close", command=root.destroy)
    button.pack(pady=10)
    
    root.mainloop()


display_plots()
