import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

# Data from the provided JSON
data = {
    "approaches": [
        {
            "name": "Traditional 2D Overlay",
            "accuracy": 60,
            "processing_speed": 30,
            "setup_complexity": 2,
            "realism_score": 4,
            "cost": "Low"
        },
        {
            "name": "3D Model-Based",
            "accuracy": 85,
            "processing_speed": 5,
            "setup_complexity": 5,
            "realism_score": 9,
            "cost": "High"
        },
        {
            "name": "AI Pose-Aware (Our Solution)",
            "accuracy": 80,
            "processing_speed": 15,
            "setup_complexity": 3,
            "realism_score": 7,
            "cost": "Medium"
        },
        {
            "name": "Commercial Solutions",
            "accuracy": 90,
            "processing_speed": 10,
            "setup_complexity": 4,
            "realism_score": 8,
            "cost": "High"
        }
    ],
    "cost_mapping": {"Low": 1, "Medium": 2, "High": 3}
}

# Brand colors
colors = ['#1FB8CD', '#DB4545', '#2E8B57', '#5D878F', '#D2BA4C']

# Prepare data for radar chart
approaches = []
metrics = ["Accuracy", "Speed (fps)", "Setup Complex", "Realism", "Cost"]

# Normalize data to 0-10 scale for better radar chart visualization
for i, approach in enumerate(data["approaches"]):
    # Abbreviate names to fit 15 char limit
    name_mapping = {
        "Traditional 2D Overlay": "Trad 2D",
        "3D Model-Based": "3D Model",
        "AI Pose-Aware (Our Solution)": "AI Pose-Aware", 
        "Commercial Solutions": "Commercial"
    }
    
    normalized_values = [
        approach["accuracy"] / 10,  # Scale 0-100 to 0-10
        approach["processing_speed"] / 3,  # Scale to roughly 0-10
        approach["setup_complexity"] * 2,  # Scale 1-5 to 2-10
        approach["realism_score"],  # Already 1-10
        data["cost_mapping"][approach["cost"]] * 3.33  # Scale 1-3 to ~3-10
    ]
    
    approaches.append({
        "name": name_mapping[approach["name"]],
        "values": normalized_values,
        "color": colors[i]
    })

# Create radar chart
fig = go.Figure()

for approach in approaches:
    fig.add_trace(go.Scatterpolar(
        r=approach["values"] + [approach["values"][0]],  # Close the polygon
        theta=metrics + [metrics[0]],  # Close the polygon
        fill='toself',
        name=approach["name"],
        line_color=approach["color"],
        fillcolor=approach["color"],
        opacity=0.3,
        cliponaxis=False
    ))

fig.update_layout(
    polar=dict(
        radialaxis=dict(
            visible=True,
            range=[0, 10]
        )),
    showlegend=True,
    title="Virtual Try-On Method Comparison",
    legend=dict(orientation='h', yanchor='bottom', y=1.05, xanchor='center', x=0.5)
)

# Save the chart
fig.write_image("virtual_tryon_comparison.png")