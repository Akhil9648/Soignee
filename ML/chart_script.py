import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

# Define the components and their properties
components_data = {
    "name": ["User Camera", "Clothing Dataset", "MediaPipe Pose", "Body Segmentation", 
             "Clothing CNN", "TPS Warping", "Style Transfer", "Flask Backend", 
             "HTML/JS Frontend", "Virtual Try-On Result"],
    "type": ["input", "input", "ml_model", "ml_model", "ml_model", "processing", 
             "ml_model", "web", "web", "output"],
    "description": ["Real-time video", "DeepFashion2", "33 landmarks", "Person/bg sep",
                   "Garment classify", "Cloth deform", "Texture map", "REST API", "User interface", "Final image"]
}

# Define connections with directional arrows
connections = [
    ("User Camera", "MediaPipe Pose"),
    ("User Camera", "Body Segmentation"),
    ("Clothing Dataset", "Clothing CNN"),
    ("MediaPipe Pose", "TPS Warping"),
    ("Body Segmentation", "TPS Warping"),  
    ("Clothing CNN", "TPS Warping"),
    ("TPS Warping", "Style Transfer"),
    ("Style Transfer", "Virtual Try-On Result"),
    ("Flask Backend", "HTML/JS Frontend"),
    ("HTML/JS Frontend", "User Camera")
]

# Define colors for different component types
color_map = {
    "ml_model": "#1FB8CD",    # Strong cyan for ML models
    "web": "#2E8B57",         # Sea green for web components  
    "processing": "#D2BA4C",  # Moderate yellow for data processing
    "input": "#DB4545",       # Bright red for inputs
    "output": "#5D878F"       # Cyan for output
}

# Create a more structured layout with clear layers
positions = {
    # Input Layer (left)
    "User Camera": (1, 4),
    "Clothing Dataset": (1, 2),
    
    # Processing Layer 1 (ML preprocessing)
    "MediaPipe Pose": (3, 5),  
    "Body Segmentation": (3, 3),
    "Clothing CNN": (3, 1),
    
    # Processing Layer 2 (core processing)
    "TPS Warping": (5, 3),
    
    # Processing Layer 3 (final ML)
    "Style Transfer": (7, 3),
    
    # Output Layer (right)
    "Virtual Try-On Result": (9, 3),
    
    # Web Layer (bottom)
    "Flask Backend": (2, 0),
    "HTML/JS Frontend": (4, 0)
}

# Create node positions and colors
x_pos = [positions[name][0] for name in components_data["name"]]
y_pos = [positions[name][1] for name in components_data["name"]]
node_colors = [color_map[comp_type] for comp_type in components_data["type"]]

# Create the figure
fig = go.Figure()

# Add directional arrows for connections
for from_node, to_node in connections:
    x0, y0 = positions[from_node]
    x1, y1 = positions[to_node]
    
    # Calculate arrow direction
    dx = x1 - x0
    dy = y1 - y0
    length = np.sqrt(dx**2 + dy**2)
    
    # Normalize and create arrow
    if length > 0:
        dx_norm = dx / length
        dy_norm = dy / length
        
        # Start and end points (adjusted for node size)
        start_x = x0 + 0.3 * dx_norm
        start_y = y0 + 0.3 * dy_norm
        end_x = x1 - 0.3 * dx_norm  
        end_y = y1 - 0.3 * dy_norm
        
        # Add arrow line
        fig.add_trace(go.Scatter(
            x=[start_x, end_x], y=[start_y, end_y],
            mode='lines',
            line=dict(width=3, color='#666'),
            showlegend=False,
            hoverinfo='none'
        ))
        
        # Add arrowhead
        arrow_size = 0.15
        arrow_x = end_x - arrow_size * dx_norm
        arrow_y = end_y - arrow_size * dy_norm
        
        # Perpendicular vector for arrowhead
        perp_x = -dy_norm * arrow_size * 0.5
        perp_y = dx_norm * arrow_size * 0.5
        
        fig.add_trace(go.Scatter(
            x=[arrow_x + perp_x, end_x, arrow_x - perp_x],
            y=[arrow_y + perp_y, end_y, arrow_y - perp_y],
            mode='lines',
            line=dict(width=3, color='#666'),
            fill='toself',
            fillcolor='#666',
            showlegend=False,
            hoverinfo='none'
        ))

# Add component nodes
fig.add_trace(go.Scatter(
    x=x_pos, y=y_pos,
    mode='markers',
    marker=dict(
        size=50,
        color=node_colors,
        line=dict(width=3, color='white')
    ),
    hovertemplate='<b>%{customdata[0]}</b><br>%{customdata[1]}<extra></extra>',
    customdata=list(zip(components_data["name"], components_data["description"])),
    showlegend=False
))

# Add component labels positioned outside nodes
for i, (name, desc) in enumerate(zip(components_data["name"], components_data["description"])):
    fig.add_annotation(
        x=x_pos[i],
        y=y_pos[i] - 0.6,  # Position below the node
        text=f"<b>{name}</b><br><i>{desc}</i>",
        showarrow=False,
        font=dict(size=11, color='black'),
        align="center",
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="rgba(0,0,0,0.2)",
        borderwidth=1
    )

# Add layer labels
layer_labels = [
    (1, 5.5, "Input Layer"),
    (5, 5.5, "Processing Layers"), 
    (9, 5.5, "Output Layer"),
    (3, -0.8, "Web Application Layer")
]

for x, y, label in layer_labels:
    fig.add_annotation(
        x=x, y=y,
        text=f"<b>{label}</b>",
        showarrow=False,
        font=dict(size=14, color='#333'),
        bgcolor="rgba(240,240,240,0.8)",
        bordercolor="rgba(0,0,0,0.3)",
        borderwidth=2
    )

# Add legend for component types with proper spacing
legend_traces = []
type_names = {
    "ml_model": "ML Models",
    "web": "Web Components", 
    "processing": "Data Processing",
    "input": "Input Sources",
    "output": "Output"
}

for comp_type, color in color_map.items():
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='markers',
        marker=dict(size=20, color=color, line=dict(width=2, color='white')),
        name=type_names[comp_type],
        showlegend=True,
        hoverinfo='none'
    ))

# Update layout with better spacing and structure
fig.update_layout(
    title="Virtual Try-On ML System Architecture",
    showlegend=True,
    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
    xaxis=dict(
        showgrid=False, 
        zeroline=False, 
        showticklabels=False,
        range=[0, 10]
    ),
    yaxis=dict(
        showgrid=False, 
        zeroline=False, 
        showticklabels=False,
        range=[-1.5, 6]
    ),
    plot_bgcolor='rgba(248,248,248,0.8)',
    paper_bgcolor='white',
    hovermode='closest'
)

# Save the chart
fig.write_image("virtual_tryon_architecture.png", width=1400, height=900)