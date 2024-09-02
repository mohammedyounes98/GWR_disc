import numpy as np
import plotly.graph_objects as go
from scipy.spatial.distance import cdist

# Function to calculate kernel weights
def calculate_kernel_weights(distances, bandwidth, power=1):
    """Calculates kernel weights based on distances, bandwidth, and power."""
    weights = np.zeros_like(distances)
    within_bandwidth = distances <= bandwidth
    weights[within_bandwidth] = (1 - (distances[within_bandwidth] / bandwidth) ** power) ** power
    return weights

# Create the data for the 3D surface plot
def create_3d_surface_data(bandwidth=1.0, power=1):
    """Creates data for a 3D surface plot representing the GWR kernel disc."""

    # Create a grid of points
    x, y = np.mgrid[-2:2:0.05, -2:2:0.05]
    coords = np.vstack((x.ravel(), y.ravel())).T

    # Calculate distances from the center
    center = np.array([0, 0])
    distances = cdist(coords, center.reshape(1, -1))

    # Calculate kernel weights
    weights = calculate_kernel_weights(distances, bandwidth, power)

    # Reshape the weights to match the grid
    z = weights.reshape(x.shape)

    return x, y, z

# Create the 3D surface plot
def plot_gwr_kernel_3d_surface(bandwidth=1.0, power=1):
    """Plots a 3D surface representing the GWR kernel disc using Plotly."""

    x, y, z = create_3d_surface_data(bandwidth, power)

    # Create the surface plot
    fig = go.Figure(data=[go.Surface(z=z, x=x, y=y, colorscale='Viridis')])

    # Update layout
    fig.update_layout(
        title='Interactive 3D GWR Kernel Disc',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Weight'
        )
    )

    # Add sliders for interactivity
    fig.update_layout(
        sliders=[
            dict(
                active=0,
                currentvalue={"prefix": "Bandwidth: "},
                pad={"t": 50},
                steps=[
                    dict(
                        label=str(bw),
                        method="update",
                        args=[{"z": [create_3d_surface_data(bandwidth=bw, power=power)[2]]}]
                    ) for bw in np.arange(0.1, 2.1, 0.1)
                ]
            ),
            dict(
                active=0,
                currentvalue={"prefix": "Distance Decay (Power): "},
                pad={"t": 50},
                steps=[
                    dict(
                        label=str(p),
                        method="update",
                        args=[{"z": [create_3d_surface_data(bandwidth=bandwidth, power=p)[2]]}]
                    ) for p in np.arange(0.1, 3.1, 0.1)
                ]
            )
        ]
    )

    fig.show()

# Initial 3D surface plot
plot_gwr_kernel_3d_surface()
