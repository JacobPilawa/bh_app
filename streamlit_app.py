import streamlit as st
st.html("""
    <style>
        .stMainBlockContainer {
            max-width:55rem;
        }
    </style>
    """
)
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from astroquery.simbad import Simbad
from astropy.coordinates import SkyCoord
import astropy.units as u
import matplotlib.pyplot as plt
#from astropy.visualization.wcsaxes import WCSAxes


# Function to read data
def read_data():
    colspecs = [(0, 12), (13, 14), (15, 20), (21, 25), (26, 31), 
                (32, 36), (37, 41), (42, 46), (47, 51), (52, 57), 
                (58, 62), (63, 68), (69, 73), (74, 78), (79, 83), 
                (84, 87), (88, 93), (94, 98), (99, 105), (106, 114)]
    
    column_names = [
        "Name", "f_Name", "Dist", "e_Dist", "BHMass", 
        "E_BHMass", "e_BHMass", "sigma", "e_sigma", "logLk", 
        "e_logLk", "Re", "e_Re", "covar", "C28", 
        "e_C28", "AGN", "e_AGN", "method", "Refs"
    ]
    
    df = pd.read_fwf('tabledata.txt', colspecs=colspecs, header=None, names=column_names, na_values='?', skiprows=136)
    return df
    
# Caching Simbad query to avoid rerunning it multiple times
@st.cache_data
def get_galaxy_coords(names):
    Simbad.add_votable_fields('ra', 'dec')
    galaxy_coords = []
    
    for name in names:
        result = Simbad.query_object(name)
        if result is not None:
            ra = result['RA'][0]
            dec = result['DEC'][0]
            coord = SkyCoord(ra, dec, unit=(u.hourangle, u.deg))
            galaxy_coords.append({'Name': name, 'RA': coord.ra.deg, 'Dec': coord.dec.deg})
        else:
            galaxy_coords.append({'Name': name, 'RA': None, 'Dec': None})
    
    return pd.DataFrame(galaxy_coords)


# Read data
df = read_data()

# Scatter Plot: BHMass vs Sigma with Asymmetric Errors (using Plotly)
st.title("SMBH Mass Measurements")

# Add a bit of background information
st.markdown("""
The data table below currently includes:
* [Table 2 from "Unification of the fundamental plane and Super Massive Black Hole Masses"](https://ui.adsabs.harvard.edu/abs/2016ApJ...831..134V/abstract)
* [Figure 8 (Red Points) from "Molecular Gas Kinematics in Local Early-Type Galaxies with ALMA"](https://ui.adsabs.harvard.edu/abs/2024Galax..12...36R/abstract)
""")


########################################
# Display the dataframe
st.subheader("Data")
st.dataframe(df)
st.markdown(":red[1. ] Name (---) Object name; :red[2. ] f_Name (---) Omission flag (1); :red[3. ] Dist (Mpc) Distance; :red[4. ] e_Dist (Mpc) Uncertainty in distance; :red[5. ] BHMass (log [M☉]) Black Hole mass in log solar units; :red[6. ] E_BHMass (log [M☉]) Upper uncertainty in BHMass; :red[7. ] e_BHMass (log [M☉]) Lower uncertainty in BHMass; :red[8. ] σ (log [km/s]) Stellar velocity dispersion inside Re, log km/s; :red[9. ] e_σ (log [km/s]) Uncertainty in σ; :red[10.] logLk (log [L☉]) Total luminosity in log solar units; :red[11.] e_logLk (log [L☉]) Uncertainty in logLk; :red[12.] Re (log [kpc]) Half-light radius in log kpc; :red[13.] e_Re (log [kpc]) Uncertainty in Re; :red[14.] covar (---) Covariance between Lk and Re; :red[15.] C28 (---) Concentration, 5logR20/R80; :red[16.] e_C28 (---) Uncertainty in C28; :red[17.] AGN (log [L☉]) Non-stellar AGN flux, log solar units (optional); :red[18.] e_AGN (log [L☉]) Uncertainty in AGN flux (optional); :red[19.] method (---) Method used for BH mass measurement; :red[20.] Refs (---) Literature references;")

# Scatter plot
st.subheader("$M_{BH}$-$\sigma$")

# Group data by 'method' for coloring purposes
methods = df['method'].unique()
color_map = {method: px.colors.qualitative.Set1[i % len(px.colors.qualitative.Set1)] for i, method in enumerate(methods)}

fig = go.Figure()

# Plot each method with its own color and black outlines
for method in methods:
    method_df = df[df['method'] == method]

    # Add traces with black marker outlines
    fig.add_trace(go.Scatter(
        x=method_df['sigma'],
        y=method_df['BHMass'],
        mode='markers',
        marker=dict(color=color_map[method], size=10, line=dict(width=1, color='black')),  # black outlines
        name=method,
        error_x=dict(type='data', array=method_df['e_sigma']),
        error_y=dict(type='data', array=method_df['E_BHMass'], arrayminus=method_df['e_BHMass']),
        hovertemplate='<b>%{text}</b><br>Sigma: %{x}<br>BH Mass: %{y}',
        text=method_df['Name']
    ))

# Add the MM13 relation:
def mm13(x):
    return 8.32 + 5.64 * np.log10(10**x / 200)

x_vals = np.linspace(df['sigma'].min(), df['sigma'].max(), 500)
x_min, x_max = x_vals.min(), x_vals.max()
y_vals = mm13(x_vals)


fig.add_trace(go.Scatter(
    x=x_vals,
    y=y_vals,
    mode='lines',
    line=dict(color='black', width=2,),
    name='MM13',
    hoverinfo='skip'  
))

fig.update_xaxes(range=[0.9*x_min, 1.05*x_max])
fig.update_yaxes(range=[df['BHMass'].min() - 0.1 * (df['BHMass'].max() - df['BHMass'].min()), df['BHMass'].max() + 0.1 * (df['BHMass'].max() - df['BHMass'].min())])

fig.update_layout(
    xaxis_title='log<sub>10</sub>(σ / km s<sup>-1</sup>)',
    yaxis_title='log<sub>10</sub>(M<sub>BH</sub> / M<sub>&#9737;</sub>)',
    height=600,
    margin=dict(l=50, r=50, t=50, b=50),
)

st.plotly_chart(fig, use_container_width=True)

# Bar chart and histogram
st.subheader("Method Counts and Distances")

col1, col2 = st.columns(2)

with col1:
    method_counts = df['method'].value_counts()
    n_total_methods = method_counts.sum()

    # Create a bar chart with individual text for each bar
    bar_fig = go.Figure(data=[
        go.Bar(
            x=method_counts.index,
            y=method_counts.values,
            text=[f"n={n}" for n in method_counts.values],  # Set 'n=xxxx' for each bar
            textposition='outside',
            marker_color=[color_map[method] for method in method_counts.index],  # Use custom colors
        )
    ])

    # Set layout for the bar chart
    bar_fig.update_layout(
        title="Distribution of Methods",
        xaxis_title="Method",
        yaxis_title="Count",
        showlegend=False,
        yaxis=dict(range=[0, method_counts.max() * 1.2]),  # Increase the y-axis range by 20%
    )

    # Add 'n_tot=xxxx' to the upper right of the plot
    bar_fig.add_annotation(
        text=f"n<sub>tot</sub>={n_total_methods}",
        xref="paper", yref="paper",
        x=0.95, y=0.95,
        showarrow=False,
        font=dict(size=12, color="black")
    )

    st.plotly_chart(bar_fig, use_container_width=True)


with col2:
    n_total_hist = len(df['Dist'])

    # Create a histogram
    hist_fig = px.histogram(
        df, x='Dist', nbins=20, 
        title="Histogram of Distances", 
        labels={'Dist': 'Distance (Mpc)'}
    )

    # Add 'n_tot=xxxx' to the upper right of the plot
    hist_fig.add_annotation(
        text=f"n<sub>tot</sub>={n_total_methods}",
        xref="paper", yref="paper",
        x=0.95, y=0.95,
        showarrow=False,
        font=dict(size=12, color="black")
    )

    st.plotly_chart(hist_fig, use_container_width=True)


########################################
# New Feature: Galaxy Locations on Sky
########################################
st.subheader("Galaxy Locations on Sky (Currently Broken)")
st.text('Plot seemed to break when moving to a deployed app. Will fix soon!')

# # Simbad query to get coordinates for galaxy names
# Simbad.add_votable_fields('ra', 'dec')
#
# galaxy_coords = []
#
# for name in df['Name']:
#     result = Simbad.query_object(name)
#     if result is not None:
#         ra = result['RA'][0]
#         dec = result['DEC'][0]
#         coord = SkyCoord(ra, dec, unit=(u.hourangle, u.deg))
#         galaxy_coords.append({'Name': name, 'RA': coord.ra.deg, 'Dec': coord.dec.deg})
#     else:
#         galaxy_coords.append({'Name': name, 'RA': None, 'Dec': None})
#
# # Retrieve galaxy coordinates, caching the results
# coord_df = get_galaxy_coords(df['Name'])
#
# # Merge with the main dataframe
# df = df.merge(coord_df, on='Name', how='left')
#
# # Filter out rows with missing coordinates
# skyplot_df = df.dropna(subset=['RA', 'Dec'])
#
# # List of methods (replace with your actual methods)
# methods = skyplot_df['method'].unique()
#
# # Create a scatter trace for each method
# fig = go.Figure()
#
# for method in methods:
#     method_df = skyplot_df[skyplot_df['method'] == method]
#
#     if not method_df.empty:
#         # Convert RA/Dec to radians for plotting in the Mollweide projection
#         coords = SkyCoord(ra=method_df['RA'].values*u.degree, dec=method_df['Dec'].values*u.degree, frame='icrs')
#
#         # Convert RA to hours for proper plotting and display RA in HMS format
#         ra_hms = [coord.ra.to_string(unit=u.hourangle, sep=':', precision=2, pad=True) for coord in coords]
#         dec_deg = coords.dec.degree  # Dec in degrees
#
#         # Create a scatter plot for each method, using the same color as in the other plots
#         fig.add_trace(go.Scattergeo(
#             lon=coords.ra.wrap_at(180 * u.deg).deg,  # RA in degrees, wrapped [-180°, 180°]
#             lat=dec_deg,
#             text=method_df['Name'],  # Object names to show on hover
#             mode='markers',
#             marker=dict(size=10, color=color_map[method], opacity=0.8, line=dict(width=1, color='black')),  # Use color_map for consistent coloring
#             hovertemplate=(
#                 "RA: %{customdata[0]}<br>" +  # Display RA in HMS format
#                 "Dec: %{lat:.2f}°<br>" +  # Display Dec in degrees
#                 "Name: %{text}<extra></extra>"  # Display galaxy name
#             ),
#             customdata=np.array(ra_hms).reshape(-1, 1),  # Pass RA in HMS format as custom data for hover
#             name=method
#         ))
#
# # Add celestial equator (Dec = 0°)
# celestial_equator_lon = np.linspace(-180, 180, 1000)  # Full range of RA in degrees
# celestial_equator_lat = np.zeros_like(celestial_equator_lon)  # Dec = 0° for celestial equator
#
# # Add the equator line trace
# fig.add_trace(go.Scattergeo(
#     lon=celestial_equator_lon,
#     lat=celestial_equator_lat,
#     mode='lines',
#     line=dict(color='blue', width=2),
#     name='Celestial Equator'
# ))
#
# # Add RA = 0h line (RA = 0°)
# ra_zero_line_lon = np.zeros(100)  # RA = 0° (constant longitude)
# ra_zero_line_lat = np.linspace(-90, 90, 100)  # Full range of Dec from -90° to +90°
#
# # Add the RA = 0h line trace
# fig.add_trace(go.Scattergeo(
#     lon=ra_zero_line_lon,  # RA = 0°
#     lat=ra_zero_line_lat,  # Dec range from -90° to 90°
#     mode='lines',
#     line=dict(color='red', width=2, dash='dash'),  # Customize the line color and style
#     name='RA = 0h'
# ))
#
# # Set the layout for the sky plot (Mollweide projection)
# fig.update_layout(
#     showlegend=True,
#     geo=dict(
#         projection_type="mollweide",
#         showcoastlines=False,
#         showland=False,
#         showframe=True,
#         resolution=50,
#         lonaxis=dict(
#             showgrid=True,  # Show longitude grid
#             gridcolor='lightgray',  # Color of the longitude grid
#             tick0=-180,  # Start RA at -180 degrees
#             dtick=30,  # Tick every 30 degrees
#         ),
#         lataxis=dict(
#             showgrid=True,  # Show latitude grid
#             gridcolor='lightgray',  # Color of the latitude grid
#             tick0=-90,  # Start Dec at -90 degrees
#             dtick=30,  # Tick every 30 degrees
#         ),
#         lonaxis_range=[-180, 180],  # Longitude range
#         lataxis_range=[-90, 90],  # Latitude range
#     ),
#     margin=dict(l=0, r=0, t=50, b=0)
# )
#
# # Display the interactive plot in Streamlit
# st.plotly_chart(fig)
