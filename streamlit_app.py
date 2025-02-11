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
import re
import tqdm
import astroquery
import astropy
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

# Simbad query to get coordinates for galaxy names
Simbad.add_votable_fields('ra', 'dec')

galaxy_coords = []

# Retrieve galaxy coordinates, caching the results
coord_df = get_galaxy_coords(df['Name'])

# Merge with the main dataframe
df = df.merge(coord_df, on='Name', how='left')

# Filter out rows with missing coordinates
skyplot_df = df.dropna(subset=['RA', 'Dec'])

# List of methods (replace with your actual methods)
methods = skyplot_df['method'].unique()

# Create a scatter trace for each method
fig = go.Figure()

for method in methods:
    method_df = skyplot_df[skyplot_df['method'] == method]

    if not method_df.empty:
        # Convert RA/Dec to radians for plotting in the Mollweide projection
        coords = SkyCoord(ra=method_df['RA'].values*u.degree, dec=method_df['Dec'].values*u.degree, frame='icrs')

        # Convert RA to hours for proper plotting and display RA in HMS format
        ra_hms = [coord.ra.to_string(unit=u.hourangle, sep=':', precision=2, pad=True) for coord in coords]
        dec_deg = coords.dec.degree  # Dec in degrees

        # Create a scatter plot for each method, using the same color as in the other plots
        fig.add_trace(go.Scattergeo(
            lon=coords.ra.wrap_at(180 * u.deg).deg,  # RA in degrees, wrapped [-180°, 180°]
            lat=dec_deg,
            text=method_df['Name'],  # Object names to show on hover
            mode='markers',
            marker=dict(size=10, color=color_map[method], opacity=0.8, line=dict(width=1, color='black')),  # Use color_map for consistent coloring
            hovertemplate=(
                "RA: %{customdata[0]}<br>" +  # Display RA in HMS format
                "Dec: %{lat:.2f}°<br>" +  # Display Dec in degrees
                "Name: %{text}<extra></extra>"  # Display galaxy name
            ),
            customdata=np.array(ra_hms).reshape(-1, 1),  # Pass RA in HMS format as custom data for hover
            name=method
        ))

# Add celestial equator (Dec = 0°)
celestial_equator_lon = np.linspace(-180, 180, 1000)  # Full range of RA in degrees
celestial_equator_lat = np.zeros_like(celestial_equator_lon)  # Dec = 0° for celestial equator

# Add the equator line trace
fig.add_trace(go.Scattergeo(
    lon=celestial_equator_lon,
    lat=celestial_equator_lat,
    mode='lines',
    line=dict(color='blue', width=2),
    name='Celestial Equator'
))

# Add RA = 0h line (RA = 0°)
ra_zero_line_lon = np.zeros(100)  # RA = 0° (constant longitude)
ra_zero_line_lat = np.linspace(-90, 90, 100)  # Full range of Dec from -90° to +90°

# Add the RA = 0h line trace
fig.add_trace(go.Scattergeo(
    lon=ra_zero_line_lon,  # RA = 0°
    lat=ra_zero_line_lat,  # Dec range from -90° to 90°
    mode='lines',
    line=dict(color='red', width=2, dash='dash'),  # Customize the line color and style
    name='RA = 0h'
))

# Set the layout for the sky plot (Mollweide projection)
fig.update_layout(
    showlegend=True,
    geo=dict(
        projection_type="mollweide",
        showcoastlines=False,
        showland=False,
        showframe=True,
        resolution=50,
        lonaxis=dict(
            showgrid=True,  # Show longitude grid
            gridcolor='lightgray',  # Color of the longitude grid
            tick0=-180,  # Start RA at -180 degrees
            dtick=30,  # Tick every 30 degrees
        ),
        lataxis=dict(
            showgrid=True,  # Show latitude grid
            gridcolor='lightgray',  # Color of the latitude grid
            tick0=-90,  # Start Dec at -90 degrees
            dtick=30,  # Tick every 30 degrees
        ),
        lonaxis_range=[-180, 180],  # Longitude range
        lataxis_range=[-90, 90],  # Latitude range
    ),
    margin=dict(l=0, r=0, t=50, b=0)
)

# Calculate the fraction of galaxies with coordinates
total_galaxies = len(df)
galaxies_with_coords = len(skyplot_df)
fraction_with_coords = galaxies_with_coords / total_galaxies * 100

# Add subtitle
st.subheader(f"Galaxy Locations on Sky (Currently Broken)")
st.text(f'Plot seemed to break when moving to a deployed app. Will fix soon!')
st.text(f'Coordinates found for {galaxies_with_coords}/{total_galaxies} galaxies '
        f'({fraction_with_coords:.2f}%)')

# Display the interactive plot in Streamlit
st.plotly_chart(fig)


st.markdown("""---""")

def references_to_hyperlinks(references):
    """
    Convert references into hyperlinks pointing to ADS using the reference code in parentheses.
    
    Args:
    references (str): A string of references formatted with author names and ADS reference codes.
    
    Returns:
    str: The modified string where each reference is a clickable hyperlink.
    """
    # Define a regex pattern to match the reference code inside parentheses
    pattern = r"\(([^)]+)\)"
    
    # Create a function to replace the matched reference code with a hyperlink
    def add_hyperlink(match):
        ref_code = match.group(1)
        link = f"https://adsabs.harvard.edu/abs/{ref_code}"
        return f"([Link]({link}))"
    
    # Replace each reference code with the corresponding hyperlink
    modified_references = re.sub(pattern, add_hyperlink, references)
    
    return modified_references

st.subheader("References")

# Split the list into two columns
col1, col2 = st.columns(2)

# List of references
references = """
1. Atkinson et al. (2005MNRAS.359..504A)
2. Barth et al. (2001ApJ...546..205B)
3. Barth et al. (2009ApJ...690.1031B)
4. Barth et al. (2011ApJ...743L...4B)
5. Barth et al. (2013ApJ...769..128B)
6. Barth et al. (2016ApJ...822L..28B)
7. Beifiori et al. (2012MNRAS.419.2497B)
8. Bentz et al. (2010ApJ...716..993B)
9. Bentz et al. (2014ApJ...796....8B)
10. Blais-Ouellette et al. (2004A&A...420..147B)
11. Bower et al. (2001ApJ...550...75B)
12. Braatz et al. (1997AAS...19110402B)
13. Capetti et al. (2005A&A...431..465C)
14. Cappellari et al. (2002ApJ...578..787C)
15. Cappellari et al. (2008IAUS..245..215C)
16. Cappellari et al. (2009ApJ...704L..34C)
17. Coccato et al. (2006MNRAS.366.1050C)
18. Cretton & van den Bosch (1999ApJ...514..704C)
19. Dalla Bonta et al. (2009ApJ...690..537D)
20. Davies et al. (2006ApJ...646..754D)
21. Davis et al. (2013Natur.494..328D)
22. de Francesco et al. (2008A&A...479..355D)
23. Denney et al. (2009ApJ...702.1353D)
24. Denney et al. (2010ApJ...721..715D)
25. Devereux et al. (2003AJ....125.1226D)
26. Dietrich et al. (2012ApJ...757...53D)
27. Doroshenko et al. (2008ARep...52..442D)
28. Emsellem et al. (1999MNRAS.303..495E)
29. Ferrarese et al. (1996ApJ...470..444F)
30. Ferrarese & Ford (1999ApJ...515..583F)
31. Gao et al. (2016ApJ...817..128G)
32. Gebhardt et al. (2001AJ....122.2469G)
33. Gebhardt et al. (2007ApJ...671.1321G)
34. Greene et al. (2016arXiv160600018G)
35. Greenhill et al. (1997ApJ...481L..23G)
36. Greenhill et al. (2003ApJ...590..162G)
37. Gultekin et al. (2009ApJ...695.1577G)
38. Gultekin et al. (2011ApJ...738...17G)
39. Gultekin et al. (2014ApJ...781..112G)
40. Haring & Rix (2004ApJ...604L..89H)
41. Herrnstein et al. (2005ApJ...629..719H)
42. Houghton et al. (2006MNRAS.367....2H)
43. Hure et al. (2011A&A...530A.145H)
44. Jardel et al. (2011ApJ...739...21J)
45. Kollatschny et al. (2014A&A...566A.106K)
46. Kondratko et al. (2005ApJ...618..618K)
47. Kondratko et al. (2008ApJ...678...87K)
48. Kormendy et al. (2010ApJ...723...54K)
49. Kormendy et al. (2011Natur.469..374K)
50. Kovavcevic et al. (2014AdSpR..54.1414K)
51. Krajnovic et al. (2009MNRAS.399.1839K)
52. Kuo et al. (2011ApJ...727...20K)
53. Lodato & Bertin (2003A&A...398..517L)
54. Lyubenova et al. (2013MNRAS.431.3364L)
55. Marconi & Hunt (2003ApJ...589L..21M)
56. McConnell et al. (2011ApJ...728..100M)
57. McConnell et al. (2011Natur.480..215M)
58. McConnell et al. (2012ApJ...756..179M)
59. Medling et al. (2011ApJ...743...32M)
60. Merritt et al. (2001Sci...293.1116M)
61. Neumayer & Walcher (2012AdAst2012E..15N)
62. Nguyen et al. (2014ApJ...794...34N)
63. Nowak et al. (2007MNRAS.379..909N)
64. Nowak et al. (2008MNRAS.391.1629N)
65. Nowak et al. (2010MNRAS.403..646N)
66. Onishi et al. (2015ApJ...806...39O)
67. Onishi et al. (MNRAS, submitted)
68. Onken & Peterson (2002ApJ...572..746O)
69. Onken et al. (2014ApJ...791...37O)
70. Pastorini et al. (2007A&A...469..405P)
71. Peterson et al. (2004ApJ...613..682P)
72. Peterson et al. (2014ApJ...795..149P)
73. Pignatelli et al. (2001MNRAS.320..124P)
74. Rusli et al. (2013AJ....146...45R)
75. Saglia et al. (2016ApJ...818...47S)
76. Sarzi et al. (2001ApJ...550...65S)
77. Scharwachter et al. (2013MNRAS.429.2315S)
78. Schulze & Gebhardt (2011ApJ...729...21S)
79. Seth et al. (2010ApJ...714..713S)
80. Shen & Gebhardt (2010ApJ...711..484S)
81. Tadhunter et al. (2003MNRAS.342..861T)
82. Thomas et al. (2016Natur.532..340T)
83. Tremaine et al. (2002ApJ...574..740T)
84. Trotter et al. (1998ApJ...495..740T)
85. Valluri et al. (2005ApJ...628..137V)
86. van der Marel & van den Bosch (1998AJ....116.2220V)
87. van den Bosch & de Zeeuw (2010MNRAS.401.1770V)
88. van den Bosch et al. (2015ApJS..218...10V)
89. Walsh et al. (2010ApJ...721..762W)
90. Walsh et al. (2012ApJ...753...79W)
91. Walsh et al. (2013ApJ...770...86W)
92. Walsh et al. (2015ApJ...808..183W)
93. Walsh et al. (2016ApJ...817....2W)
94. Wold et al. (2006A&A...460..449W)
95. Yamauchi et al. (2004PASJ...56..605Y)
96. Yildirim et al. (2015MNRAS.452.1792Y)
97. Ruffa & Davis (2024Galax..12...36R)
"""

modified_references = references_to_hyperlinks(references)

# Split the references into two parts
references_list = modified_references.strip().split("\n")
mid = len(references_list) // 2
col1_references = references_list[:mid]
col2_references = references_list[mid:]

# Display the references in the two columns
with col1:
    for ref in col1_references:
        st.markdown(ref)

with col2:
    for ref in col2_references:
        st.markdown(ref)
