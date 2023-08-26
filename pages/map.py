import pandas as pd
import streamlit as st
import plotly.express as px

# Load the data
Yangon_cities = pd.read_csv('GHI_yangon_v2.csv')

# App title and initial map display
st.title('Solar Generation Area in Yangon Region')
fig = px.scatter_mapbox(Yangon_cities, lat="Y_Coordinate", lon="X_Coordinate", hover_name="TS_Name", hover_data=["ID", "Category","Shape_Area(sqm)", "GHI_MW","GHI_kWh","Cost_USD(M)"],
                        color_discrete_sequence=["fuchsia"], size_max=15, zoom=10, height=700)
fig.update_layout(mapbox_style="open-street-map")
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
st.plotly_chart(fig, use_container_width=True)
st.markdown("##")

# Title and sidebar for filtering
st.title('Solar Generation Area based on Selected Tsp in Yangon Region')
st.sidebar.title('Please select township')

# Create a radio button to select township
tsp = st.sidebar.radio("What's your interested tsp", ('Dagon Myothit (East)','Dala','Hlegu','Hmawbi','Htantabin','Kawhmu','Kayan','Kungyangon','Kyauktan','Kyeemyindaing','Taikkyi','Thanlyin','Thongwa','Twantay'))


dft = pd.read_csv('Mean_Category.csv')
st.title('Mean value of three clusters in terms of Area, MW and Cost (M)')
st.dataframe(dft)
st.markdown("##")


# Create a multiselect to choose filtering by label column (Category)
Category_column = st.sidebar.radio("Check the mean value of Category and Select Category Column you want", (0, 1, 2))

# Filter data based on the selected label column (Category) and township
filtered_data = Yangon_cities.loc[(Yangon_cities['TS_Name'] == tsp) & (Yangon_cities['Category'] == Category_column)]

# Create a multiselect to choose filtering within the selected category
category_filter = st.sidebar.multiselect("Filter within Category", options=filtered_data['Category'].unique())

# Apply additional filter if needed
if category_filter:
    filtered_data = filtered_data[filtered_data['Category'].isin(category_filter)]
    
# Create a multiselect to choose filtering by MW or cost
filter_by = st.sidebar.multiselect("Filter by", options=['GHI_MW', 'Cost_USD(M)'])

# Apply additional filter if needed
if filter_by:
    if 'GHI_MW' in filter_by:
        min_mw, max_mw = st.sidebar.slider("Select GHI_MW Range", float(filtered_data['GHI_MW'].min()), float(filtered_data['GHI_MW'].max()), (float(filtered_data['GHI_MW'].min()), float(filtered_data['GHI_MW'].max())))
        filtered_data = filtered_data[(filtered_data['GHI_MW'] >= min_mw) & (filtered_data['GHI_MW'] <= max_mw)]
    if 'Cost_USD(M)' in filter_by:
        min_cost, max_cost = st.sidebar.slider("Select Cost Range", float(filtered_data['Cost_USD(M)'].min()), float(filtered_data['Cost_USD(M)'].max()), (float(filtered_data['Cost_USD(M)'].min()), float(filtered_data['Cost_USD(M)'].max())))
        filtered_data = filtered_data[(filtered_data['Cost_USD(M)'] >= min_cost) & (filtered_data['Cost_USD(M)'] <= max_cost)]

# ... (previous code remains unchanged)

# Update the map based on the filtered data if it's not empty
if not filtered_data.empty:
    taifig = px.scatter_mapbox(filtered_data, lat="Y_Coordinate", lon="X_Coordinate", hover_name="TS_Name", hover_data=["ID", "Category","Shape_Area(sqm)", "GHI_MW","GHI_kWh","Cost_USD(M)"],
                            color_discrete_sequence=["fuchsia"], size_max=15, zoom=10, height=700)
    taifig.update_layout(mapbox_style="open-street-map")
    taifig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    st.plotly_chart(taifig, use_container_width=True)
else:
    st.warning("No data available for the selected filters.")

