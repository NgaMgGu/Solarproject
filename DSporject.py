import matplotlib
matplotlib.use("TkAgg")  # Use the TkAgg backend
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import streamlit as st
import plotly.express as px

st.title('The Project of Solar Energy Assessemnt in the Yangon Region')
df = pd.read_csv(r'C:\\Users\\Asus\\Desktop\\project\\solarprjv3\\GHI_yangon.csv')
st.subheader('Overview of DataSet')
# Printing the dataswet shape
print ("Dataset Length: ", len(df))
print ("Dataset Shape: ", df.shape)

# Display the dataset shape using Streamlit
st.write("Dataset Length:", len(df))
st.write("Dataset Shape:", df.shape)


st.dataframe(df)
st.markdown("##")
st.sidebar.title('Select check box')

shapearea=st.sidebar.checkbox('Shape_Area(sqm)')
ghikWh=st.sidebar.checkbox('GHI_kWh')
ghimw=st.sidebar.checkbox('GHI_MW')
    
#making 3 cols left_column, middle_column, right_column

shapearea_column,ghikWh_column,ghimw_column= st.columns(3)




shapearea= px.box(df, y="Shape_Area(sqm)") 
ghikwh= px.box(df, y="GHI_kWh")
ghiwt= px.box(df, y="GHI_MW")


          
        

                 
if shapearea:
    with shapearea_column:
        st.subheader("Boxplot for Size of Area (sqm)")
        shapearea_column.plotly_chart(shapearea,use_container_width=True)
        
if ghikWh:
    with ghikWh_column:
        st.subheader("Boxplot for Electricity Potential (kWh)")
        ghikWh_column.plotly_chart(ghikwh,use_container_width=True)        
if ghimw:
    with ghimw_column:
        st.subheader("Box plot for Power (MW)")
        ghimw_column.plotly_chart(ghiwt,use_container_width=True)






