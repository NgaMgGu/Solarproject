from sklearn.cluster import KMeans
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv(r'C:\Users\toe\Desktop\New folder\data\b4prj\solarprjv1\GHI_yangon.csv')
dataindex= df.iloc[:, [11]]
print(dataindex)

st.title('Create Solar Power Genereation Group Using Kmean')
#where Yi is centroid for observation Xi.
#The main goal is to maximize number of clusters and in limiting case each data point becomes its own cluster centroid.
#Compute K-Means clustering for different values of K by varying K from 1 to 10 clusters.
#For each K, calculate the total within-cluster sum of square (WCSS).
#Plot the curve of WCSS vs the number of clusters K.
#The location of a bend (knee) in the plot is generally considered as an indicator of the appropriate number of clusters.
st.header('Chose number of cluster using elbow Method')
wcss = []
for k in range(1,11):
    kmeans = KMeans(n_clusters=k, init="k-means++")
    kmeans.fit(df.iloc[:,11:])
    wcss.append(kmeans.inertia_)
plt.figure(figsize=(12,6))
plt.grid()
plt.plot(range(1,11),wcss, linewidth=2, color="red", marker ="8")
plt.xlabel("K Value")
plt.xticks(np.arange(1,11,1))
plt.ylabel("WCSS")
st.pyplot(plt.gcf())

# Features for 3D plot
features_3d = ["Shape_Area", "GHI_suitability", "GHI_MW"]

# Create the 3D scatter plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# KMeans clustering
km = KMeans(n_clusters=3)
clusters = km.fit(df[features_3d])
clusters1 = clusters.predict(df[features_3d])
df["label"] = clusters1

# Print selected columns and cluster labels
table_data = df[features_3d + ["label"]]
print(table_data)

st.header('Plot data points in 3D with different colors for each cluster')

# Plot data points in 3D with different colors for each cluster
for label in df["label"].unique():
   
    ax.scatter(df.loc[df["label"] == label, features_3d[0]],
               df.loc[df["label"] == label, features_3d[1]],
               df.loc[df["label"] == label, features_3d[2]],
               label=f'Cluster {label}')

ax.set_xlabel(features_3d[0])
ax.set_ylabel(features_3d[1])
ax.set_zlabel(features_3d[2])
ax.legend()

st.pyplot(fig)

st.title('Cluster the new circustances')

area = st.sidebar.number_input('Shape_Area')
ghisu = st.sidebar.number_input('GHI_suitability')
ghimw = st.sidebar.number_input('GHI_MW')

data = {'Shape_Area': area,
            'GHI_suitability': ghisu,
            'GHI_MW': ghimw}
features = pd.DataFrame(data, index=[0])
#features['Shape_Area'] = features['Shape_Area'].astype(str).astype(float)
#features['GHI_suitability'] = features['GHI_suitability'].astype(str).astype(float)
#features['GHI_MW'] = features['GHI_MW'].astype(str).astype(float)
#features = features.astype({'Shape_Area':'float', 'GHI_suitability':float, 'GHI_MW':float})
st.write(area)
clusters2 = clusters.predict(features)
if (clusters2==0.0):
    st.write('The Solar generaton category is category 1')
elif (clusters2==2):
    st.write('The Solar generaton category is category 2')
else:
    st.write('The Solar generaton category is category 3')


st.markdown("##")


# Load the DataFrame from the provided data
data = {
    'LCOE (2021 USD/kWh)': ['5th percentile', 'Weighted average', '95th percentile'],
    2010: [0.208229, 0.417149, 0.521406],
    2011: [0.175056, 0.311298, 0.517898],
    2012: [0.140173, 0.232633, 0.415521],
    2013: [0.121647, 0.179401, 0.375253],
    2014: [0.098981, 0.161258, 0.357170],
    2015: [0.079001, 0.121080, 0.285726],
    2016: [0.073099, 0.106340, 0.240542],
    2017: [0.053062, 0.083660, 0.209989],
    2018: [0.048439, 0.071139, 0.194411],
    2019: [0.046283, 0.062119, 0.165319],
    2020: [0.038281, 0.055444, 0.161514],
    2021: [0.029304, 0.048346, 0.119826]
}

df = pd.DataFrame(data)

# Filter the DataFrame for years 2010 to 2021
years = list(range(2010, 2022))
filtered_df = df[df.columns.intersection(years)]

# Extract 'x' and 'y' values
x_values = filtered_df.columns.tolist()  # Years
y_values = filtered_df.loc[1, :].tolist()  # Weighted average LCOE values

print("Years:", x_values)
print("Weighted average LCOE values:", y_values)
x=x_values
y=y_values



import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt


# Load the DataFrame from the provided data
# ... (Your previous code for loading data and extracting x_values and y_values)

# Reshape x_values to a 2D array
x = np.array(x_values).reshape(-1, 1)

# Create a LinearRegression model and fit it
model = LinearRegression()
model.fit(x, y)

# Predict LCOE value for the year 2023
year_2023 = np.array([2023]).reshape(-1, 1)  # Reshape for prediction
predicted_lcoe_2023 = model.predict(year_2023)

# Predict y values using the model
predicted_y = model.predict(x)

# Calculate R-squared
r2 = r2_score(y, predicted_y)

# Calculate mean squared error
mse = mean_squared_error(y, predicted_y)

# Create the linear equation string
linear_equation = f'LCOE = {model.intercept_:.4f} + {model.coef_[0]:.4f} * Year'

# Plot the data, linear regression line, and the predicted 2023 value
plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='blue', label='Actual Data')
plt.plot(x, predicted_y, color='red', label='Linear Regression Line')
plt.scatter(year_2023, predicted_lcoe_2023, color='green', label='Predicted 2023')
plt.xlabel('Years')
plt.ylabel('Weighted average LCOE values')
plt.title('Linear Regression of Weighted average LCOE values')
plt.legend()
plt.grid(True)

# Add linear equation, R-squared, and MSE to the figure
plt.text(2010, 0.4, linear_equation, fontsize=12, color='black')
plt.text(2010, 0.35, f'R-squared: {r2:.4f}', fontsize=12, color='black')
plt.text(2010, 0.3, f'MSE: {mse:.4f}', fontsize=12, color='black')

# Show the plot

st.pyplot(plt.gcf())





