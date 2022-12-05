#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


# Loading the data

df = pd.read_csv('Bengaluru_House_Data.csv')
df.head()


# In[3]:


df.shape


# In[4]:


# DATA CLEANING
# Grouping the values based on the 'area_type' feature and counting how many data points are there in each type

df.groupby('area_type')['area_type'].agg('count')


# In[5]:


# Dropping unnecessary features

df1 = df.drop(['area_type', 'availability', 'society', 'balcony'], axis = 'columns')
df1.head()


# In[6]:


# Checking for null values

df1.isnull().sum()


# In[7]:


# Dropping missing values
# Since missing values are very small in number, we drop those values
# dropna() - removes the rows that contains null values
# dropna(0) - removes rows, dropna(1) - removes columns, default - 0(rows)

df2 = df1.dropna()
df2.isnull().sum()


# In[8]:


df2.shape


# In[9]:


# Checking for unique values of 'size' column

df2['size'].unique()


# In[10]:


# FEATURE ENGINEERING - creating new columns from the existing ones
# Extracting new column 'bhk' from the 'size' feature - stores number of bedrooms
# apply function - takes a function as an input and applies this function to an entire DataFrame
# lambda function[lambda arguments: expression] - anonymous function, n number of arguments, only one expression

df2['size'] = df2['size'].apply(lambda x: int(x.split(' ')[0]))
df2.rename(columns = {'size': 'bhk'}, inplace = True) # Renaming the 'size' column as 'bhk'
df2.head()


# In[11]:


df2['bhk'].unique()


# In[12]:


# Checking for unique values of 'total_sqft' column

df2['total_sqft'].unique()


# In[13]:


# This function is used to check whether a value can be converted into float or not in the 'total_sqft' column
# try - check for errors, executes only when there is no error in the program
# except - executes when there is some error in the preceding block

def is_float(x):
    try:
        float(x)
    except:
        return False
    return True


# In[14]:


# Printing the values which are not float in the 'total_sqft' column

df2[~df2['total_sqft'].apply(is_float)].head(10)

# The values which cannot be converted into float are: range values and some string values
# There are many range values, so we convert them to a float value and ignore the other type of values such as strings.


# In[15]:


# This function is used to convert the range values into float by finding the average of the range
# The values which cannot be converted to float are replaced as null values

def cnvt_rng_to_num(x):
    tokens = x.split('-')
    if len(tokens) == 2:
        return (float(tokens[0]) + float(tokens[1])) / 2
    try:
        return float(x)
    except:
        return None


# In[16]:


# Copying the data frame into a new one and applying the function 'cnvt_rng_to_num'

df3 = df2.copy()
df3['total_sqft'] = df3['total_sqft'].apply(cnvt_rng_to_num)
df3.head()


# In[17]:


df3.isnull().sum()

# There are null values after applying the 'cnvt_rng_to_num' function


# In[18]:


# Removing those null values

df4 = df3.dropna()
df4.isnull().sum()


# In[19]:


df4.shape


# In[20]:


# FEATURE ENGINEERING
# Creating a new column 'price_per_sqft' from the columns 'price' and 'total_sqft'

df5 = df4.copy()
df5['price_per_sqft'] = df5['price'] * 100000 / df5['total_sqft']
df5.head()


# In[21]:


# Checking for unique values of 'location' column

df5['location'].nunique()

# 'location' is a categorical feature, one hot encoding leads to high dimensions - Dimensionality curse
# Hence reducing the dimensions


# In[22]:


# DIMENSIONALITY REDUCTION
# Grouping the values based on the 'location' feature
# Count how many data points are there in each location and print them in descending order
# strip - returns a copy of the string with both leading and trailing characters removed (based on the string argument passed)

df5['location'] = df5['location'].apply(lambda x: x.strip())
location_stats = df5.groupby('location')['location'].agg('count').sort_values(ascending = False)
location_stats


# In[23]:


len(location_stats[location_stats <= 10])

# 1047 locations seems to have less than or equal to ten data points


# In[24]:


# Storing those location values to a variable

location_stats_less_than_10 = location_stats[location_stats <= 10]
location_stats_less_than_10


# In[25]:


# Checking for unique values of 'location' column

df5['location'].nunique()


# In[26]:


# The locations which have less than or equal to 10 data points, we label it as 'Other' & print the remaining locations as it is
# Check the unique values now

df5['location'] = df5['location'].apply(lambda x : 'Other' if x in location_stats_less_than_10 else x)
df5['location'].nunique()


# In[27]:


df5.head(10)


# In[28]:


# OUTLIER DETECTION
# Threshold - 300, the sqft per bedroom cannot be less than 300 [Domain Knowledge]
# Print the houses with bedrooms less than 300 sqft

df5[df5.total_sqft / df5.bhk < 300]


# In[29]:


df5.shape


# In[30]:


# OUTLIER REMOVAL
# Removing the data points where the bedroom size is less than 300 sqft

df6 = df5[~(df5.total_sqft / df5.bhk < 300)]
df6.shape


# In[31]:


# Checking into 'price_per_sqft' column and printing the statsitical values

df6['price_per_sqft'].describe()


# In[32]:


# OUTLIER REMOVAL
# Removing 'price_per_sqft' outliers per location

def remove_pps_outliers(df):
    
    df_out = pd.DataFrame() # Creates empty dataframe
    
    '''key - name of the locations, subdf - a sub dataframe of each location
       241 sub dataframes (groups) --> 241 unique values in 'location' feature'''
    
    for key, subdf in df.groupby('location'):
        
        m = np.mean(subdf.price_per_sqft) # mean and sd of 'price_per_sqft' column for each groups
        sd = np.std(subdf.price_per_sqft) # it gives 241 mean and 241 sd for 241 groups
        
        # Assume the data follows Normal Distribution
        # Keep only 68% of the data which falls within 1 standard deviation from the mean (i.e.) between μ – σ and μ + σ
        
        reduced_df = subdf[(subdf.price_per_sqft >= (m - sd)) & (subdf.price_per_sqft <= (m + sd))]
        # Here '&' is used, we can't use 'and', because in pandas series, 'and', 'or' and 'not' are considered ambiguous
        # This is because numpy arrays and pandas series use the bitwise operators rather than logical as you are comparing 
        # every element in the array/series with another
        
        df_out = pd.concat([df_out, reduced_df], ignore_index = True)
        # concat - joins two data frames
        # ignore_index - clear the existing index and reset it in the result
        
    return df_out


# In[33]:


df7 = remove_pps_outliers(df6)
df7.shape


# In[34]:


# Customizing the plots by adjusting the rc parameters
# rc - runtime configuration, it contains default styles for every plot you create
# (width, height) in inches

plt.rcParams['figure.figsize'] = (20, 10)


# In[35]:


# Plot the 2 bhk and 3 bhk houses (total_sqft, price) per location in a scatter plot
# We're looking for whether a 2 bhk house price is more than a 3 bhk house price for the same sqft and location

def plot_scatter_chart(df, location):
    bhk2 = df[(df.location == location) & (df.bhk == 2)]
    bhk3 = df[(df.location == location) & (df.bhk == 3)]
    plt.scatter(bhk2.total_sqft, bhk2.price, color = 'blue', label = '2bhk', s = 50)
    plt.scatter(bhk3.total_sqft, bhk3.price, color = 'green', label = '3bhk', s = 50)
    plt.xlabel('Total Square Feet Area')
    plt.ylabel('Price')
    plt.title(location)
    plt.legend()


# In[36]:


plot_scatter_chart(df7, 'Hebbal')


# In[37]:


# OUTLIER REMOVAL
# Removing 2 bhk houses whose price_per_sqft is less than the mean price_per_sqft of 1 bhk houses

def remove_bhk_outliers(df):
    
    exclude_indices = np.array([]) # Creates empty array
    
    for location, location_df in df.groupby('location'):
        bhk_stats = {} # Creates empty dictionary for each location (i.e.) 241 empty dictionaries
        
        for bhk, bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk] = {
                  'mean': np.mean(bhk_df.price_per_sqft),
                  'std': np.std(bhk_df.price_per_sqft),
                  'count': bhk_df.shape[0] # it gives number of rows (i.e. number of houses) in a dataframe
              } 
        # Appends this dictionary to the 'bhk_stats' dictionary for each location (dictionary of stats per bhk)
            
        for bhk, bhk_df in location_df.groupby('bhk'):
            stats = bhk_stats.get(bhk - 1)
        # Get the stats of previous bhk house (i.e. for 2 bhk it prints the stats of 1 bhk)
        
            # It checks if there is dictionary present (we didn't have for 1 bedroom group) because None value will throw error
            # It also checks if it has more than 5 values or not
            if stats and stats['count'] > 5:
                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft < (stats['mean'])].index.values)
                # store the index of the current bedroom group's element if it is lower than the previous bedroom's mean value
                
    return df.drop(exclude_indices, axis = 'index') 


# In[38]:


df8 = remove_bhk_outliers(df7)
df8.shape


# In[39]:


plot_scatter_chart(df8, 'Hebbal')


# In[40]:


# A Histogram showing number of houses per sqft value

plt.hist(df8.price_per_sqft, rwidth = 0.8)
plt.xlabel('Price Per Square Feet')
plt.ylabel('Count')


# In[41]:


# Checking for unique values of 'bath' column

df8.bath.unique()


# In[42]:


# A Histogram showing number of houses per bathroom value

plt.hist(df8.bath, rwidth = 0.8)
plt.xlabel('Bathrooms')
plt.ylabel('Count')


# In[43]:


# The houses with bathrooms more than the number of bedrooms + 2

df8[df8.bath > df8.bhk+2]


# In[44]:


# OUTLIER REMOVAL
# Removing 'bath' outliers

df9 = df8[df8.bath < df8.bhk+2]
df9.shape


# In[45]:


# Dropping the unnecessary features before model building
# Dropping 'price_per_sqft' column as it is used only for outlier detection, we don't need the column for model building

df10 = df9.drop('price_per_sqft', axis = 'columns')
df10.head()


# In[46]:


# ONE-HOT ENCODING (dummies) - convert categorical data into numerical data

dummies = pd.get_dummies(df10.location)
dummies.head()


# In[47]:


# Dropping one column inorder to avoid 'Dummy Variable Trap'
# This column is represented as all zeros

'''Dummy Variable Trap - Attributes are highly correlated (Multicollinear)
   One dummy variable can be predicted with the help of other dummy variables.
   Hence, one dummy variable is highly correlated with other dummy variables. 
   Using all dummy variables for regression models leads to a dummy variable trap. 
   So, the regression models should be designed to exclude one dummy variable.'''

df11 = pd.concat([df10, dummies.drop('Other', axis = 'columns')], axis = 'columns')
df11.head()


# In[48]:


# Dropping the 'location' column as it was converted into dummies columns

df12 = df11.drop('location', axis = 'columns')
df12.head()


# In[49]:


df12.shape


# In[50]:


# MODEL BUILDING

x = df12.drop('price', axis = 'columns') # independent variables
y = df12.price # dependent variables


# In[51]:


# TRAIN TEST SPLIT
# Train dataset - To train the model
# Test dataset - To evaluate the model perfomance

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 10)


# In[52]:


from sklearn.linear_model import LinearRegression

linear = LinearRegression()
linear.fit(x_train, y_train) # model training
linear.score(x_test, y_test) # model evaluation


# In[53]:


# Testing the model for a few samples

def predict_price(location, bhk, sqft, bath):
    
    loc_index = np.where(x.columns == location)[0][0] # It gives the index of the 'location' column specifically
    
    z = np.zeros(len(x.columns), dtype = int) # Return a new array of given shape and size, filled with zeros.
    z[0] = bhk
    z[1] = sqft
    z[2] = bath
    if loc_index >= 3:
        z[loc_index] = 1
        
    return linear.predict([z])[0]


# In[54]:


predict_price('1st Block Jayanagar', 2, 2000, 2)

