# -*- coding: utf-8 -*-
"""
Created on Mon May 27 21:48:46 2024

@author: vedan
"""

import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Load model and data
filename = 'svc_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))
df = pd.read_csv("Clustered_Customer_Data.csv")

# Set Streamlit options
st.set_option('deprecation.showPyplotGlobalUse', False)

# Style and title
st.markdown('<style>body{background-color: Blue;}</style>', unsafe_allow_html=True)
st.title("Prediction")

# Function to calculate additional columns
def add_columns(df):
    df['Kids'] = df['Kidhome'] + df['Teenhome']
    df['Expenses'] = (df['MntWines'] + df['MntFruits'] + df['MntMeatProducts'] +
                      df['MntFishProducts'] + df['MntSweetProducts'] + df['MntGoldProds'])
    df['TotalAcceptedCmp'] = (df['AcceptedCmp1'] + df['AcceptedCmp2'] + df['AcceptedCmp3'] +
                              df['AcceptedCmp4'] + df['AcceptedCmp5'])
    df['NumTotalPurchases'] = (df['NumDealsPurchases'] + df['NumWebPurchases'] + df['NumCatalogPurchases'] +
                               df['NumStorePurchases'])
    return df

df = add_columns(df)

# Cluster descriptions
cluster_descriptions = {
    0: ("Cluster 0:\n"
        "Highest number of customers\n"
        "Majority of them have at least one or two children\n"
        "Majority did their graduation, post-graduation, and very few did their basic education\n"
        "Low income\n"
        "Expenses on different categories are done more by middle-aged people\n"
        "Higher expenses are made by households having no children\n"
        "Postgraduate people make higher expenses"),
    1: ("Cluster 1:\n"
        "Second highest number of customers\n"
        "Majority of them have at least one child\n"
        "Majority did their graduation and post-graduation\n"
        "Income below 60000\n"
        "Expenses on different categories are done more by middle-aged people\n"
        "Higher expenses are made by households having no children\n"
        "Postgraduate people make higher expenses"),
    2: ("Cluster 2:\n"
        "Most households do not have any kids\n"
        "Well educated\n"
        "Mostly middle-aged people followed by equal numbers of senior citizens and adults\n"
        "Highest income\n"
        "More than 20 purchases\n"
        "Accepted a lot of campaigns"),
    3: ("Cluster 3:\n"
        "Most households do not have any kids\n"
        "Well educated\n"
        "Mostly middle-aged and senior citizens\n"
        "Highest income\n"
        "More than 25 purchases\n"
        "Accepted a lot of campaigns")
}

# Mapping for education levels
education_options = {
    0: '0: Basics',
    1: '1: 2nd Cycle',
    2: '2: Graduation',
    3: '3: Master',
    4: '4: PhD'
}

# Reverse mapping for filtering
education_reverse_options = {v: k for k, v in education_options.items()}

# Mapping for marital status
marital_status_options = {
    0: '0: Absurd',
    1: '1: Alone',
    2: '2: Divorced',
    3: '3: Married',
    4: '4: Single',
    5: '5: Together',
    6: '6: Widow',
    7: '7: YOLO'
}

# Reverse mapping for filtering
marital_status_reverse_options = {v: k for k, v in marital_status_options.items()}

# Streamlit app
with st.form("my_form"):
    st.title("Customer Data Analysis")

    # Display the dataset
    if st.checkbox('Show raw data'):
        st.subheader('Raw Data')
        st.write(df)

    # Sidebar for user inputs
    st.sidebar.header('Filter Options')

    # Filter by marital status
    marital_status_labels = [marital_status_options[key] for key in df['Marital_Status'].unique()]
    marital_status_selected = st.sidebar.multiselect(
        'Select Marital Status',
        options=marital_status_labels,
        default=marital_status_labels
    )
    marital_status = [marital_status_reverse_options[label] for label in marital_status_selected]

    # Filter by education
    education_labels = [education_options[key] for key in df['Education'].unique()]
    education_selected = st.sidebar.multiselect(
        'Select Education Level',
        options=education_labels,
        default=education_labels
    )
    education = [education_reverse_options[label] for label in education_selected]

    # Filter by income range
    min_income, max_income = st.sidebar.slider(
        'Select Income Range',
        min_value=int(df['Income'].min()),
        max_value=int(df['Income'].max()),
        value=(int(df['Income'].min()), int(df['Income'].max()))
    )

    # Filter by number of kids
    min_kids, max_kids = st.sidebar.slider(
        'Select Number of Kids',
        min_value=int(df['Kids'].min()),
        max_value=int(df['Kids'].max()),
        value=(int(df['Kids'].min()), int(df['Kids'].max()))
    )

    # Filter by expenses
    min_expenses, max_expenses = st.sidebar.slider(
        'Select Expenses Range',
        min_value=int(df['Expenses'].min()),
        max_value=int(df['Expenses'].max()),
        value=(int(df['Expenses'].min()), int(df['Expenses'].max()))
    )

    # Filter by age range
    min_age, max_age = st.sidebar.slider(
        'Select Age Range',
        min_value=int(df['Age'].min()),
        max_value=int(df['Age'].max()),
        value=(int(df['Age'].min()), int(df['Age'].max()))
    )

    # Apply filters to data
    filtered_data = df[
        (df['Marital_Status'].isin(marital_status)) &
        (df['Education'].isin(education)) &
        (df['Income'] >= min_income) &
        (df['Income'] <= max_income) &
        (df['Kids'] >= min_kids) &
        (df['Kids'] <= max_kids) &
        (df['Expenses'] >= min_expenses) &
        (df['Expenses'] <= max_expenses) &
        (df['Age'] >= min_age) &
        (df['Age'] <= max_age)
    ]
    
    submitted = st.form_submit_button("Submit")

    if submitted:
        # Ensure 'filtered_data' has the necessary columns for model prediction
        columns_needed_for_prediction = [
            'Income', 'Kidhome', 'Teenhome', 'MntWines', 'MntFruits', 
            'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 
            'MntGoldProds', 'AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 
            'AcceptedCmp4', 'AcceptedCmp5', 'NumDealsPurchases', 'NumWebPurchases', 
            'NumCatalogPurchases', 'NumStorePurchases', 'Age', 'Marital_Status', 
            'Education', 'Response', 'TotalAcceptedCmp', 'NumWebVisitsMonth', 
            'Recency', 'Kids', 'Complain', 'NumTotalPurchases', 'Expenses'
        ]
        
        if all(column in filtered_data.columns for column in columns_needed_for_prediction):
            data = filtered_data[columns_needed_for_prediction]
            clust = loaded_model.predict(data)[0]
            st.write(f'The data belongs to {cluster_descriptions[clust]}')  # Display cluster details

            cluster_df1 = df[df['cluster'] == clust]
            plt.rcParams["figure.figsize"] = (10, 6)
            
            for c in cluster_df1.drop(['cluster'], axis=1):
                fig, ax = plt.subplots()
                sns.histplot(cluster_df1[c], kde=True, ax=ax)
                ax.set_title(f'Distribution of {c} in Cluster {clust}')
                ax.set_xlabel(c)
                ax.set_ylabel('Frequency')
                st.pyplot(fig)
        else:
            st.error("Filtered data does not contain all the necessary columns for prediction.")
