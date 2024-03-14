import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


def shorten_categories(categories, cutoff):
    categorical_map = {}
    for i in range(len(categories)):
        if categories.values[i] >= cutoff:
            categorical_map[categories.index[i]] = categories.index[i]
        else:
            categorical_map[categories.index[i]] = 'Other'
    return categorical_map


def clean_experience(x):
    if x ==  'More than 50 years':
        return 50
    if x == 'Less than 1 year':
        return 0.5
    return float(x)

# after refresh streamlit will load data again and again, to avoid I'll use cache
@st.cache_data
def load_data():
    df = pd.read_csv("survey_results_public_2023.csv")
    df = df[["Country", "EdLevel", "YearsCodePro", "Employment", "ConvertedCompYearly", "DevType", "RemoteWork"]]
    df = df.rename({"ConvertedCompYearly": "Salary"}, axis=1)
    df = df[df["Salary"].notnull()]
    df = df.dropna()
    df.isnull().sum()
    df = df[df["Employment"] == "Employed, full-time"]
    df = df.drop("Employment", axis=1)
    
    country_map = shorten_categories(df.Country.value_counts(), 300)
    df['Country'] = df['Country'].map(country_map)
    df = df[df["Salary"] <= 250000]
    df = df[df["Salary"] >= 10000]
    df = df[df['Country'] != 'Other']
    
    df['YearsCodePro'] = df['YearsCodePro'].apply(clean_experience)
    df['EdLevel'] = df['EdLevel']
    return df

df = load_data()

def show_explore_page():
    st.title("Explore Sortware Engineer Salary")

    st.write(
        """
        ### Stack OverFlow Developer Survey 2021
    """
    )
    # Building charts eg, pie, plot, 
    
    data = df["Country"].value_counts()

    fig1,ax1 = plt.subplots(figsize =(16, 16))
    ax1.pie(data, labels=data.index, autopct="%1.1f%%", shadow=True, startangle=90)
    ax1.axis("equal")  # Equal aspect ratio ensures that pie is drawn a circle

    # to display
    st.write("""### Number of data from different countries""")
    
    st.pyplot(fig1)

    st.write("""
    ### Mean Salary Based on Country 
    """
    )

    data = df.groupby(["Country"])["Salary"].mean().sort_values(ascending=True)
    st.bar_chart(data)

    
    st.write("""
    ### Mean Salary Based on Experience
    """
    )

    data = df.groupby(["YearsCodePro"])["Salary"].mean().sort_values(ascending=True)
    st.line_chart(data)
