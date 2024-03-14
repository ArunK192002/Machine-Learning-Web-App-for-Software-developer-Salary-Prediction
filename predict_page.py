# Loading the libraries
import  streamlit as st
import pickle
import numpy as np

def load_model():
    with open('model.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

regressor = data["model"]
le_country = data["le_country"]
le_education = data["le_education"]
le_devtype = data["le_devtype"]
le_work = data["le_work"]

#Streamlit
def show_predict_page():
    st.title("Software Developer Salary Prediction")

    st.write("""### We need some information to predict the salary""")

    countries = (
        'United States of America',
        'United Kingdom of Great Britain and Northern Ireland',
        'Finland',
        'Australia', 
        'Netherlands', 
        'Germany', 
        'Sweden', 
        'France', 
        'Other', 
        'Spain', 
        'Brazil', 
        'Portugal', 
        'Italy', 
        'Canada', 
        'Switzerland', 
        'India', 
        'Russian Federation', 
        'Austria', 
        'Norway', 
        'Turkey', 
        'Belgium', 
        'Denmark', 
        'Israel', 
        'Ukraine', 
        'Poland', 
        'New Zealand'
    )
    
    education = (
        'Bachelor’s degree (B.A., B.S., B.Eng., etc.)', 
        'Some college/university study without earning a degree',
        'Master’s degree (M.A., M.S., M.Eng., MBA, etc.)', 
        'Professional degree (JD, MD, Ph.D, Ed.D, etc.)', 
        'Associate degree (A.A., A.S., etc.)',
        'Secondary school (e.g. American high school, German Realschule or Gymnasium, etc.)',
        'Primary/elementary school', 
        'Something else'
    )

    devtype= (
        'Developer, back-end', 
        'Developer, front-end', 
        'Developer, full-stack', 
        'System administrator', 
        'Developer, QA or test', 
        'Data scientist or machine learning specialist', 
        'Data or business analyst', 
        'Security professional', 
        'Research & Development role', 
        'Developer, desktop or enterprise applications', 
        'Engineer, data', 
        'Product manager', 
        'Other', 
        'Developer, embedded applications or devices', 
        'Developer Experience', 'Other (please specify):', 
        'Cloud infrastructure engineer', 
        'Developer, mobile', 
        'DevOps specialist', 
        'Engineering manager', 
        'Senior Executive (C-Suite, VP, etc.)', 
        'Engineer, site reliability', 
        'Project manager', 
        'Academic researcher', 
        'Developer, game or graphics', 
        'Hardware Engineer', 
        'Scientist'
    )

    work = (
        'Hybrid (some remote, some in-person)', 
        'Remote', 
        'In-person'
    )

    
    # creating selectbox for country and education
    country = st.selectbox("Country",sorted(countries))
    education = st.selectbox("Education Level",education)
    devtype = st.selectbox("DevType", sorted(devtype))
    work = st.selectbox("Work Type", work)

    experience = st.slider("Years of experience", 0,50,2)

    # adding a button to start the prediction
    ok = st.button("Calculate Salary")
    if ok:
        # country, edlevel, yearscode
        X = np.array([[country, education, experience, devtype, work]])
        X[:, 0] = le_country.transform(X[:,0])
        X[:, 1] = le_education.transform(X[:,1])
        X[:, 3] = le_devtype.transform(X[:,3])
        X[:, 4] = le_work.transform(X[:,4])
        X = X.astype(float)
        
        # Salary Prediction using pickled model
        salary = regressor.predict(X)
        st.subheader(f"The estimated salary is $ {salary[0]:.2f}")