#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[5]:


import streamlit as st
import numpy as np
import pickle

# Load model (save it first using pickle)
model = pickle.load(open("diabetes_model.pkl","rb"))

st.title("Diabetes Prediction App")
st.subheader("Enter your health data to check your diabetes risk")

# --- Sidebar Navigation ---
page = st.sidebar.radio("Navigate", ["Prediction", "About", "Health Tips"])

# --- Input sliders ---
if page == "Prediction":
    st.header("Health Inputs")

    pregnancies = st.slider("Pregnancies", 0, 20, 1)
    glucose = st.slider("Glucose Level", 50, 200, 100)
    blood_pressure = st.slider("Blood Pressure", 30, 130, 80)
    skin_thickness = st.slider("Skin Thickness", 0, 100, 20)
    insulin = st.slider("Insulin", 0, 900, 100)
    bmi = st.slider("BMI", 10.0, 70.0, 25.0)
    dpf = st.slider("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
    age = st.slider("Age", 10, 100, 30)

    # --- Prediction ---
    if st.button("Predict"):
        user_input = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
        result = model.predict(user_input)[0]
        confidence = model.predict_proba(user_input)[0][1]

        if result == 1:
            st.error(f"You may be at risk of diabetes. Confidence: {confidence:.2%}")
        else:
            st.success(f"You are likely not diabetic. Confidence: {confidence:.2%}")

        # --- Personalized Tips ---
        st.markdown("### Health Advice:")
        if glucose > 125:
            st.warning("Your glucose level is high. Consider reducing sugar intake and consulting a doctor.")
        if bmi > 30:
            st.warning("Your BMI is high. Try to maintain a healthy weight with exercise and balanced meals.")
        if blood_pressure > 120:
            st.warning("Elevated blood pressure detected. Limit salt intake and monitor regularly.")
        if age > 40 :
            st.warning("Ensure regular checkups. Age is a predisposing factor to diabetes.")

# --- Data Visualization ---
    st.markdown("### Sample Data Visualization (Glucose & BMI Boxplot)")
    sample_data = np.array([[glucose, bmi]])
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    sns.boxplot(y=sample_data[:, 0], ax=ax[0], color='skyblue')
    ax[0].set_title("Glucose Level")
    sns.boxplot(y=sample_data[:, 1], ax=ax[1], color='lightgreen')
    ax[1].set_title("BMI")
    st.pyplot(fig)

# --- About Page ---
elif page == "About":
    st.header("About This App")
    st.write("""
        This app uses a machine learning model to predict the likelihood of diabetes based on health inputs. 
        It's for educational purposes and should not replace medical advice.
    """)

# --- Health Tips Page ---
elif page == "Health Tips":
    st.header("General Health Tips")
    st.markdown("""
    - **Eat balanced meals**: High fiber, low sugar, lean proteins.
    - **Stay active**: At least 30 minutes a day.
    - **Monitor health metrics** regularly (BP, glucose, BMI).
    - **Stay hydrated** and get enough sleep.
    """)



# In[ ]:





# In[2]:


#convert and save my file as .py

get_ipython().system('jupyter nbconvert --to script diabetes_app.ipynb')


# In[ ]:




