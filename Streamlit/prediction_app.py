import streamlit as st
import joblib
import pandas as pd

st.set_page_config(page_title="Anime Global Hit Predictor", layout="centered")

st.markdown(f"""
    <style>
    /* Main App Background */
    .stApp {{
        background-color: #e63946;
    }}

    /* Global Text Color */
    h1, h2, h3, p, span, label, .stMarkdown {{
        color: #ffffff !important;
        font-family: 'Rubik', sans-serif;
    }}

    /* Sidebar Background */
    [data-testid="stSidebar"] {{
        background-color: #232a32;
    }}

    /* Input Box Backgrounds */
    .stSelectbox div[data-baseweb="select"], .stNumberInput input {{
        background-color: #ffffff !important;
        color: #000000 !important;
    }}

    /* THE BUTTON: #121b33 */
    .stButton>button {{
        background-color: #121b33 !important;
        color: #ffffff !important;
        border: 2px solid #ffffff;
        border-radius: 10px;
        font-weight: bold;
        padding: 0.75rem 2rem;
        width: 100%;
    }}

    /* RESULT TEXT: #3d705c */
    .success-text {{
        color: #3d705c !important;
        background-color: #ffffff; /* White background for visibility */
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        font-size: 24px;
    }}
    </style>
    """, unsafe_allow_html=True)

#loading the package 
modelst = joblib.load('xgboost_model.pkl')
category_mappings = joblib.load('category_mappings.pkl')

st.title("Anime Global Hit Predictor")

#Extracting the dictionary 
origin_map = category_mappings.get('Originated_From', {})
genre_map = category_mappings.get('Main_Genre', {})
studio_map = category_mappings.get('Studios', {})

#UI 
st.sidebar.header("Input Production Features")

#Select box will be the number
origin = st.sidebar.selectbox("Originated_From", options=list(origin_map.keys()),format_func=lambda x : origin_map[x])

popularity = st.sidebar.number_input("Popularity_Rank", min_value=1, value=500)
score = st.sidebar.slider("Score", 0.0, 10.0, 7.5)

genre = st.sidebar.selectbox("Main_Genre", options=list(genre_map.keys()), format_func=lambda x:genre_map[x])
studio = st.sidebar.selectbox("Studios", options=list(studio_map.keys()), format_func=lambda x: studio_map[x])

if st.button("Predict Global Success"):
    #in the order of how ML read the feature column & the exact 5 
    input_data = pd.DataFrame([[
        origin, 
        popularity,                                              
        score,                                                  
        genre,
        studio
    ]], columns=['Originated_From', 'Popularity_Rank', 'Score', 'Main_Genre', 'Studios'])

#prediction execution 
    prediction = modelst.predict(input_data)
    
    st.divider()

    if prediction[0] == 1:
        st.balloons()
        st.markdown('<div class="success-text">🚀Result: GLOBAL HIT</div>', unsafe_allow_html=True)
    else:
        st.error("### Result: **Niche/Local** 📺")