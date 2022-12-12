import streamlit as st
from PIL import Image

st.markdown("<h1 style='text-align: center; color: darkgreen;'>House Price Prediction</h1>", unsafe_allow_html=True)

st.write('Hello Folks! Welcome to my _House Price Prediction_ website.')
st.write('Here you will find prices for the houses in **Bengaluru, Karnataka**. Use my prediction website '
         'to find the price of your dream house in Bengaluru.')

st.markdown(
    """
    <style>
        button[title^=Exit]+div [data-testid=stImage]{
            text-align: center;
            display: block;
            margin-left: auto;
            margin-right: auto;
            width: 100%;
        }
    </style>
    """, unsafe_allow_html=True
)

image = Image.open('house.jpg')
st.image(image, width = 680)

st.write('_Happy Predicting!_')

st.sidebar.markdown('## Home')
