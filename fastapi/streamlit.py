import streamlit as st
import main
st.title('MoroccoAI Data Challenge : Automatic Number Plate Recognition (ANPR) in Morocco Licensed Vehicles.')

st.write('''This application is a demo result of our work in the comepetiton organized by MoroccoAI in the context of the first MoroccoAI Data Challenge.
it takes an image that contains one or multiple cars and return the plates and the recognized characters on each plate''')

image = st.file_uploader('insert image')  # image upload widget


if st.button('Get Plate detected'):
    main.get_plate_detection(image)
if st.button('Get Plate OCR'):
    main.get_plate_ocr(image)
