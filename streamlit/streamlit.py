import streamlit as st
from requests_toolbelt.multipart.encoder import MultipartEncoder
import requests
from PIL import Image
import io
import zipfile
import glob
import os
import json

st.title('MoroccoAI Data Challenge : Automatic Number Plate Recognition (ANPR) in Morocco Licensed Vehicles.')

# fastapi endpoint
url = 'http://168.61.19.23:8000'
endpoint = '/platedetector'

st.write('''This application is a demo result of our work in the comepetiton organized by MoroccoAI in the context of the first MoroccoAI Data Challenge. 
it takes an image that contains one or multiple cars and return the plates and the recognized characters on each plate''') # description and instructions

image = st.file_uploader('insert image')  # image upload widget


def process(image, server_url: str):

    m = MultipartEncoder(
        fields={'file': ('filename', image, 'image/jpeg')}
        )

    r = requests.post(server_url,
                      data=m,
                      headers={'Content-Type': m.content_type},
                      timeout=8000)

    return r


if st.button('Get Plate detected'):
    res = process(image, url+endpoint)
    #rejs = json.loads(res.text)
    print(io.BytesIO(res.content))
    #print(res.request.body)
    st.image(io.BytesIO(res.content))
