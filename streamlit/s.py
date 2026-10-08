import streamlit as st
import requests

API_BASE_URL = "http://127.0.0.1:8000"
st.write("Hi, I am here to assist you regarding our Policies and work culture")
# st.session_state.api_client = API_BASE_URL
q = st.text_input("Query")
if st.button("Ask question"):
    if q:
        response = requests.post(f"{API_BASE_URL}/get_response", json={"query": q})
        answer = response.json()
        st.write(answer)
