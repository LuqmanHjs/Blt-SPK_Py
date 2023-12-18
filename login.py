import streamlit as st
import mysql.connector
from subprocess import Popen
headerSection = st.container()
mainSection = st.container()
loginSection = st.empty()
logOutSection = st.container()

# Koneksi ke database MySQL
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="streamlit"
)

# # Fungsi untuk validasi login
# def validate_login(username, password):
#     cursor = db.cursor()
#     query = "SELECT * FROM users WHERE username = %s AND password = %s"
#     values = (username, password)
#     cursor.execute(query, values)
#     result = cursor.fetchone()
#     if result:
#         return True
#     else:
#         return False
    
def validate_login(username, password):
    # Membuat koneksi ke database MySQL
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="streamlit" )
        
    # Mengeksekusi query untuk mencari username dan password yang sesuai
    cursor = conn.cursor()
    query = "SELECT * FROM users WHERE username = %s AND password = %s"
    cursor.execute(query, (username, password))
    # Memeriksa apakah ada hasil yang cocok
    result = cursor.fetchone()
    cursor.close()
    return result is not None

def LoggedOut_Clicked():
    st.session_state['loggedIn'] = False
    
# def LoggedIn_Clicked(username, password):
#     if validate_login(username, password):
#         st.session_state['loggedIn'] = True
#     else:
#         st.session_state['loggedIn'] = False
#         st.error("Invalid user name or password")
       
def show_login_page():
        with loginSection.form("login"):
            if st.session_state['loggedIn'] == False:
                st.markdown("<h1 style='text-align: center; color:black;'>LOGIN</h1>", unsafe_allow_html=True)
                username = st.text_input ("Masukkan Username")
                password = st.text_input ("Masukkan Password", type="password")
                if st.form_submit_button("Login"):
                    if validate_login(username, password):
                            st.session_state['loggedIn'] = True
                    else:
                            st.session_state['loggedIn'] = False
                            st.error("Invalid user name or password")
               

# def show_login_page():
#              with loginSection.form("Login"):
#                 if st.session_state['loggedIn'] == False:
#                     # st.title("Log-in Application")
#                     st.markdown("#### Enter your credentials")
#                     username  = st.text_input(label="", value="", placeholder="Username")
#                     password = st.text_input(label="", value="", placeholder="Password", type="password")
#                     st.form_submit_button("Login", on_click=LoggedIn_Clicked, args=(username, password))
#                         # st.text_input (label="", value="", placeholder="Enter your user name")S