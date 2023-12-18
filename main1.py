from datetime import datetime
import datetime
from bs4 import ResultSet
import sys
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import subprocess
import numpy as np
import pickle
import xlsxwriter
import io
import joblib
import altair as alt
from pandas import ExcelWriter
from pandas import ExcelFile
from jcopml.plot import plot_confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
import plost
from sklearn.pipeline import Pipeline
import mysql.connector
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.compose import ColumnTransformer
from jcopml.pipeline import num_pipe, cat_pipe
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer   
from streamlit_option_menu import option_menu
from login import validate_login, show_login_page,LoggedOut_Clicked

buffer = io.BytesIO()

# -------------------------------------------------------------------------------------------------------------
# 00000000000000000000000000000000000000000-FUNGSI-000000000000000000000000000000000000000000000000000000000000
# -------------------------------------------------------------------------------------------------------------
# koneksi 
def create_connection():
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="streamlit" )
    return conn

# @@@@@@-DATA LATIH-@@@@@@

# Fungsi untuk menambah data ke tabel
def tambah_data(dataframe):
    conn = create_connection()
    cursor = conn.cursor()
    # query = "INSERT INTO datalatih (NIK , nama, alamat, jenis_pkj, jml_phsl, jml_art, pengeluaran, status_tmpt, klasifikasi) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)"
    # data = (NIK , nama, alamat, jenis_pkj, jml_phsl, jml_art, pengeluaran, status_tmpt, klasifikasi)
    # cursor.execute(query, data)
    # conn.commit()
    # st.success('Data berhasil ditambahkan!')
    for index, row in dataframe.iterrows():
        insert_query = f"INSERT INTO datalatih VALUES ("
        for value in row:
            insert_query += f"'{str(value)}', "
        insert_query = insert_query[:-2] + ");"
        cursor.execute(insert_query)
        conn.commit()
# def edit_data(new_nama,new_alamat,new_jenis_pkj,new_jml_phsl,new_jml_art,new_pengeluaran,new_status_tmpt,new_klasifikasi,new_tgl,NIK):
#     conn = create_connection()
#     cursor = conn.cursor()
#     query = f"UPDATE datalatih SET nama= %s,alamat=%s,jenis_pkj=%s,jml_phsl=%s,jml_art=%s,pengeluaran=%s,status_tmpt=%s, klasifikasi=%s, tgl=date WHERE NIK= %s"
#     data = (new_nama,new_alamat,new_jenis_pkj,new_jml_phsl,new_jml_art,new_pengeluaran,new_status_tmpt,new_klasifikasi,new_tgl,NIK)
#     cursor.execute(query, data)
#     conn.commit()
#     st.success('Data berhasil diubah!')
# def view_NIK():
#         conn = create_connection()
#         cursor = conn.cursor()
#         cursor.execute('SELECT DISTINCT NIK FROM datalatih')
#         data = cursor.fetchall()
#         return data
# def get_NIK(NIK):
#         conn = create_connection()
#         cursor = conn.cursor()
#         cursor.execute('SELECT * FROM datalatih WHERE NIK="{}"'.format(NIK))
#         data = cursor.fetchall()
#         return data
def all_data():
        conn = create_connection()
        cursor = conn.cursor()
        query = "SELECT * FROM datalatih"
        cursor.execute(query)
        data = cursor.fetchall()
        return data
def split_frame(input_df, rows):
        df = [input_df.loc[i : i + rows - 1, :] for i in range(0, len(input_df), rows)]
        return df

# @@@@@@-DATA UJI-@@@@@@

# Fungsi untuk menambah data ke tabel
def tambah_data_warga(dataframe):
    conn = create_connection()
    cursor = conn.cursor()
    # query = "INSERT INTO datalatih (NIK , nama, alamat, jenis_pkj, jml_phsl, jml_art, pengeluaran, status_tmpt, klasifikasi) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)"
    # data = (NIK , nama, alamat, jenis_pkj, jml_phsl, jml_art, pengeluaran, status_tmpt, klasifikasi)
    # cursor.execute(query, data)
    # conn.commit()
    # st.success('Data berhasil ditambahkan!')
    for index, row in dataframe.iterrows():
        insert_query = f"INSERT INTO datawarga VALUES ("
        for value in row:
            insert_query += f"'{str(value)}', "
        insert_query = insert_query[:-2] + ");"
        cursor.execute(insert_query)
        conn.commit()
# def edit_data_warga(new_nama,new_alamat,new_jenis_pkj,new_jml_phsl,new_jml_art,new_pengeluaran,new_status_tmpt,NIK):
#     conn = create_connection()
#     cursor = conn.cursor()
#     query = f"UPDATE datawarga SET nama= %s,alamat=%s,jenis_pkj=%s,jml_phsl=%s,jml_art=%s,pengeluaran=%s,status_tmpt=%s WHERE NIK= %s"
#     data = (new_nama,new_alamat,new_jenis_pkj,new_jml_phsl,new_jml_art,new_pengeluaran,new_status_tmpt,NIK)
#     cursor.execute(query, data)
#     conn.commit()
#     st.success('Data berhasil diubah!')
def edit_task_data(new_nama,new_alamat,new_jenis_pkj,new_jml_phsl,new_jml_art,new_pengeluaran,new_status_tmpt,NIK):
    conn = create_connection()
    cursor = conn.cursor()
    query = "UPDATE datawarga SET nama= %s,alamat=%s,jenis_pkj=%s,jml_phsl=%s,jml_art=%s,pengeluaran=%s,status_tmpt=%s WHERE NIK= %s"
    data = (new_nama,new_alamat,new_jenis_pkj,new_jml_phsl,new_jml_art,new_pengeluaran,new_status_tmpt, NIK) 
    cursor.execute(query, data)
    conn.commit()
    st.success('Data berhasil diubah!')

# def edit_kl(kl_df):
#     conn = create_connection()
#     cursor = conn.cursor()
#     for index, row in kl_df.iterrows():
#         NIK = row['NIK']
#         nama = row['nama']
#         alamat = row['alamat']
#         jenis_pkj = row['jenis_pkj']
#         jml_phsl = row['jml_phsl']
#         jml_art = row['jml_art']
#         pengeluaran = row['pengeluaran']
#         status_tmpt = row['status_tmpt']
#         klasifikasi = row['klasifikasi']
#         layak = row['layak']
#         tidaklayak = row['tidaklayak']
#         tgl = row['tgl']
#         query = "UPDATE klasifikasi SET nama= %s,alamat=%s,jenis_pkj=%s,jml_phsl=%s,jml_art=%s,pengeluaran=%s,status_tmpt=%s, klasifikasi=%s,layak=%s,tidaklayak=%s, tgl=%s WHERE NIK= %s"
        
#         cursor.execute(query, data)
#         conn.commit()
#     st.success('Data berhasil diubah!')

def view_NIK_warga():
        conn = create_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT DISTINCT NIK FROM datawarga')
        data = cursor.fetchall()
        return data
        
def get_NIK_warga(NIK):
        conn = create_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT NIK , nama, alamat, jenis_pkj, jml_phsl, jml_art, pengeluaran, status_tmpt FROM datawarga WHERE NIK="{}" '.format(NIK))
        data = cursor.fetchall()
        return data

def all_data_warga():
        conn = create_connection()
        cursor = conn.cursor()
        query = "SELECT * FROM datawarga"
        cursor.execute(query)
        data = cursor.fetchall()
        return data

def all_data_warga_beside_KL():
        conn = create_connection()
        cursor = conn.cursor()
        query = "SELECT * FROM datawarga WHERE NIK NOT IN (SELECT NIK FROM klasifikasi)"
        cursor.execute(query)
        data = cursor.fetchall()
        return data
def all_layak_KL():
        conn = create_connection()
        cursor = conn.cursor()
        query = "SELECT * FROM klasifikasi WHERE klasifikasi = 'LAYAK' "
        cursor.execute(query)
        data = cursor.fetchall()
        return data

def all_data_klasifikasi():
        conn = create_connection()
        cursor = conn.cursor()
        query = "SELECT * FROM klasifikasi"
        cursor.execute(query)
        data = cursor.fetchall()
        return data

def delete_data(NIK):
        conn = create_connection()
        cursor = conn.cursor()
        cursor.execute('DELETE FROM datawarga WHERE NIK="{}"'.format(NIK))
        conn.commit()

def delete_ALL():
        conn = create_connection()
        cursor = conn.cursor()
        cursor.execute('TRUNCATE TABLE datawarga')
        conn.commit()

# @@@@@@-save-@@@@@@

def save(dataframe, table_name, connection):
    cursor = connection.cursor()
    # Mengambil nama kolom dari DataFrame
    columns = dataframe.columns.tolist()
    # Membangun query SQL untuk memasukkan data ke dalam tabel
    query = f"INSERT INTO {table_name} (id_kl, NIK , nama, alamat, jenis_pkj, jml_phsl, jml_art, pengeluaran, status_tmpt, klasifikasi, layak, tidaklayak, tgl) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
    # Menggunakan metode iterrows() untuk mengambil setiap baris dalam DataFrame
    for _, row in dataframe.iterrows():
        values = tuple(row)
        cursor.execute(query, values)
    # Commit perubahan ke database
    connection.commit()
    # Menutup kursor dan koneksi
    cursor.close()
    connection.close()
 
 
        
# def generate_id(data):
#     # Menghitung jumlah baris data yang ada
#     num_rows = len(data)
#     # Membuat daftar ID kosong
#     id_list = []
#     # Menghasilkan ID untuk setiap baris data
#     for i in range(num_rows):
#         # Menghasilkan ID dengan format tertentu (misal: ID001, ID002, dst.)
#         new_id = 'KLS' + str(i + 1).zfill(3)
#         id_list.append(new_id)

#     return id_list

# def get_last_code():
#     conn = create_connection()
#     cursor = conn.cursor()
#     cursor.execute("SELECT id_kl FROM klasifikasi ORDER BY id_kl DESC LIMIT 1")
#     result = cursor.fetchone()
#     if result:
#         last_code = result[0]
#         return last_code
#     else:
#         # Menghitung jumlah baris data yang ada
#         num_rows = len(data)
#         # Membuat daftar ID kosong
#         id_list = []
#         # Menghasilkan ID untuk setiap baris data
#         for i in range(num_rows):
#             # Menghasilkan ID dengan format tertentu (misal: ID001, ID002, dst.)
#             new_id = 'KLS' + str(i + 1).zfill(3)
#             id_list.append(new_id)

#         return id_list # Kode awal jika tabel kosong


# def generate_next_kode_id(existing_kode_ids):
#     if not existing_kode_ids:
#         return 'KLS001'  # Jika tabel kosong, mulai dari KLS001

#     last_kode_id = existing_kode_ids[-1]
#     prefix = last_kode_id[:3]
#     number = int(last_kode_id[3:])
#     next_number = number + 1
#     next_kode_id = f'{prefix}{next_number:03d}'
#     return next_kode_id

# # Mengakses tabel database dan mendapatkan kode_id yang telah ada
# def get_existing_kode_ids():
#     conn = create_connection()
#     cursor = conn.cursor()
#     cursor.execute("SELECT id_kl FROM klasifikasi ORDER BY id_kl DESC LIMIT 1")
#     result = cursor.fetchone()
#     return result

# # Menghasilkan dataframe baru dengan kode_id otomatis
# def generate_new_dataframe(num_rows):
#     existing_kode_ids = get_existing_kode_ids()
#     next_kode_id = generate_next_kode_id(existing_kode_ids)

#     data = {
#         'Kode_ID': [f'{next_kode_id}{i+1:02d}' for i in range(num_rows)],
#         # Kolom-kolom lainnya
#     }

#     df = pd.DataFrame(data)
#     return df
# def generate_sequential_id(existing_ids):
#     if len(existing_ids) == 0:
#         return "KLS001"
#     else:
#         last_id = existing_ids[-1]
#         prefix = last_id[:-3]
#         number = int(last_id[-3:])
#         next_number = number + 1
#         return f"{prefix}{next_number:03d}"
    

# cursor.execute("SELECT * FROM klasifikasi")
# existing_data = cursor.fetchone()



# -------------------------------------------------------------------------------------------------------------
# 000000000000000000000000000000000000000000000000-FUNGSI NAIVE BAYES-00000000000000000000000000000000000000000
# -------------------------------------------------------------------------------------------------------------
def accuracy_score(y_true, y_pred):
        return round(float(sum(y_pred == y_true))/float(len(y_true)) * 100 ,2)
# NB 1
def pre_processing(df):
        
        X = df.drop([df.columns[-1]], axis = 1)
        y = df[df.columns[-1]]
        return X, y
# NB 2

class NaiveBayes:
        def __init__(self):
            self.features = {}
            self.prob_label = {}
            self.likelihoods = {}
            self.pred_priors = {}

            self.prior = {}
            self.likelihood = {}
            self.evidence = {}

            self.X_train = np.array
            self.y_train = np.array
            self.train_size = int
            self.alpha = 1

        def fit(self, X, y, alpha=1):
            self.alpha = alpha  # Store the alpha value for Laplace smoothing
            self.features = list(X.columns)
            self.X_train = X
            self.y_train = y
            self.train_size = X.shape[0]

            for feature in self.features:
                self.likelihoods[feature] = {}
                self.pred_priors[feature] = {}

                for feat_val in np.unique(self.X_train[feature]):
                    self.pred_priors[feature].update({feat_val: 0})

                    for hasil in np.unique(self.y_train):
                        self.likelihoods[feature].update({feat_val + '_' + hasil: 0})
                        self.prob_label.update({hasil: 0})

            self._calc_class_prior()
            self._calc_likelihoods()
            self._calc_predictor_prior()
                        
        def _calc_class_prior(self):

            # P(c) - Prior Class Probability 

            for hasil in np.unique(self.y_train):
                hasil_count = sum(self.y_train == hasil)
                self.prob_label[hasil] = hasil_count / self.train_size

      
        def _calc_likelihoods(self):
            for feature in self.features:
                for feat_val in np.unique(self.X_train[feature]):
                    for hasil in np.unique(self.y_train):
                        hasil_count = sum(self.y_train == hasil)

                        # Apply Laplace smoothing
                        num = sum((self.X_train[feature] == feat_val) & (self.y_train == hasil)) + self.alpha
                        denom = hasil_count + self.alpha * len(np.unique(self.X_train[feature]))

                        self.likelihoods[feature][feat_val + '_' + hasil] = num / denom

        def _calc_predictor_prior(self):

            # P(x) - Evidence 
            for feature in self.features:
                feat_vals = self.X_train[feature].value_counts().to_dict()

                for feat_val, count in feat_vals.items():
                    self.pred_priors[feature][feat_val] = count / self.train_size

        def predict(self, X):

            # Calculates Posterior probability P(c|x) 
            results = []
            X = np.array(X)

            for i in X:
                probs_hasil = {}
                for hasil in np.unique(self.y_train):
                    prior = self.prob_label[hasil]
                    likelihood = 1
                    evidence = 1

                    for feat, feat_val in zip(self.features, i):
                        likelihood *= self.likelihoods[feat][feat_val + '_' + hasil]
                        evidence *= self.pred_priors[feat][feat_val]

                    posterior = (likelihood * prior) / evidence
                    probs_hasil[hasil] = posterior  

                result = max(probs_hasil, key=lambda x: probs_hasil[x])
                results.append(result)

            return np.array(results)

        
        def predictt(self, X):

            # Calculates Posterior probability P(c|x)
            
            resultss = []
            X = np.array(X)

            for i in X:
                probs_hasil = {}
                for hasil in np.unique(self.y_train):
                    prior = self.prob_label[hasil]
                    likelihood = 1
                    evidence = 1

                    for feat, feat_val in zip(self.features, i):
                        likelihood *= self.likelihoods[feat][feat_val + '_' + hasil]
                        evidence *= self.pred_priors[feat][feat_val]

                    posterior = (likelihood * prior) / (evidence)
                    probs_hasil[hasil] = posterior
                
                result = probs_hasil
                # result = max(probs_hasil, key = lambda x: probs_hasil[x])
                resultss.append(result)
                # st.write(print("Posterior =  ", probs_hasil))
            return (resultss)
        
        def predicttt(self, X):
                results = []
                likelihoods = []
                evidences = []
                class_priors = []
                X = np.array(X)
                for i in X:
                    probs_hasil = {}
                    likelihood_row = {}
                    evidence_row = {}
                    class_prior_row = {}
                    for hasil in np.unique(self.y_train):
                        prior = self.prob_label[hasil]
                        likelihood = 1
                        evidence = 1
                        for feat, feat_val in zip(self.features, i):
                            likelihood *= self.likelihoods[feat][feat_val + '_' + hasil]
                            evidence *= self.pred_priors[feat][feat_val]
                        posterior = (likelihood * prior) / evidence
                        probs_hasil[hasil] = posterior
                        likelihood_row[hasil] = likelihood
                        evidence_row[hasil] = evidence
                        class_prior_row[hasil] = prior
                    result = max(probs_hasil, key=lambda x: probs_hasil[x])
                    results.append(result)
                    likelihoods.append(likelihood_row)
                    evidences.append(evidence_row)
                    class_priors.append(class_prior_row)
                return np.array(results), likelihoods, evidences, class_priors

        def print_probabilities(self, X):
                predictions, likelihoods, evidences, class_priors = self.predicttt(X)
                table_data = []
                for i, (likelihood, evidence, class_prior) in enumerate(zip(likelihoods, evidences, class_priors)):
                    row_data = {}
                    for label, value in likelihood.items():
                        row_data[f'Likelihood (P(x|{label}))'] = value
                    for label, value in evidence.items():
                        row_data['Evidence (P(X))'] = value
                    for label, value in class_prior.items():
                        row_data[f'Class Prior (P(C={label}))'] = value
                    table_data.append(row_data)
                return table_data


        def calculate_likelihood(self, feature, target, value, cls, alpha=1):
            numerator = len(feature[(target == cls) & (feature == value)]) + alpha
            denominator = len(feature[target == cls]) + alpha * len(np.unique(feature))
            return numerator / denominator

        def train(self, X, y, alpha=1):
            self.X_train = X
            self.y_train = y
            unique_classes = np.unique(y)

            # Calculate prior probabilities
            for cls in unique_classes:
                self.prior[cls] = (len(y[y == cls]) ) / (len(y))

            # Calculate likelihood probabilities for each feature and class
            for feature in X.columns:
                self.likelihood[feature] = {}
                self.evidence[feature] = {}
                unique_values = np.unique(X[feature])
                for cls in unique_classes:
                    self.likelihood[feature][cls] = {}
                    self.evidence[feature][cls] = {}
                    for value in unique_values:
                        likelihood_value = self.calculate_likelihood(X[feature], y, value, cls, alpha)
                        self.likelihood[feature][cls][value] = likelihood_value
                        self.evidence[feature][cls][value] = (len(X[X[feature] == value])) / (len(y) )

        def hasil(self, X):
            predictions = []
            rules = []
            for _, row in X.iterrows():
                posterior = {}
                rule = {}
                for cls in self.prior.keys():
                    posterior[cls] = self.prior[cls]
                    rule[cls] = []
                    for feature in X.columns:
                        likelihood_value = self.likelihood[feature][cls][row[feature]]
                        evidence_value = self.evidence[feature][cls][row[feature]]
                        rule[cls].append(f"P({feature}={row[feature]}|{cls}) = {likelihood_value}")
                    for feature in X.columns:
                        evidence_value = self.evidence[feature][cls][row[feature]]
                        rule[cls].append(f"Evidence({feature}={row[feature]}) = {evidence_value}")
                    rule[cls].append(f"P({cls}) = {self.prior[cls]}")
                    # Calculate posterior probability
                    posterior[cls] = np.prod([self.likelihood[feature][cls][row[feature]] for feature in X.columns]) * self.prior[cls] /np.prod([self.evidence[feature][cls][row[feature]] for feature in X.columns])
                
                # predicted_class = max(posterior, key=posterior.get)
                # predictions.append(predicted_class)
                rules.append(rule)
            cs = pd.DataFrame(rules)
            # X['Predicted Class'] = predictions
            # X['Rules'] = cs
            td = X
            tg = pd.merge(td, cs, left_index=True, right_index=True)
            return tg



        
# Preprocessing
d1 = pd.read_excel("D:/TA/XCEL Over/Latih2050V1_Over.xlsx")
df = pd.DataFrame(d1, columns=["jenis_pkj", "jml_phsl", "jml_art", "pengeluaran", "status_tmpt","klasifikasi"])
df2 = df.loc[:,'jenis_pkj':'klasifikasi']
# st.write(df2)
X,y  = pre_processing(df2)
nb_clf = NaiveBayes()
nb_clf.fit(X, y)
nb_clf.train(X, y)

num_obs = len(df)
#Cek jumlah true
num_true = len(df.loc[df['klasifikasi'] == "LAYAK"])
#Cek Jumlah False
num_false = len(df.loc[df['klasifikasi'] == "TIDAK LAYAK"])
print("Jumlah layak : {0} ({1:2.2f}%)".format(num_true, (num_true/num_obs) * 100))
print("Jumlah tidak layak : {0} ({1:2.2f}%)".format(num_false, (num_false/num_obs) * 100))
dd = num_true/num_obs
de = num_false/num_obs

# Uji Data
d1 = pd.read_excel("D:/TA/XCEL Over/Uji2050V1_Over.xlsx")
df = pd.DataFrame(d1, columns=["jenis_pkj", "jml_phsl", "jml_art", "pengeluaran", "status_tmpt","klasifikasi"])
df2 = df.loc[:,'jenis_pkj':'klasifikasi']
X_test,y_test  = pre_processing(df2)

headerSection = st.container()
mainSection = st.container()
loginSection = st.empty()
placeholder = st.empty()
logOutSection = st.container()
 
def show_main_page():
   
    with mainSection:
        
        today = datetime.datetime.today()
        # -------------------------------------------------------------------------------------------------------------
        # 0000000000000000000000000000000000000000-SIDEBAR NAVIGASI MENU-0000000000000000000000000000000000000000000000
        # -------------------------------------------------------------------------------------------------------------

        with open('style.css') as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
        # navigasi sidebar
        with st.sidebar :
            selected = option_menu('Penerima Bantuan BLT',
            ['Dashboard',
            'Data Warga',
            'Klasifikasi Warga',
            'Prioritas warga'],
            icons= ["house","hdd","book","stack"],
            default_index=0)

        # -------------------------------------------------------------------------------------------------------------
        # 000000000000000000000000000000000000000000000-DASHBOARD-00000000000000000000000000000000000000000000000000000
        # ------------------------------------------------------------------------------------------------------------- 
        # --> DASHBOARD
        
        if (selected == 'Dashboard' ):
            # Row A
            # 
            col1, col2, col3 = st.columns(3)
            col1.metric("Accuracy Latih",(" {}% ".format(accuracy_score(y, nb_clf.predict(X)))) )
            col2.metric("Accuracy Uji",(" {}% ".format(accuracy_score(y_test, nb_clf.predict(X_test)))) )
            col3.metric("Data Latih",("{0} Data ".format(num_obs)),)
            # Row B
            data_likelihoods = nb_clf.likelihoods
            # data_likelihoods = pd.DataFrame(data_likelihoods)
            data_clas_priors = nb_clf.prob_label
            data_pred_prior  = nb_clf.pred_priors
            # # f = nb_clf.features
            # st.dataframe(data_likelihoods)
            # st.dataframe(data_pred_prior)
            c1, c2 = st.columns((7,3))
            with c1:
                st.markdown('### Evidence')
                chart_data1 = pd.DataFrame(
                data= data_pred_prior)
                st.bar_chart(chart_data1)
            with c2:
                st.markdown('### Class Prior')
                values=[9,37]
                labels = 'LAYAK', 'TIDAK LAYAK'
                sizes = [dd, de]
                # only "explode" the 2nd slice (i.e. 'Hogs')
                fig1, ax1 = plt.subplots()
                ax1.pie(sizes, labels=labels, autopct='%1.1f%%',shadow=True, startangle=50)
                ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
                st.pyplot(fig1)
            # Row C
            st.markdown('### Likelihood')
            chart_data3 = pd.DataFrame(data= data_likelihoods)
            st.area_chart(chart_data3)

            # def calculate(expression):
            #     try:
            #         result = eval(expression)
            #         return result
            #     except:
            #         return "Hasil"
            # def calculator(num1, num2, num3):
            #     result = (num1 * num2) / num3
            #     return result

            # st.sidebar.title("Kalkulator")
            # expression = st.sidebar.text_input("Berhitung : ")
            # result = calculate(expression)
            # st.sidebar.code(result)
            # num1 = st.sidebar.number_input("Masukkan Likelihood:")
            # num2 = st.sidebar.number_input("Masukkan Probabilitas Class:")
            # num3 = st.sidebar.number_input("Masukkan Evidence:")
            
            # if st.sidebar.button("Hitung"):
            #     output = calculator(num1, num2, num3)
            #     st.sidebar.success(f"Hasil: {output}")

            kolom = st.columns(9)
            with kolom[0]:
                 st.write()
            with kolom[1]:
                 st.write()
            with kolom[2]:
                 st.write()
            with kolom[3]:
                 st.write()
            with kolom[4]:
                 st.write()
            with kolom[5]:
                 st.write()
            with kolom[6]:
                 st.write()
            with kolom[7]:
                 st.write()
            with kolom[8]:
                loginSection.empty()
                st.button ("Log Out", key="logout", on_click=LoggedOut_Clicked)
          

        # -------------------------------------------------------------------------------------------------------------
        # 00000000000000000000000000000000000000000000-VIEW DATA WARGA-00000000000000000000000000000000000000000000000000000
        # -------------------------------------------------------------------------------------------------------------
        if (selected == 'Data Warga' ):
            menu = ["Data Latih","Data Uji"]
            choice = st.sidebar.selectbox("Tampilkan :",menu)
            
            if choice == "Data Latih":
                    pil = ["Data"]
                    opts = st.sidebar.radio("Aksi :",pil)
                    
                # @@@@@@-VIEW DATA LATIH -@@@@@@ # 00000000000000000000000000000000000000000000-VIEW DATA WARGA-00000000000000000000000000000000000000000000000000000
                    if opts == "Data":
                        st.header("Data Training")
                        file_path = all_data()
                        if file_path:
                            dataset = pd.DataFrame(file_path, columns=["NIK" , "nama", "alamat", "jenis_pkj", "jml_phsl", "jml_art", "pengeluaran", "status_tmpt", "klasifikasi"])
                            top_menu = st.columns(4)
                            with top_menu[0]:
                                sort = st.radio("Sort Data", options=["Yes", "No"], horizontal=1, index=1)
                            if sort == "Yes":
                                with top_menu[1]:
                                    sort_field = st.selectbox("Sort By", options=dataset.columns)
                                with top_menu[2]:
                                    sort_direction = st.radio("Direction", options=["⬆️", "⬇️"], horizontal=True)
                                dataset = dataset.sort_values(by=sort_field, ascending=sort_direction == "⬆️", ignore_index=True)
                            pagination = st.container()
                            bottom_menu = st.columns((4, 1, 1))
                            with bottom_menu[2]:
                                batch_size = st.selectbox("Page Size", options=[25, 50, 100])
                            with bottom_menu[1]:
                                total_pages = (int(len(dataset) / batch_size) if int(len(dataset) / batch_size) > 0 else 1)
                                current_page = st.number_input("Page", min_value=1, max_value=total_pages, step=1)
                            with bottom_menu[0]:
                                st.markdown(f"Page **{current_page}** of **{total_pages}** ")
                            pages = split_frame(dataset, batch_size)
                            pagination.dataframe(data=pages[current_page - 1], use_container_width=True)
                            
                            dta= all_data()
                            dta_Latih = pd.DataFrame(dta, columns=["NIK" , "nama", "alamat", "jenis_pkj", "jml_phsl", "jml_art", "pengeluaran", "status_tmpt", "klasifikasi"])
                            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                                # Write each dataframe to a different worksheet.
                                dta_Latih.to_excel(writer, sheet_name='Sheet1', index=False)
                                # Close the Pandas Excel writer and output the Excel file to the buffer
                                writer.save()
                                download2 = st.download_button(
                                    label="Download Data Latih",
                                    data=buffer,
                                    file_name='ALL_data_Latih.xlsx',
                                    mime='application/vnd.ms-excel')

                # @@@@@@-TAMBAH DATA-@@@@@@
                    # elif opts == "Tambah Data":
                    #     st.header("Tambah Data")
                    #     upload_file = st.file_uploader("Upload Excel file",type=['xlsx'])
                    #     if upload_file:
                    #         df = pd.read_excel(upload_file)
                    #         df= pd.DataFrame(df)
                    #     else:
                    #         def input_user():
                    #             cl1,cl2 = st.columns(2)
                    #             with cl1:
                    #                 NIK = st.text_input("NIK")
                    #                 nama = st.text_input("Nama")
                    #                 alamat = st.selectbox('Alamat',('RW 1','RW 2','RW 3','RW 4','RW 5','RW 6','RW 7','RW 8') )
                    #                 jenis_pkj = st.selectbox('Pekerjaan',('PEDAGANG','BURUH','PETANI','WIRASWASTA','PNS','TIDAK BEKERJA'))
                    #                 jml_phsl = st.selectbox('Penghasilan',('Rp.0 - Rp.1500000','Rp.1500000 - Rp.3000000','Lebih Dari 3000000'))
                    #             with cl2:
                    #                 jml_art = st.selectbox('Anggota Keluarga',('lebih dari 5 orang','3-5 Orang','kurang dari 3 orang'))
                    #                 pengeluaran = st.selectbox('Pengeluaran',('kurang Rp.50.000','Rp.50.000 - Rp.100.000','lebih Rp.100.000'))
                    #                 status_tmpt = st.selectbox('Status Tempat Tinggal',('MILIK SENDIRI','SEWA','NUMPANG'))
                    #                 klasifikasi = st.selectbox('Klasifikasi',('LAYAK','TIDAK LAYAK'))
                    #                 # tgl         = st.date_input("Tanggal")
                    #                 data = {'NIK': NIK,
                    #                         'nama': nama,
                    #                         'alamat': alamat,
                    #                         'jenis_pkj': jenis_pkj,
                    #                         'jml_phsl':  jml_phsl,
                    #                         'jml_art': jml_art,
                    #                         'pengeluaran': pengeluaran,
                    #                         'status_tmpt': status_tmpt,
                    #                         'klasifikasi': klasifikasi}
                    #                 featur = pd.DataFrame(data, index=[0])
                    #                 return featur
                    #         df= input_user()
                    #     # Tombol tambah data
                    #     if upload_file is not None:
                    #         st.table(df)
                    #         dataframe = df
                    #         if st.button('Tambah Data'):
                    #              tambah_data(dataframe) 
                    #              st.success("Success...")
                    #     else:
                    #         st.table(df)
                    #         dataframe = df
                    #         if st.button('Tambah Data'):
                    #              tambah_data(dataframe) 
                    #              st.success("Success...")
                    #     # preview data
                    #     with st.expander("View Tambah data"):
                    #         result = all_data()
                    #         # st.write(result)
                    #         clean_df = pd.DataFrame(result, columns=["NIK" , "nama", "alamat", "jenis_pkj", "jml_phsl", "jml_art", "pengeluaran", "status_tmpt", "klasifikasi"])
                    #         st.dataframe(clean_df)

                # @@@@@@-EDIT DATA-@@@@@@
                    # elif opts == "Edit Data":
                        # Form edit data
                        # st.header('Edit Data')
                        # list_of_NIKS = [i[0] for i in view_NIK()]
                        # selected_NIKS = st.selectbox("NIK",list_of_NIKS)
                        # NIK_result = get_NIK(selected_NIKS)
                        # if NIK_result:
                        #         NIK = NIK_result[0][0]
                        #         nama = NIK_result[0][1]
                        #         alamat = NIK_result[0][2]
                        #         jenis_pkj = NIK_result[0][3]
                        #         jml_phsl = NIK_result[0][4]
                        #         jml_art = NIK_result[0][5]
                        #         pengeluaran = NIK_result[0][6]
                        #         status_tmpt = NIK_result[0][7]
                        #         klasifikasi = NIK_result[0][8]
                        #         tgl         = NIK_result[0][9]
                        #         col1,col2 = st.columns(2)
                        #         with col1:
                        #             NIK = st.code(NIK)
                        #             new_nama        = st.text_input("Nama",nama) 
                        #             new_alamat      = st.selectbox(alamat,('RW 1','RW 2','RW 3','RW 4','RW 5','RW 6','RW 7','RW 8'))
                        #             new_jenis_pkj   = st.selectbox(jenis_pkj,('PEDAGANG','BURUH','PETANI','WIRASWASTA','PNS','TIDAK BEKERJA'))
                        #             new_jml_phsl    = st.selectbox(jml_phsl,('Rp.0 - Rp.1500000','Rp.1500000 - Rp.3000000','Lebih Dari 3000000'))
                        #         with col2:
                        #             new_jml_art = st.selectbox(jml_art,('lebih dari 5 orang','3-5 Orang','kurang dari 3 orang'))
                        #             new_pengeluaran = st.selectbox(pengeluaran,('kurang Rp.50.000','Rp.50.000 - Rp.100.000','lebih Rp.100.000'))
                        #             new_status_tmpt = st.selectbox(status_tmpt,('MILIK SENDIRI','SEWA','NUMPANG'))
                        #             new_klasifikasi = st.selectbox(klasifikasi,('LAYAK','TIDAK LAYAK'))
                        #             new_tgl         = st.date_input("Tanggal")
                        #         if st.button("Update Data"):
                        #             edit_data(new_nama,new_alamat,new_jenis_pkj,new_jml_phsl,new_jml_art,new_pengeluaran,new_status_tmpt,new_klasifikasi,new_tgl,NIK)
                        #             st.success("Updated success...")
                        #         with st.expander("View Updated Data"):
                        #             result = all_data()
                        #             # st.write(result)
                        #             clean_df = pd.DataFrame(result, columns=["NIK" , "nama", "alamat", "jenis_pkj", "jml_phsl", "jml_art", "pengeluaran", "status_tmpt", "klasifikasi","tgl"])
                        #             st.dataframe(clean_df)
                    
        # @@@@@@-000000000-@@@@@@ # 00000000000000000000000000000000000000000000-VIEW DATA WARGA-00000000000000000000000000000000000000000000000000000
            elif choice == "Data Uji":
                    pil = ["Data","Tambah Data","Edit Data","Hapus data"]
                    opts = st.sidebar.radio("Aksi :",pil)
                
                # @@@@@@-VIEW DATA WARGA-@@@@@@
                    if opts == "Data":
                        st.header("Data Warga")
                        file_path = all_data_warga()
                        if file_path:
                            dataset = pd.DataFrame(file_path, columns=["NIK" , "nama", "alamat", "jenis_pkj", "jml_phsl", "jml_art", "pengeluaran", "status_tmpt", "tgl","update"])
                            top_menu = st.columns(4)
                            with top_menu[0]:
                                sort = st.radio("Sort Data", options=["Yes", "No"], horizontal=1, index=1)
                            if sort == "Yes":
                                with top_menu[1]:
                                    sort_field = st.selectbox("Sort By", options=dataset.columns)
                                with top_menu[2]:
                                    sort_direction = st.radio("Direction", options=["⬆️", "⬇️"], horizontal=True)
                                dataset = dataset.sort_values(by=sort_field, ascending=sort_direction == "⬆️", ignore_index=True)
                            pagination = st.container()
                            bottom_menu = st.columns((4, 1, 1))
                            with bottom_menu[2]:
                                batch_size = st.selectbox("Page Size", options=[25, 50, 100])
                            with bottom_menu[1]:
                                total_pages = (int(len(dataset) / batch_size) if int(len(dataset) / batch_size) > 0 else 1)
                                current_page = st.number_input("Page", min_value=1, max_value=total_pages, step=1)
                            with bottom_menu[0]:
                                st.markdown(f"Page **{current_page}** of **{total_pages}** ")
                            pages = split_frame(dataset, batch_size)
                            pagination.dataframe(data=pages[current_page - 1], use_container_width=True)

                            dta= all_data_warga()
                            dta_warga = pd.DataFrame(dta, columns=["NIK" , "nama", "alamat", "jenis_pkj", "jml_phsl", "jml_art", "pengeluaran", "status_tmpt", "tgl","update"])
                            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                                # Write each dataframe to a different worksheet.
                                dta_warga.to_excel(writer, sheet_name='Sheet1', index=False)
                                # Close the Pandas Excel writer and output the Excel file to the buffer
                                writer.save()
                                download2 = st.download_button(
                                    label="Download Data Warga",
                                    data=buffer,
                                    file_name='ALL_data_Warga.xlsx',
                                    mime='application/vnd.ms-excel'
                                )
                                 # 00000000000000000000000000000000000000000000-VIEW DATA WARGA-00000000000000000000000000000000000000000000000000000
                # @@@@@@-TAMBAH DATA WARGA-@@@@@@
                    elif opts == "Tambah Data":
                        st.header("Tambah Data")
                        today = datetime.datetime.today()
                        upload_file = st.file_uploader("Upload Excel file",type=['xlsx'])
                        if upload_file:
                            df = pd.read_excel(upload_file)
                            df['tgl'] = today
                            df['update'] = today
                            # df= pd.DataFrame(df)
                            with st.expander("INPUT DATA WARGA"):
                                result = df
                                # st.write(result)
                                clean_df = result
                                st.table(clean_df)

                        else:
                            def input_user():
                                cl1,cl2 = st.columns(2)
                                with cl1:
                                    NIK = st.text_input("NIK")
                                    nama = st.text_input("Nama")
                                    alamat = st.selectbox('Alamat',('RW 01','RW 02','RW 03','RW 04','RW 05','RW 06','RW 07','RW 08') )
                                    jenis_pkj = st.selectbox('Pekerjaan',('PEDAGANG','BURUH','PETANI','WIRASWASTA','PNS','TIDAK BEKERJA'))
                                    jml_phsl = st.selectbox('Penghasilan',('Rp.0 - Rp.1500000','Rp.1500000 - Rp.3000000','Lebih Dari 3000000'))
                                with cl2:
                                    jml_art = st.selectbox('Anggota Keluarga',('lebih dari 5 orang','3-5 Orang','kurang dari 3 orang'))
                                    pengeluaran = st.selectbox('Pengeluaran',('kurang Rp.50.000','Rp.50.000 - Rp.100.000','lebih Rp.100.000'))
                                    status_tmpt = st.selectbox('Status Tempat Tinggal',('MILIK SENDIRI','SEWA','NUMPANG'))
                                    tgl = today
                                    update= today
                                    data = {'NIK': NIK,
                                            'nama': nama,
                                            'alamat': alamat,
                                            'jenis_pkj': jenis_pkj,
                                            'jml_phsl':  jml_phsl,
                                            'jml_art': jml_art,
                                            'pengeluaran': pengeluaran,
                                            'status_tmpt': status_tmpt,
                                            'tgl': tgl,
                                            'update':update }
                                    featur = pd.DataFrame(data, index=[0])
                                    return featur
                            df= input_user()
                        # Tombol tambah data
                        if upload_file is not None:
                            # st.table(df)
                            dataframe = df
                            if st.button('Tambah Data'):
                                tambah_data_warga(dataframe) 
                                st.success("Berhasil Menambah Data....!!")
                        else:
                            # st.table(df)
                            dataframe = df
                            if st.button('Tambah Data'):
                                tambah_data_warga(dataframe) 
                                st.success("Berhasil Menambah Data....!!")
                        # preview data
                        with st.expander("View Tambah data"):
                            result = all_data_warga()
                            # st.write(result)
                            clean_df = pd.DataFrame(result, columns=["NIK" , "nama", "alamat", "jenis_pkj", "jml_phsl", "jml_art", "pengeluaran", "status_tmpt", "tgl","update"])
                            st.dataframe(clean_df)
                                         # 00000000000000000000000000000000000000000000-VIEW DATA WARGA-00000000000000000000000000000000000000000000000000000
                # @@@@@@-EDIT DATA-@@@@@@
                    elif opts == "Edit Data":
                        # Form edit data
                        today = datetime.datetime.today()
                        
                        st.header('Edit Data')
                        list_of_NIKS = [i[0] for i in view_NIK_warga()]
                        selected_NIKS = st.selectbox("NIK",list_of_NIKS)
                        NIK_result = get_NIK_warga(selected_NIKS)
                        if NIK_result:
                            NIK = NIK_result[0][0]
                            nama = NIK_result[0][1]
                            alamat = NIK_result[0][2]
                            jenis_pkj = NIK_result[0][3]
                            jml_phsl = NIK_result[0][4]
                            jml_art = NIK_result[0][5]
                            pengeluaran = NIK_result[0][6]
                            status_tmpt = NIK_result[0][7]
                                # tgl        = NIK_result[0][8]
                                # update       = NIK_result[0][9]
                            col1,col2 = st.columns(2)
                            with col1:
                                    NIK = NIK
                                    new_nama        = st.text_input("Nama",nama) 
                                    new_alamat      = st.selectbox(alamat,('RW 01','RW 02','RW 03','RW 04','RW 05','RW 06','RW 07','RW 08'))
                                    new_jenis_pkj   = st.selectbox(jenis_pkj,('PEDAGANG','BURUH','PETANI','WIRASWASTA','PNS','TIDAK BEKERJA'))
                                    new_jml_phsl    = st.selectbox(jml_phsl,('Rp.0 - Rp.1500000','Rp.1500000 - Rp.3000000','Lebih Dari 3000000'))
                            with col2:
                                    new_jml_art = st.selectbox(jml_art,('lebih dari 5 orang','3-5 Orang','kurang dari 3 orang'))
                                    new_pengeluaran = st.selectbox(pengeluaran,('kurang Rp.50.000','Rp.50.000 - Rp.100.000','lebih Rp.100.000'))
                                    new_status_tmpt = st.selectbox(status_tmpt,('MILIK SENDIRI','SEWA','NUMPANG'))
                                    # new_tgl       = year
                                    data = {'NIK': NIK,
                                            'nama': new_nama,
                                            'alamat': new_alamat,
                                            'jenis_pkj': new_jenis_pkj,
                                            'jml_phsl':  new_jml_phsl,
                                            'jml_art': new_jml_art,
                                            'pengeluaran': new_pengeluaran,
                                            'status_tmpt': new_status_tmpt }
                            featur = pd.DataFrame(data, index=[0])
                            st.table(featur)
                            
                        
                            if st.button("Update Data"):
                                def normalize_posterior_dataframe(df_prob):
                                    row_sums = df_prob.sum(axis=1)
                                    normalized_dataframe = df_prob.div(row_sums, axis=0)
                                    return normalized_dataframe
                                edit_task_data(new_nama, new_alamat, new_jenis_pkj, new_jml_phsl, new_jml_art, new_pengeluaran, new_status_tmpt, NIK)
                                result = featur
                                    # st.write(result)
                                df          = pd.DataFrame(result, columns=["NIK" , "nama", "alamat", "jenis_pkj", "jml_phsl", "jml_art", "pengeluaran", "status_tmpt"])
                                dff         = df.loc[:,'jenis_pkj':'status_tmpt']
                                df_cus      = df.loc[:,'NIK':'status_tmpt']
                                prediction  = nb_clf.predict(dff)
                                            # prediction_proba = load_clf.predict_proba(df)
                                            # st.subheader('Keterangan Klasifikasi')
                                            # klasifikasi_pd = np.array(['TIDAK LAYAK','LAYAK'])
                                            # st.write(klasifikasi_pd)
                                df_cus["klasifikasi"]   = nb_clf.predict(dff)
                                df_prob                 = pd.DataFrame(nb_clf.predictt(dff)) 
                                nu                       = normalize_posterior_dataframe(df_prob)
                                
                                sisip                   = df_cus.columns.get_loc(key="klasifikasi")
                                df_asli                 = df_cus.loc[:,:]
                                df_gabung               = pd.merge(df_asli, nu, left_index=True, right_index=True)
                                kl_df                   = pd.DataFrame(df_gabung)
                                # st.dataframe(kl_df)
                                
                                conn = create_connection()
                                cursor = conn.cursor()
                                for index, row in kl_df.iterrows():
                                    NIK = row["NIK"]
                                    new_nama = row["nama"]
                                    new_alamat = row["alamat"]
                                    new_jenis_pkj = row["jenis_pkj"]
                                    new_jml_phsl = row["jml_phsl"]
                                    new_jml_art = row["jml_art"]
                                    new_pengeluaran = row["pengeluaran"]
                                    new_status_tmpt = row["status_tmpt"]
                                    new_klasifikasi = row["klasifikasi"]
                                    new_layak = row["LAYAK"]
                                    new_tidaklayak = row["TIDAK LAYAK"]
                                    query = "UPDATE klasifikasi SET nama= %s,alamat=%s,jenis_pkj=%s,jml_phsl=%s,jml_art=%s,pengeluaran=%s,status_tmpt=%s, klasifikasi=%s, layak=%s, tidaklayak=%s WHERE NIK= %s"
                                    data = (new_nama,new_alamat,new_jenis_pkj,new_jml_phsl,new_jml_art,new_pengeluaran,new_status_tmpt,new_klasifikasi,new_layak,new_tidaklayak,NIK)
                                    cursor.execute(query, data)
                                    conn.commit()
                                # NIK = kl_df['NIK']
                                # nama = kl_df['nama']
                                # alamat = kl_df['alamat']
                                # jenis_pkj = kl_df['jenis_pkj']
                                # jml_phsl = kl_df['jml_phsl']
                                # jml_art = kl_df['jml_art']
                                # pengeluaran = kl_df['pengeluaran']
                                # status_tmpt = kl_df['status_tmpt']
                                # klasifikasi = kl_df['klasifikasi']
                                # layak       = kl_df['layak']
                                # tidaklayak  = kl_df['tidaklayak']
                                # tgl         = kl_df['tgl']
                                # edit_kl(nama,alamat,jenis_pkj,jml_phsl,jml_art,pengeluaran,status_tmpt,klasifikasi,layak,tidaklayak,tgl,NIK)
                                # st.success("Updated klasifikasi success...")
                            

                        
                            # with st.expander("View Updated Data"):
                            #         result = all_data_warga()
                            #         # st.write(result)
                            #         clean_df = pd.DataFrame(result, columns=["NIK" , "nama", "alamat", "jenis_pkj", "jml_phsl", "jml_art", "pengeluaran", "status_tmpt","tgl","update"])
                            #         st.dataframe(clean_df)
                                 # 00000000000000000000000000000000000000000000-VIEW DATA WARGA-00000000000000000000000000000000000000000000000000000
                    elif opts  == "Hapus data":
                        st.subheader("Hapus Data")
                        with st.expander("View Data"):
                            result = all_data_warga()
                            # st.write(result)
                            clean_df = pd.DataFrame(result, columns=["NIK" , "nama", "alamat", "jenis_pkj", "jml_phsl", "jml_art", "pengeluaran", "status_tmpt","tgl","update"])
                            st.dataframe(clean_df)

                        unique_list = [i[0] for i in view_NIK_warga()]
                        delete_by_NIK =  st.selectbox("Pilih Warga Yang Ingin Dihapus",unique_list)
                        v1= st.columns(2)
                        with v1[0]:
                            if st.button("Hapus"):
                                delete_data(delete_by_NIK)
                                st.warning("Data: '{}' Terhapus".format(delete_by_NIK))
                        with v1[1]:
                               if st.button("Delete ALL"):
                                    delete_ALL()
                                    st.warning("Deleted all... ")
                            
                        with st.expander("Data Warga"):
                            result = all_data_warga()
                            # st.write(result)
                            clean_df = pd.DataFrame(result, columns=["NIK" , "nama", "alamat", "jenis_pkj", "jml_phsl", "jml_art", "pengeluaran", "status_tmpt","tgl","update"])
                            st.dataframe(clean_df)



# =============================================================================================================
# 00000000000000000000000000000000000000000000-NAIVE BAYES MENU-00000000000000000000000000000000000000000000000
# =============================================================================================================
        # --> NAIVE BAYES
        if (selected == 'Klasifikasi Warga' ):
            # st.title('Klasifikasi Warga')
        
            st.write("""
            # KLASIFIKASI PENERIMA BLT
            """)
            # try:
            sort = st.radio("Klasifikasi Data", options=["Periksa", "Data Baru", "Result"], horizontal=1, index=0)
            if sort == "Data Baru":
                with st.expander("View Data Warga"):
                    result = all_data_warga_beside_KL()
                    # st.write(result)
                    clean_df = pd.DataFrame(result, columns=["NIK" , "nama", "alamat", "jenis_pkj", "jml_phsl", "jml_art", "pengeluaran", "status_tmpt","tgl","update"])
                    st.dataframe(clean_df)

                st.subheader("Nilai Setiap Record:")
                with st.expander("Probabilitas setiap kategori:"):
                    st.caption("\n Berikut Nilai :green[Likelihood], :blue[Evidence] dan :red[ClassPrior] dari setiap data uji baru:")
                    dff = clean_df.loc[:,'jenis_pkj':'status_tmpt']
                    dff = pd.DataFrame(dff)   
                    d=nb_clf.hasil(dff)
                    st.dataframe(d, use_container_width=True)
               
                st.subheader("Variabel Hitung:")
                with st.expander("Variable Hitung:"):
                    st.caption("\n Berikut Nilai :green[Likelihood], :blue[Evidence] dan :red[ClassPrior] dari setiap data uji baru:")
                    dff = clean_df.loc[:,'jenis_pkj':'status_tmpt']
                    dff = pd.DataFrame(dff)     
                    nama = clean_df.loc[:,['nama']]
                    b = pd.DataFrame(nb_clf.print_probabilities(dff))
                    b["Nama"]      = nama
                    fers            = b.pop('Nama')
                    b.insert(0,'Nama',fers)
                    st.dataframe(b, use_container_width=True)
                
                st.subheader("Hasil Klasifikasi:")
                with st.expander("Result:"):
                    st.caption("\nHasil klasifikasi berupa probabilitas posterior setiap hipotesa class, yang diperoleh dengan formula berikut:")
                    st.latex(r'''
                            P(H|X)=\left(\frac{P(X|H)*P(H)}{P(X)}\right) 
                            ''')
                    st.caption("Dua kolom terahir adalah posterior klasifikasi setiap class!")
                    dff = clean_df.loc[:,'jenis_pkj':'status_tmpt']
                    dff = pd.DataFrame(dff)
                    # st.dataframe(dff)
                    
                    # Normalisasi posterior
                    def normalize_posterior_dataframe(df_prob):
                            row_sums = df_prob.sum(axis=1)
                            normalized_dataframe = df_prob.div(row_sums, axis=0)
                            return normalized_dataframe
                    
                    result = all_data_warga_beside_KL()
                    # st.write(result)
                    df          = pd.DataFrame(result, columns=["NIK" , "nama", "alamat", "jenis_pkj", "jml_phsl", "jml_art", "pengeluaran", "status_tmpt","tgl","update"])
                    df_cus      = df.loc[:,'NIK':'status_tmpt']
                    prediction  = nb_clf.predict(dff)
                            # prediction_proba = load_clf.predict_proba(df)
                            # st.subheader('Keterangan Klasifikasi')
                            # klasifikasi_pd = np.array(['TIDAK LAYAK','LAYAK'])
                            # st.write(klasifikasi_pd)
                    df_cus["klasifikasi"]   = nb_clf.predict(dff)
                    df_prob                 = pd.DataFrame(nb_clf.predictt(dff)) 
                    # df_prob["tgl"]          = today
                    # st.dataframe(df_prob)
                    n                       = normalize_posterior_dataframe(df_prob)
                    n["tgl"]                = today
                    # df_prob["update"]       = today
                    # sisip                   = df_cus.columns.get_loc(key="klasifikasi")
                    df_asli                 = df_cus.loc[:,:]
                    df_gabung               = pd.merge(df_asli, n, left_index=True, right_index=True)
                    df_gabung["id_kl"]      = None
                    fers                    = df_gabung.pop('id_kl')
                    df_gabung.insert(0,'id_kl',fers)
                    # Menghasilkan kode ID otomatis
                    
                    row_count = len(df_gabung)
                    conn = create_connection()
                    cursor = conn.cursor()
                    cursor.execute("SELECT COUNT(*) FROM klasifikasi")
                    result = cursor.fetchone()
                    is_database_empty = result[0] == 0
                    if is_database_empty:
                        # Jika database kosong, kode ID dimulai dari 'kls001'
                        kode_id = ['KLS' + str(i).zfill(3) for i in range(1, row_count + 1)]
                    else:
                        # Jika terdapat data lain di database, mengambil kode ID terakhir yang tersimpan
                        cursor.execute("SELECT id_kl FROM klasifikasi ORDER BY id_kl DESC LIMIT 1")
                        result = cursor.fetchone()
                        last_kode_id = result[0]

                        last_number = int(last_kode_id[3:])
                        kode_id = ['KLS' + str(last_number + i).zfill(3) for i in range(1, row_count + 1)]

                    # Buat daftar kode ID berurutan
                    # Tambahkan kolom kode ID ke dataframe
                    # df['kode_id'] = id_list 
                    
                    df_gabung['id_kl'] = kode_id

                    clean_df1 = df_gabung
                    st.dataframe(clean_df1)
                    
                
                if st.button('Save'):
                        connection = create_connection()
                        save(clean_df1, 'klasifikasi', connection)  
                        st.success('Data inserted successfully!')
            
                 # 00000000000000000000000000000000000000000000-NAIVE BAYES MENU-00000000000000000000000000000000000000000000000
                    # X_Hasil = df_gabung.loc[:,['NIK','nama','alamat','jenis_pkj','jml_phsl','jml_art','pengeluaran','status_tmpt',]]
                    # st.write(hasil)

                    # st.subheader('Target Layak')
                    # X_layak = df.loc[df['klasifikasi']=='LAYAK',['NIK','nama','alamat','jenis_pkj','jml_phsl','jml_art','pengeluaran','status_tmpt','klasifikasi']]
                    # st.table(X_layak)

                    # # download button 2 to download dataframe as xlsx
                    # with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                    #     # Write each dataframe to a different worksheet.
                    #     df_gabung.to_excel(writer, sheet_name='Sheet1', index=False)
                    #     # Close the Pandas Excel writer and output the Excel file to the buffer
                    #     writer.save()

                    #     download2 = st.download_button(
                    #         label="Download Hasil",
                    #         data=buffer,
                    #         file_name='ALL.xlsx',
                    #         mime='application/vnd.ms-excel'
                    #     )
                    # with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                    #     # Write each dataframe to a different worksheet.
                    #     X_layak.to_excel(writer, sheet_name='Sheet1', index=False)
                    #     # Close the Pandas Excel writer and output the Excel file to the buffer
                    #     writer.save()

                    #     download2 = st.download_button(
                    #         label="Download Hasil Layak",
                    #         data=buffer,
                    #         file_name='Layak.xlsx',
                    #         mime='application/vnd.ms-excel'
                    #     )
             # 00000000000000000000000000000000000000000000-NAIVE BAYES MENU-00000000000000000000000000000000000000000000000
                                
            if sort == "Periksa":
                # Normalisasi posterior
                def normalize_posterior_dataframe(dataframe):
                    total = dataframe.sum().sum()
                    normalized_dataframe = dataframe / total
                    return normalized_dataframe
                
                menu = ["Pilih NIK","Checking"]
                choice = st.sidebar.selectbox("Tampilkan :",menu)
                if choice == "Pilih NIK":
                    list_of_NIKS = [i[0] for i in view_NIK_warga()]
                    selected_NIKS = st.selectbox("NIK",list_of_NIKS)
                    NIK_result = get_NIK_warga(selected_NIKS)
                    def input_user():
                        NIK = NIK_result[0][0]
                        nama = NIK_result[0][1]
                        alamat = NIK_result[0][2]
                        jenis_pkj = NIK_result[0][3]
                        jml_phsl = NIK_result[0][4]
                        jml_art = NIK_result[0][5]
                        pengeluaran = NIK_result[0][6]
                        status_tmpt = NIK_result[0][7]
                        data = {'NIK': NIK,
                                'nama': nama,
                                'alamat': alamat,
                                'jenis_pkj': jenis_pkj,
                                'jml_phsl':  jml_phsl,
                                'jml_art': jml_art,
                                'pengeluaran': pengeluaran,
                                'status_tmpt': status_tmpt}
                        featur = pd.DataFrame(data, index=[0])
                        return featur
                    df = input_user()
                    dff = df.loc[:,'jenis_pkj':'status_tmpt']
                    if NIK_result:
                        st.table(df)

                        # prediction  = nb_clf.predict(dff)
                        # prediction_proba = load_clf.predict_proba(df)
                        # st.subheader('Keterangan Klasifikasi')
                        # klasifikasi_pd = np.array(['TIDAK LAYAK','LAYAK'])
                        # st.write(klasifikasi_pd)
                        df["klasifikasi"]       = nb_clf.predict(dff)
                        df_prob                 = pd.DataFrame(nb_clf.predictt(dff)) 
                        # st.dataframe(df_prob)
                        nor                     = normalize_posterior_dataframe(df_prob)
                        nor["tgl"]              = today

                        # sisip                   = df.columns.get_loc(key="klasifikasi")
                        df_asli                 = df.loc[:,:]
                        df_gabung               = pd.merge(df_asli, nor, left_index=True, right_index=True)
                       
                        st.caption("Nilai Setiap Record")
                        d=nb_clf.hasil(dff)
                        st.dataframe(d)

                        st.caption("Variabel Hitung")
                        nama = df.loc[:,['nama']]
                        b= pd.DataFrame(nb_clf.print_probabilities(dff))
                        b["Nama"]      = nama
                        fers             = b.pop('Nama')
                        b.insert(0,'Nama',fers)
                         
                        st.dataframe(b)

                        st.caption("Klasifikasi")
                        clean_df = df_gabung
                        st.dataframe(clean_df)

                    # dataframe = pd.DataFrame(clean_df)
                    # # st.dataframe(dataframe)
                    # if st.button('save'):
                    #     connection = create_connection()
                    #     save(dataframe, 'klasifikasi', connection)  
                    #     st.success('Data inserted successfully!')
                # ---------------------------------------------------------------------------------------
                if choice == "Checking": 
                        
                    def normalize_posterior_dataframe(df_prob):
                            row_sums = df_prob.sum(axis=1)
                            normalized_dataframe = df_prob.div(row_sums, axis=0)
                            return normalized_dataframe
                                 
                    def input_user():
                        NIK = st.sidebar.text_input("NIK")
                        nama = st.sidebar.text_input("Nama")
                        alamat = st.sidebar.selectbox('Alamat',('RW 1','RW 2','RW 3','RW 4','RW 5','RW 6','RW 7','RW 8') )
                        jenis_pkj = st.sidebar.selectbox('Pekerjaan',('PEDAGANG','BURUH','PETANI','WIRASWASTA','PNS','TIDAK BEKERJA'))
                        jml_phsl = st.sidebar.selectbox('Penghasilan',('Rp.0 - Rp.1500000','Rp.1500000 - Rp.3000000','Lebih Dari 3000000'))
                        jml_art = st.sidebar.selectbox('Anggota Keluarga',('lebih dari 5 orang','3-5 Orang','kurang dari 3 orang'))
                        pengeluaran = st.sidebar.selectbox('Pengeluaran',('kurang Rp.50.000','Rp.50.000 - Rp.100.000','lebih Rp.100.000'))
                        status_tmpt = st.sidebar.selectbox('Status Tempat Tinggal',('MILIK SENDIRI','SEWA','NUMPANG'))
                        # klasifikasi = st.sidebar.selectbox('Klasifikasi',('LAYAK','TIDAK LAYAK'))
                        data = {'NIK': NIK,
                                'nama': nama,
                                'alamat': alamat,
                                'jenis_pkj': jenis_pkj,
                                'jml_phsl':  jml_phsl,
                                'jml_art': jml_art,
                                'pengeluaran': pengeluaran,
                                'status_tmpt': status_tmpt}
                        featur = pd.DataFrame(data, index=[0])
                        return featur
                    df= input_user()
                    dff = df.loc[:,'jenis_pkj':'status_tmpt']
                    prediction  = nb_clf.predict(dff)
                    # prediction_proba = load_clf.predict_proba(df)
                    # st.subheader('Keterangan Klasifikasi')
                    # klasifikasi_pd = np.array(['TIDAK LAYAK','LAYAK'])
                    # st.write(klasifikasi_pd)
                    df["klasifikasi"]   = nb_clf.predict(dff)
                    df_prob             = pd.DataFrame(nb_clf.predictt(dff)) 
                    # df_prob["tgl"]      = today
                    norm                = normalize_posterior_dataframe(df_prob)
                    norm["tgl"]         = today
                   
                    # sisip                   = df.columns.get_loc(key="klasifikasi")
                    df_asli             = df.loc[:,:]
                    df_gabung           = pd.merge(df_asli, norm, left_index=True, right_index=True)
                    
                    clean_df = df_gabung
                    st.dataframe(clean_df)
                    with st.expander("Perhitungan:"):
                            st.caption("Nilai Setiap Record")
                            d=nb_clf.hasil(dff)
                            st.dataframe(d)

                            st.caption("Variable Hitung")
                            nama = df.loc[:,['nama']]
                            b= pd.DataFrame(nb_clf.print_probabilities(dff))
                            b["Nama"]      = nama
                            fers           = b.pop('Nama')
                            b.insert(0,'Nama',fers)
                            st.dataframe(b)

                            st.caption("Klasifikasi")
                            clean_df = df_gabung
                            st.dataframe(clean_df)


             # 00000000000000000000000000000000000000000000-NAIVE BAYES MENU-00000000000000000000000000000000000000000000000

            if sort == "Result":
                with st.expander("View Data Warga"):
                    result = all_data_klasifikasi()
                    # st.write(result)
                    clean_df = pd.DataFrame(result, columns=["id_kl" ,"NIK" , "nama", "alamat", "jenis_pkj", "jml_phsl", "jml_art", "pengeluaran", "status_tmpt","klasifikasi","Layak","Tidak layak","tgl"])
                    st.dataframe(clean_df)
                dataframe= clean_df
                with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                            # Write each dataframe to a different worksheet.
                            dataframe.to_excel(writer, sheet_name='Sheet1', index=False)
                            # Close the Pandas Excel writer and output the Excel file to the buffer
                            writer.save()
                            download2 = st.download_button(
                                label="Download Hasil Klasifikasi",
                                data=buffer,
                                file_name='ALL_Klasifikasi.xlsx',
                                mime='application/vnd.ms-excel'
                            )

            # Collects user input features into dataframe
        
                # Reads in saved classification model
            # load_clf = pickle.load(open('modell.pkl','rb'))
                # Apply model to make predictions
            # st.table(dff)
            

            # csv
            # st.download_button(label='Download Target Layak',data=X_layak.to_excel(header=True, index=False),file_name='Layak.xlas') 
            # st.download_button(label='Download Semua',data=X_Hasil.to_excel(header=True, index=False),file_name='ALL.xlsx')

            # st.subheader('Prediction Probability')
            # st.write(prediction_proba)
            # except:
            #     st.write("Please load a file to continue...")






# -------------------------------------------------------------------------------------------------------------
# 00000000000000000000000000000000000000000-WEIGHTED PRODUCT-00000000000000000000000000000000000000000000000000
# -------------------------------------------------------------------------------------------------------------


        # --> WEIGHTED PRODUCT
        if (selected == 'Prioritas warga' ):
            # st.title('Prioritas warga')
            st.write("""
            # RANGKING PRIORITAS PENERIMA BLT
            """)
            # 00000000000000000000000000000000000000000-WEIGHTED PRODUCT-00000000000000000000000000000000000000000000000000
            st.subheader("1. Daftar Klasifikasi Layak")
            with st.expander("View Data Warga"):
                result = all_layak_KL()
                    # st.write(result)
                df_c= pd.DataFrame(result, columns=["id_kl" ,"NIK" , "nama", "alamat", "jenis_pkj", "jml_phsl", "jml_art", "pengeluaran", "status_tmpt","klasifikasi","Layak","Tidak layak","tgl"])
                dg =  df_c.loc[:,'id_kl':'klasifikasi']
                st.dataframe(dg)

            dgg = df_c.drop([ "Layak","Tidak layak","tgl"], axis=1)
              # dgg = sls.drop(['id_kl','NIK', 'nama','alamat', 'klasifikasi','Layak','Tidak layak', 'tgl'], axis=1)
            k = dgg.columns
            kriteria = k.values[4:9]
            alternatif = dgg.iloc[:,2].values
                # st.table(alternatif)
        # Langkah 1# 00000000000000000000000000000000000000000-WEIGHTED PRODUCT-00000000000000000000000000000000000000000000000000
            def conversionAllNumberInput(conversion):
                # Conversion for every number of test
                    if 'PEDAGANG' == conversion: return 3
                    elif 'PETANI' == conversion: return 5
                    elif 'TIDAK BEKERJA' == conversion: return 6
                    elif 'BURUH' == conversion: return 4
                    elif 'WIRASWASTA' == conversion: return 2
                    elif 'PNS' == conversion: return 1

                    elif 'MILIK SENDIRI' == conversion: return 1
                    elif 'SEWA' == conversion: return 5
                    elif 'NUMPANG' == conversion: return 2
                    
                    elif 'kurang Rp.50.000' == conversion: return 1
                    elif 'Rp.50.000 - Rp.100.000' == conversion: return 3
                    elif 'lebih Rp.100.000' == conversion: return 5

                    elif 'Lebih Dari 3000000' == conversion: return 5
                    elif 'Rp.0 - Rp.1500000' == conversion: return 1
                    elif 'Rp.1500000 - Rp.3000000' == conversion: return 3

                    elif '3-5 Orang' == conversion: return 3
                    elif 'kurang dari 3 orang' == conversion: return 1
                    elif 'lebih dari 5 orang' == conversion: return 5
            #data setelah diconversi
            dataTestValues = dgg.iloc[:, 4:9].values
    
            # Konversi seluruh data sesuai range yang sudah ditentukan
            for i in range(len(dataTestValues)) :  
                    for j in range(len(dataTestValues[i])) :
                        dataTestValues[i][j] = conversionAllNumberInput(dataTestValues[i][j])
            # st.dataframe(dataTestValues)
        # langkah 2# 00000000000000000000000000000000000000000-WEIGHTED PRODUCT-00000000000000000000000000000000000000000000000000
            #devinisi Cost Benefit
            st.subheader("2. Aturan Pemeringkatan")
            with st.expander("Langkah Weighted Product:"):
                st.markdown("Sifat kriteria yang dai gunakan tiap kriterianya sebagaimana berikut:")
                st.markdown("a. Sifat kriteria : _['benefit' ,'cost', 'benefit' ,'cost' ,'benefit']_.")
                st.markdown("b. Bobot Criteria : _[4,5,4,3,4]_.")
                st.markdown("c. Nilai Pangkat : _[0.2, -0.25, 0.2, -0.15, 0.2]_.")
                st.markdown("d. Mencari Nilai S.")
                st.latex(r'''
                            S_i  = \Pi_{j=1}^{n} {X_{ij}}^{w_j} 
                            ''')
                st.markdown("e. Mencari Nilai Vektor. ")
                st.latex(r'''
                            V_i  = \left(\frac{\Pi_{j=1}^{n}{X_{ij}}^{w_j}} {\Pi_{j=1}^{n}(X_j)^{w_j}}\right) 
                            ''')
            
            costbenefit = ["benefit" ,"cost", "benefit" ,"cost" ,"benefit"]
            bobotCriteria = [4,5,4,3,4] # Penentuan bobot 
                # st.dataframe(bobotCriteria)

            jumlahkepentingan = 0
            for i in range(len(kriteria)) :
                    jumlahkepentingan = jumlahkepentingan + bobotCriteria[i]    
            # st.write(jumlahkepentingan)
            bobotkepentingan = []
            for i in range(len(kriteria)) :
                    bobotkepentingan.append(bobotCriteria[i]/jumlahkepentingan)
            # st.write(bobotkepentingan)
        # langkah 3# 00000000000000000000000000000000000000000-WEIGHTED PRODUCT-00000000000000000000000000000000000000000000000000
            pangkat = []
            for i in range(len(kriteria)):
                    pangkat.append(0)
                    if costbenefit[i] == 'cost':
                        pangkat[i] = -1 * bobotkepentingan[i]
                    elif costbenefit[i] == 'benefit':
                        pangkat[i] = 1 * bobotkepentingan[i]
            # st.markdown(pangkat)
            # st.markdown("2. Nilai Pangkat(W):,")

        # langkah 4 # 00000000000000000000000000000000000000000-WEIGHTED PRODUCT-00000000000000000000000000000000000000000000000000
            nilai_s = []
            total_s = 0
            for i in range(len(alternatif)):
                    nilai_s.append(1)
                    for j in range(len(kriteria)):
                        nilai_s[i] = nilai_s[i] * (dataTestValues[i][j] ** pangkat[j])
                    total_s = total_s + nilai_s[i]

        # langkah 5    #  # 00000000000000000000000000000000000000000-WEIGHTED PRODUCT-00000000000000000000000000000000000000000000000000
            vektor_V = []
            for i in range(len(alternatif)):
                    vektor_V.append(nilai_s[i]/total_s)
                # st.table(vektor_V)

                # kolomkriteria = pd.DataFrame({'Nama': alternatif,'Hasil':vektor_V})
                # st.table(kolomkriteria)

            st.subheader("3. Nilai Vektor S")
            with st.expander("View Nilai Vektor S:"):
                hsll=  dg["Vektor S"] = pd.DataFrame(nilai_s)
                X_HasilS = dg.loc[:,["id_kl" ,'NIK','nama','alamat','jenis_pkj','jml_phsl','jml_art','pengeluaran','status_tmpt','klasifikasi','Vektor S']]
                st.dataframe(X_HasilS)

            st.subheader("4. Nilai Vektor V")
            with st.expander("View Nilai Vektor V:"):
                hasill = dg["Vektor V"] = pd.DataFrame(vektor_V)
                X_HasilV = dg.loc[:,["id_kl" ,'NIK','nama','alamat','jenis_pkj','jml_phsl','jml_art','pengeluaran','status_tmpt','klasifikasi','Vektor V']]
                st.dataframe(X_HasilV)

        # sorting 6# 00000000000000000000000000000000000000000-WEIGHTED PRODUCT-00000000000000000000000000000000000000000000000000
            st.subheader("5. Urutan Prioritas")
            with st.expander("View Peringkat Prioritas:"):
                Sorting = X_HasilV.sort_values(by=['Vektor V'], ascending=False)
                Sorting['Peringkat'] =  range( 1, len(Sorting) + 1)

                st.dataframe(Sorting, use_container_width=True)

                # if st.button('save'):
                #         connection = create_connection()
                #         save(dataframe, 'klasifikasi', connection)  
                #         st.success('Data inserted successfully!')

            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                    # Write each dataframe to a different worksheet.
                    Sorting.to_excel(writer, sheet_name='Sheet1', index=False)
                    # Close the Pandas Excel writer and output the Excel file to the buffer
                    writer.save()

                    download2 = st.download_button(
                        label="Download Hasil",
                        data=buffer,
                        file_name='ALLWP.xlsx',
                        mime='application/vnd.ms-excel'
                    )



            # st.sidebar.header('INPUTAN')
            # Collects user input features into dataframe
        
                # upload_file = st.file_uploader("Upload Excel file", type=["xlsx"])
                # if upload_file:
                #     dg = pd.read_excel(upload_file)
                #     st.table(dg)
            
                
            #         dgg = dg.drop(['NIK','alamat', 'klasifikasi'], axis=1)
            # # Dapatkan Header alias kriteria setiap file
            

            # Maka kita bisa mengambil potensi nilai tertinggi (Melakukan Pengurutan Dari Terbesar ke Terkecil)
            # sorting = (vektor_V.sort)
            # st.write(sorting)


            # Simpan menjadi sebuah file Excel


            # kolomkriteria.to_excel(writer,'Sheet1',index=False,header=False)
            # st.write(kolomkriteria)

            # list = [
            #     ["Nama","Hasil"],
            #     [alternatif],
            #     [vektor_V]
            # ]
            # st.write(list)

            # Simpan menjadi sebuah file Excel
            # barisalternatif = pd.DataFrame(barisalternatif[0:])
            # kolomkriteria = pd.DataFrame(['Nama','Hasil'])
            # kolomkriteria = kolomkriteria.transpose()
            # dfNilaiVektorV = pd.DataFrame(vektor_V)
            # writer = ExcelWriter('HasilWP2.xlsx')
            # barisalternatif.to_excel(writer,'Sheet1',index=False,header=False,startrow=1)
            # kolomkriteria.to_excel(writer,'Sheet1',index=False,header=False)
            # dfNilaiVektorV.to_excel(writer,'Sheet1',index=False,header=False,startrow=1,startcol=1)
            # writer.save()


            # dfNilaiVektorV = pd.DataFrame(vektor_V)
            # writer = ExcelWriter('HasilWP2.xlsx')
            # barisalternatif.to_excel(writer,'Sheet1',index=False,header=False,startrow=1)
            # kk.to_excel(writer,'Sheet1',index=False,header=False)
            # dfNilaiVektorV.to_excel(writer,'Sheet1',index=False,header=False,startrow=1,startcol=1)
            # kw=writer.save()
            # st.write(kw)
            # urutan prioritas
            #print(hasil_akhir .sort_values(by=['Hasil'], ascending=False).iloc[:])


        # -------------------------------------------------------------------------------------------------------------
        # 0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
        # -------------------------------------------------------------------------------------------------------------



with headerSection:
    #first run will have nothing in session_state
    if 'loggedIn' not in st.session_state:
        st.session_state['loggedIn'] = False
        show_login_page() 
    else:
        if st.session_state['loggedIn']:
            show_main_page()  
        else:
            show_login_page()

