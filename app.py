import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import streamlit as st
import altair as alt
import locale
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error
import joblib as jbl

# locale.setlocale(locale.LC_ALL, 'ID')

# config
st.set_page_config(layout="wide")

# pm = pd.read_csv('./files/data/hackathon-pelihara_alatangkutan.csv')
# aset = pd.read_csv('./files/data/hackathon-masteraset_alatangkutan.csv')

pm = pd.read_csv('./files/data/dummy.csv')
aset = pd.read_csv('./files/data/dummy_aset.csv')

daftar_satker = sorted(set(pm['KODE_SATKER']))
daftar_satker.insert(0, "Pilih Satker")

# format function
fmt = lambda x: "{:,}".format(x)

# filter satker:
select_satker = st.sidebar.selectbox(
  'Satuan Kerja',
  (daftar_satker)
)

## sidebar - filter satker
def navbar():
  navigation = st.sidebar.radio(
    'Menu',
    ('Ringkasan', 'Anomali', 'Prediksi Anggaran','Lanjutan', 'Quick Report'))
  return navigation

navigation = ''

# Preprocessing Semua Satker
pm = pm[['KODE_SATKER','KODE_BARANG','NUP','NO_DIPA','TGL_SP2D','NILAI','JENIS_PEMELIHARAAN']]
aset = aset[['KODE_SATKER','KODE_BARANG','NUP','NAMA_BARANG','TANGGAL_PEROLEHAN','NILAI_PEROLEHAN_PERTAMA','NILAI_BUKU','KONDISI','MERK']]

pm['ID_BARANG']=pm['KODE_SATKER']+pm['KODE_BARANG'].astype('str')+pm['NUP'].astype('str')
aset['ID_BARANG']=aset['KODE_SATKER']+aset['KODE_BARANG'].astype('str')+aset['NUP'].astype('str')

aset.drop(['KODE_SATKER', 'KODE_BARANG', 'NUP'], axis=1,inplace=True)

df = pm.join(aset.set_index('ID_BARANG'), on='ID_BARANG')

df['TGL_SP2D'] = pd.to_datetime(df['TGL_SP2D'])
df['TANGGAL_PEROLEHAN'] = pd.to_datetime(df['TANGGAL_PEROLEHAN'])



# all raw data
def show_raw():
  # Biaya Pemeliharaan Satker
  df_i = df.groupby(['KODE_SATKER','ID_BARANG']).agg({'NILAI':'sum'}).reset_index()
  df_i = df_i.groupby(['KODE_SATKER']).agg({'NILAI':'mean'}).reset_index()
  
  cost_all = alt.Chart(df_i).mark_bar().encode(
      x = 'KODE_SATKER:O',
      y = 'NILAI:Q'
  ).properties(
      height=400,
      title = 'Rata-Rata Biaya Pemeliharaan Satker per Tahun'
  )

  # Frekuensi pemeliharaan
  df_ii = df.groupby(['KODE_SATKER','ID_BARANG',df['TGL_SP2D'].dt.year]).agg({'NILAI':'count'}).reset_index()
  df_ii = df_ii.groupby(['KODE_SATKER']).agg({'NILAI':'mean'}).reset_index()

  frek_all = alt.Chart(df_ii).mark_bar().encode(
      x = 'KODE_SATKER:O',
      y = 'NILAI:Q'
  ).properties(
      height=400,
      title = 'Rata-Rata Frekuensi Pemeliharaan Barang per Tahun'
  )
  
  # Biaya Pemeliharaan Tahun
  df_ia = df.groupby([df['TGL_SP2D'].dt.year]).agg({'NILAI':'sum'}).reset_index()
  df_ia = df_ia.rename(columns={'TGL_SP2D':'TAHUN'})
  cost_all_ia = alt.Chart(df_ia).mark_line().encode(
      x = 'TAHUN:O',
      y = 'NILAI:Q'
  ).properties(
      height=400,
      title = 'Nilai Pemeliharaan Barang'
  )
  
  
  # st.write(df)
  st.text('')
  
  left_column, right_column = st.columns(2)
  
  with left_column:
    st.altair_chart(cost_all, use_container_width=True)
    st.session_state.rata_cost = df_i.NILAI.mean()
    st.write('Rata-Rata Biaya Pemeliharaan/Tahun : ', fmt(df_i.NILAI.mean()))
    st.header('')
    st.altair_chart(cost_all_ia, use_container_width=True)

  with right_column:
    st.altair_chart(frek_all, use_container_width=True)
    st.session_state.rata_frek = df_ii.NILAI.mean()
    st.write('Rata-Rata Frekuensi Pemeliharaan/Tahun : ', "{:.2f}".format(df_ii.NILAI.mean()))

# visual satker
def visual(navigation, satker):
  if navigation == 'Ringkasan':
    st.header('Ringkasan Data Pemeliharaan Aset pada Satker '+satker)
    show_summary(satker)
  elif navigation == 'Anomali':
    st.header('Anomali Data Barang pada Satker '+satker)
    show_anomaly(satker)
  elif navigation =='Prediksi Anggaran':
    st.header('Prediksi Anggaran Tahun Depan')
    show_predict(satker)
  elif navigation =='Lanjutan':
    st.header('Analisis Lanjutan')
    show_advanced(satker)
  elif navigation =='Quick Report':
    st.header('Quick Report')
    show_quickreport(satker)
  
# satker summary
def show_summary(satker):
  left_column, right_column = st.columns(2)
  
  ## Biaya pemeliharaan (dalam tahun)
  df_iii = df[df['KODE_SATKER']==satker]
  df_iii = df_iii.reset_index(drop=True)
  
  dfiii_1 = df_iii.groupby(['ID_BARANG',df_iii['TGL_SP2D'].dt.year]).agg({'NILAI':'mean'}).reset_index()
  dfiii_1 = dfiii_1.groupby([df_iii['TGL_SP2D'].dt.year]).agg({'NILAI':'sum'}).reset_index()
  
  dfiii_1 = dfiii_1.rename(columns={'TGL_SP2D':'TAHUN_PEMELIHARAAN'})
  
  iii1a = int(dfiii_1.mean()['NILAI'])
  iii1b = int(dfiii_1['NILAI'].max())
  iii1c = int(dfiii_1.sort_values(['NILAI'],ascending=False).reset_index()['TAHUN_PEMELIHARAAN'][0])

  ### chart - biaya pemeliharaan (dalam tahun)
  mean_nilai = alt.Chart(dfiii_1).mark_rule(color='red').encode(
    y = 'mean(NILAI)',
    size=alt.value(2)
  )
  dfiii1_chart = alt.Chart(dfiii_1).mark_line().encode(
    x = 'TAHUN_PEMELIHARAAN:O',
    y = 'NILAI:Q',
  ).properties(
      height=400,
      title = 'Grafik Biaya Pemeliharaan Tiap Tahun'
  )
  
  # id barang
  id_barang = df_iii.groupby(['ID_BARANG',df_iii['TGL_SP2D'].dt.year]).agg({'KODE_BARANG':'min','NAMA_BARANG':'min','NUP':'min'}).reset_index()
  id_barang = id_barang.astype(str)
  # id_barang = set((id_barang['KODE_BARANG']+id_barang['NUP']+', '+id_barang['NAMA_BARANG']).tolist())
  id_barang = set((id_barang['ID_BARANG']))
  
  ## Biaya pemeliharaan rata-rata (per barang)
  dfiii_2 = df_iii.groupby(['ID_BARANG',df_iii['TGL_SP2D'].dt.year]).agg({'NILAI':'mean'}).reset_index()
  dfiii_2 = dfiii_2.rename(columns={'TGL_SP2D':'TAHUN'})
  
  iii2a = int(dfiii_2['NILAI'].mean())
  iii2b = int(dfiii_2['NILAI'].max())
  iii2c = dfiii_2.sort_values(['NILAI'],ascending=False).reset_index()['ID_BARANG'][0]
  iii2c_nama = df_iii[df_iii['ID_BARANG']==iii2c].reset_index()['NAMA_BARANG'][0]
  iii2c_merk = df_iii[df_iii['ID_BARANG']==iii2c].reset_index()['MERK'][0]
  
  
  ## Frekuensi Pemeliharaan (per barang)
  dfiii_3 = df_iii.groupby(['ID_BARANG',df_iii['TGL_SP2D'].dt.year]).agg({'NILAI':'count'}).reset_index()
  dfiii_3 = dfiii_3.rename(columns={'TGL_SP2D':'TAHUN','NILAI':'FREK'})
  
  iii3a = int(dfiii_3['FREK'].mean())
  iii3b = int(dfiii_3['FREK'].max())
  iii3c = dfiii_3.sort_values(['FREK'],ascending=False).reset_index()['ID_BARANG'][0]
  iii3c_nama = df_iii[df_iii['ID_BARANG']==iii3c].reset_index()['NAMA_BARANG'][0]
  iii3c_merk = df_iii[df_iii['ID_BARANG']==iii3c].reset_index()['MERK'][0]
  
  ## show summary
  with left_column:
    ## Show - Biaya pemeliharaan (dalam tahun)
    st.subheader('Biaya pemeliharaan (dalam tahun)')
    # st.write(dfiii_1)
    st.altair_chart(dfiii1_chart+mean_nilai, use_container_width=True)
    st.write('- Rata-rata biaya pemeliharaan per tahun sebesar',fmt(iii1a))
    st.write('- Biaya pemeliharaan tertinggi terdapat pada tahun', str(iii1c), 'sebesar', fmt(iii1b))

  with right_column:
    ## Show - Biaya Pemeliharaan per Barang
    st.subheader('Biaya Pemeliharaan per Barang')
    
    select_barang_by = st.multiselect(
      'Silahkan pilih kode barang untuk ditampilkan:',
      id_barang,
      key=0
      )
    st.session_state.select_barang_by=select_barang_by
    try:
      dfiii2_data = dfiii_2[dfiii_2['ID_BARANG'].isin(st.session_state.select_barang_by)]
    except:
      dfiii2_data = dfiii_2.copy()
      
    ### Chart - Biaya pemeliharaan rata-rata (per barang)
    dfiii2_chart = alt.Chart(dfiii2_data).mark_line().encode(
      x = 'TAHUN:O',
      y = 'NILAI:Q',
      color='ID_BARANG:N'
    ).properties(
        height=400,
        title = 'Grafik Biaya Pemeliharaan per Barang'
    )
    st.altair_chart(dfiii2_chart, use_container_width=True)
    
    st.write('- Rata-rata biaya pemeliharaan per barang sebesar',fmt(iii2a))
    st.write('- Rata-rata biaya pemeliharaan tertinggi terdapat pada barang', str(iii2c),'yaitu',iii2c_nama,'merk',iii2c_merk, 'sebesar', fmt(iii2b))  

    ## Show - Frekuensi Pemeliharaan per Barang
    st.subheader('Frekuensi Pemeliharaan per Barang')
    select_barang_fr = st.multiselect(
      'Silahkan pilih kode barang untuk ditampilkan:',
      id_barang,
      key=1
      )
    st.session_state.select_barang_fr=select_barang_fr
    try:
      dfiii3_data = dfiii_3[dfiii_3['ID_BARANG'].isin(st.session_state.select_barang_fr)]
    except:
      dfiii3_data = dfiii_3.copy()
      
    ### Chart - Frekuensi Pemeliharaan (per barang)
    dfiii3_chart = alt.Chart(dfiii3_data).mark_line().encode(
      x = 'TAHUN:O',
      y = 'FREK:Q',
      color = 'ID_BARANG:N',
    ).properties(
        height=400,
        title = 'Grafik Frekuensi Pemeliharaan per Barang'
    )
    
    st.altair_chart(dfiii3_chart, use_container_width=True)
    
    st.write('- Rata-rata frekuensi pemeliharaan / barang sebanyak',str(iii3a),'kali')
    st.write('- Frekuensi pemeliharaan tertinggi pada barang', iii3c,'yaitu',iii3c_nama,'merk',iii3c_merk, 'sebanyak', str(iii3b), 'kali')

# anomaly
def show_anomaly(satker):
  df_iii = df[df['KODE_SATKER']==satker]
  
  dfiii_4 = df_iii[['ID_BARANG','NILAI','TGL_SP2D']].reset_index(drop=True).copy()
  dfiii_4['TAHUN_PM']=dfiii_4['TGL_SP2D'].dt.year
  dfiii_4.drop(['TGL_SP2D'],axis=1,inplace=True)
  dfiii_4['FREK']=dfiii_4.groupby(['TAHUN_PM','ID_BARANG'])['ID_BARANG'].transform('count')
  dfiii_4['NILAI']=dfiii_4.groupby(['TAHUN_PM','ID_BARANG'])['NILAI'].transform('sum')
  dfiii_4.drop_duplicates(subset=['ID_BARANG','TAHUN_PM','NILAI'], inplace=True)
  scale = StandardScaler()
  dfiii_4a = dfiii_4.drop(['ID_BARANG','TAHUN_PM'], axis=1)
  kmeans = KMeans(n_clusters=1, random_state=0)
  data = scale.fit_transform(dfiii_4a)
  dfiii_4 = dfiii_4.reset_index(drop=True)
  # menggabungkan data yang sudah discale dengan dataframe referensi
  scaled_df = pd.concat([dfiii_4,pd.DataFrame(data)],axis=1).rename(columns={0:'SCALED_N',1:'SCALED_F'})
  
  cluster = kmeans.fit_predict(data)
  centroids = kmeans.cluster_centers_
  points = np.empty((0,len(data[0])), float)
  distances = np.empty((0,len(data[0])), float)
  
  for i, center_elem in enumerate(centroids):
    distances = np.append(distances, cdist([center_elem],data[cluster == i], 'euclidean')) 
    points = np.append(points, data[cluster == i], axis=0)
    
  # outliers ditentukan dengan jarak lebih dari nilai percentile
  left_column, right_column = st.columns(2)
  with left_column:
    if 'percentile' in st.session_state:
      percentile = st.session_state.percentile
    else:
      percentile = 99.0
    st.write('Silahkan tentukan jarak outlier dengan slider berikut (default 99):')
    percentile_val = st.slider(
        '',
        min_value = 0.0,
        max_value = 100.0,
        value = percentile
      )
    
  try:
    percentile = percentile_val
  except:
    percentile = 99
    
  outliers = points[np.where(distances > np.percentile(distances, percentile))]
  
  data = pd.DataFrame(data)
  data.rename(columns={0:'x',1:'y'}, inplace=True)
  outliers = pd.DataFrame(outliers)
  outliers.rename(columns={0:'x',1:'y'}, inplace=True)
  outliers1 = pd.DataFrame(outliers)
  center_point = pd.DataFrame(centroids)
  center_point.rename(columns={0:'x',1:'y'}, inplace=True)
  
  # mencari barang yang menjadi outliers
  outliers2 = scaled_df[(scaled_df['SCALED_N'].isin(outliers1['x']))&(scaled_df['SCALED_F'].isin(outliers1['y']))]
  
  ano_chart = alt.Chart(data).mark_circle(size=60).encode(
    x='x',
    y='y',
  ).properties(
    height=400,
    width=400
  )
  
  outl_chart = alt.Chart(outliers).mark_point(size=300).encode(
    x='x',
    y='y',
    opacity=alt.value(0.8),
    color=alt.value('red')
  )
  
  centr_chart = alt.Chart(center_point).mark_point(size=20).encode(
    x='x',
    y='y',
    opacity=alt.value(0.8),
    color=alt.value('red')
  )
  st.altair_chart(ano_chart+outl_chart+centr_chart)
  st.write('Jika ingin menghapus outliers pada saat train model, klik tombol di bawah untuk menyimpan terlebih dahulu')
  def save_outliers():
    st.session_state.outliers = outliers2[['ID_BARANG','NILAI','TAHUN_PM','FREK']]
    st.session_state.percentile = percentile
    st.write('Outlier tersimpan!')
  saveout_btn = st.button(
    'Simpan',
  )
  if saveout_btn:
    save_outliers()
  st.write(outliers2[['ID_BARANG','NILAI','TAHUN_PM','FREK']])
  
  
  # transaksi outliers
  for index, row in outliers2.iterrows():
    trans_outl = df_iii[(df_iii['ID_BARANG']==row['ID_BARANG'])]
    st.subheader(row['ID_BARANG'])
    st.write(trans_outl[['TGL_SP2D','NILAI','JENIS_PEMELIHARAAN']].sort_values(['TGL_SP2D']))

# prediction
def show_predict(satker):
  df_iii = df[df['KODE_SATKER']==satker]
  
  if 'dfiii_removed' in st.session_state:
    df_iii = st.session_state.dfiii_removed
  
  
  df_iv = df_iii[['ID_BARANG','TGL_SP2D','NILAI','TANGGAL_PEROLEHAN']].reset_index(drop=True)
  df_iv['TAHUN_PM'] = df_iv['TGL_SP2D'].dt.year
  df_iv['TAHUN_PR'] = df_iv['TANGGAL_PEROLEHAN'].dt.year
  df_iv = df_iv.rename(columns={'NILAI':'NILAI_PM'})
  df_iv.drop(['TGL_SP2D','TANGGAL_PEROLEHAN'],axis=1,inplace=True)
  enc_barang = LabelEncoder()
  df_iv['ID_BARANG'] = enc_barang.fit_transform(df_iv['ID_BARANG'])
  df_iv['NILAI_PM'] = df_iv.groupby(['TAHUN_PM','ID_BARANG'])['NILAI_PM'].transform('mean')
  df_iv = df_iv.drop_duplicates(subset=['ID_BARANG','NILAI_PM']).reset_index(drop=True)
  
  X_train = df_iv.drop(['NILAI_PM'],axis=1)
  y_train = df_iv['NILAI_PM']
  
  dfiii_1 = df_iii[['ID_BARANG','TGL_SP2D','NILAI','TANGGAL_PEROLEHAN']].reset_index(drop=True)
  dfiii_1['TAHUN_PM'] = dfiii_1['TGL_SP2D'].dt.year
  dfiii_1['TAHUN_PR'] = dfiii_1['TANGGAL_PEROLEHAN'].dt.year
  dfiii_1 = dfiii_1.rename(columns={'NILAI':'NILAI_PM'})
  
  dfiii_1['NILAI_PM'] = dfiii_1.groupby(['TAHUN_PM','ID_BARANG'])['NILAI_PM'].transform('mean')
  dfiii_1 = dfiii_1.drop_duplicates(subset=['ID_BARANG','NILAI_PM'])
  
  # st.write(dfiii_1.drop(['TGL_SP2D','TANGGAL_PEROLEHAN'],axis=1).sort_values(['ID_BARANG','TAHUN_PM']).reset_index(drop=True))
  
  model=None
  
  def search_model(method):
    if method == 'RandomForest':
      first = 'rf_'
    if method == 'DecisionTree':
      first = 'dt_'
    if method == 'LightGBM':
      first = 'lg_'
    return first
  
  if 'outliers' in st.session_state:
    outl =  st.session_state.outliers
    st.write('Daftar outliers: ')
    st.write(st.session_state.outliers.reset_index(drop=True))
    outl_btn = st.button(
      'Hapus outliers (opsional)',
    )
    if outl_btn:
      rem_outliers(outl)
  else:
    st.write('**_Outlier tidak ditemukan. Jika ingin menghapus outlier, silahkan buka menu Anomali_**')
  
  st.subheader('Pilih metode:')
  method = st.radio(
    '',
    ('RandomForest', 'DecisionTree', 'LightGBM'),
    index =0 
  )
  
  first = search_model(method)
  st.header('')
  try:
    model = jbl.load('./files/models/'+first+satker+'.pkl')
    st.write('**_Model berhasil dimuat, klik tombol Fit Model jika ingin melakukan fit/train ulang (file lama akan terhapus!)_**')
    st.write(first+satker+'.pkl')
  except:
    st.write("Model tidak ditemukan atau belum dibuat. Klik tombol di bawah untuk fit model")
  
  st.button(
    'Fit Model',
    on_click=fit_model,
    args = (first, X_train, y_train, satker)
  )
  
  if model:
    st.button(
      'Predict',
      on_click=pred_result,
      args=(X_train, model, enc_barang, df_iii)
    )
  
  if 'prediction_res' in st.session_state:
    df_prediction = st.session_state.prediction_res
    st.write(df_prediction)
    st.write(df_prediction.groupby(['NAMA_BARANG']).agg({'PREDIKSI_ANGG':'sum'}).reset_index())
    st.write('Total Kebutuhan Anggaran Pemeliharaan tahun depan = ', fmt(st.session_state.prediction_res['PREDIKSI_ANGG'].sum()))
    
## remove outliers
def rem_outliers(outliers):
  st.session_state.dfiii_removed = df_iii[(df_iii['ID_BARANG']==outliers['ID_BARANG'])&(df_iii['TAHUN_PM']==outliers['TAHUN_PM'])]

## model fit
def fit_model(first, X_train, y_train, satker):
  if first=='rf_':
    md = RandomForestRegressor(random_state=0)
  if first=='dt_':
    md = DecisionTreeRegressor(random_state=0)
  if first=='lg_':
    md = LGBMRegressor(random_state=0)
    
  md.fit(X_train, y_train)
  jbl.dump(md,'./files/models/'+first+satker+'.pkl', compress=9)

## prediction results
def pred_result(X_train, model, enc_barang, df_iii):
  # X_test = pd.DataFrame({'ID_BARANG':np.arange(0,X_train['ID_BARANG'].max())})
  X_test = X_train.groupby(['ID_BARANG']).agg({'TAHUN_PR':'mean'}).reset_index()
  # buat dummy X_test untuk prediksi, tahunnya diambil dari tahun pemeliharaan terakhir + 1
  X_test['TAHUN_PM']=X_train['TAHUN_PM'].max()+1
  X_test['TAHUN_PR']=X_train['TAHUN_PR'].astype(int)
  rf_res = model.predict(X_test)
  
  df_v = X_test.reset_index(drop=True).copy()
  df_v['ID_BARANG'] = enc_barang.inverse_transform(df_v['ID_BARANG'].to_list())
  df_v1 = df_iii.drop(['KODE_SATKER','NO_DIPA','JENIS_PEMELIHARAAN','TANGGAL_PEROLEHAN','NILAI_BUKU'],axis=1).copy()
  df_v1['TAHUNPM_AKHIR'] = df_v1['TGL_SP2D'].dt.year
  df_v1['TAHUNPM'] = df_v1['TGL_SP2D'].dt.year
  df_v1['TAHUNPM_AKHIR'] = df_v1.groupby(['ID_BARANG'])['TAHUNPM_AKHIR'].transform('max')
  
  df_v1['NILAI'] = df_v1.groupby(['TAHUNPM','ID_BARANG'])['NILAI'].transform('sum')
  df_v1['FREK'] = df_v1.groupby(['TAHUNPM','ID_BARANG'])['KODE_BARANG'].transform('count')
  df_v1 = df_v1.rename(columns={'NILAI':'NILAI_TOTAL','NILAI_PEROLEHAN_PERTAMA':'NILAI_PR'})
  df_v1 = df_v1.drop_duplicates(subset=['ID_BARANG','NILAI_TOTAL','FREK']).reset_index(drop=True)
  df_v = pd.concat([df_v,pd.DataFrame({'PREDIKSI_ANGG':rf_res})],axis=1)
  
  df_v = df_v.join(df_v1.drop(['TGL_SP2D'],axis=1).set_index('ID_BARANG'), on='ID_BARANG').reset_index(drop=True)
  df_v['PREDIKSI_ANGG'] = df_v['PREDIKSI_ANGG'].astype('int')
  
  df_v['NILAI_TOTAL']=df_v.groupby(['ID_BARANG'])['NILAI_TOTAL'].transform('mean')
  df_v['FREK']=df_v.groupby(['ID_BARANG'])['FREK'].transform('mean')
  
  df_v = df_v.drop_duplicates(subset=['ID_BARANG']).reset_index(drop=True)
  
  st.session_state.prediction_res = df_v[['KODE_BARANG','NUP','NAMA_BARANG','KONDISI','MERK','FREK','PREDIKSI_ANGG']]
  st.session_state.df_vi = df_v
  

# advanced
def show_advanced(satker):
  df_iii = df[df['KODE_SATKER']==satker]
  try:
    st.write(st.session_state.df_vi)
    show_options()
  except:
    st.write('_Sebelum melakukan analisis lanjutan, mohon lakukan proses Fit Model (jika belum dilakukan) dan Predict pada menu Prediksi Anggaran_')

## advanced options
def show_options():
  st.subheader('Opsi Analisis Lanjutan')
  with st.form("my_form"):
    st.write('1. Frekuensi Pemeliharaan')
    def_frek = round(st.session_state.rata_frek,2)
    frek_val = st.number_input(
      'Frekuensi pemeliharaan rata-rata per tahun lebih dari:', 
      min_value=0.0, 
      max_value=30.0, 
      value=def_frek,
      step = 1.0
    )
    st.write('2. Kondisi Barang')
    cond_val = [False, False, False]
    cond_val[0] = st.checkbox('Baik')
    cond_val[1] = st.checkbox(
      'Rusak Ringan',
      value=True
      )
    cond_val[2] = st.checkbox(
      'Rusak Berat',
      value=True
      )

    st.write('3. Biaya Pemeliharaan')
    cost_val = st.number_input(
      'Persentase biaya pemeliharaan dibandingkan dengan nilai perolehan (dalam %):',
      min_value = 0.0,
      step=10.0,
      value = 100.0
    )
    
    st.write('4.Perkiraan Biaya Pemeliharaan')
    meancost_val = st.number_input(
      'Persentase nilai perkiraan biaya pemeliharaan dibandingkan dengan rata-rata pemeliharaan per tahun (dalam %):',
      min_value = 100.0,
      value = 200.0,
      step=10.0,
    )
    kondisi = ['Baik','Rusak Ringan','Rusak Berat']
    option_cond = [cond_val[0],cond_val[1],cond_val[2]]
    
    submitted = st.form_submit_button("Submit")
    if submitted:
        str_kondisi = []
        for i in range(3):
            if option_cond[i]:
                str_kondisi.append(kondisi[i])

        str_cond = ['-','','','','']
        str_cond[1] = '- Rata-rata frekuensi pemeliharaan > '+ str(frek_val)
        str_cond[2] = '- Kondisi Barang : ' + ', '.join(str_kondisi)
        str_cond[3] = '- Total biaya pemeliharaan telah mencapai/melebihi '+ str(cost_val) + '% biaya perolehan'
        str_cond[4] = '- Perkiraan biaya pemeliharaan mencapai/melebihi ' + str(meancost_val) + '% biaya pemeliharaan rata-rata per tahun'
        
        st.session_state.str_cond = str_cond
        
        st.write('_Berikut tabel barang yang memenuhi minimal 1 kondisi di bawah:_')
        for i in range(1,5):
          st.write(str_cond[i])
        str_cond[2] = ''
        
        condition_table = show_advanced_report(frek_val, cond_val, cost_val, meancost_val)
        show_detail_report(condition_table)
  st.write(cond_val)    
      
def cond_1(row, frek_val):
  if row['FREK']>frek_val:
      return 1
  return 0

def cond_2(row, cond_val):
  if (row['KONDISI']=='Baik' and cond_val[0]) or (row['KONDISI']=='Rusak Ringan' and cond_val[1]) or (row['KONDISI']=='Rusak Berat' and cond_val[2]):
      return 1
  return 0

def cond_3(row, cost_val):
    if row['NILAI_TOTAL']>row['NILAI_PR']*(cost_val/100):
        return 1
    return 0
  
def cond_4(row, meancost_val):
    if row['PREDIKSI_ANGG']>((meancost_val/100)*(row['NILAI_TOTAL']/(row['TAHUNPM_AKHIR']-row['TAHUN_PR']+1))):
        return 1
    return 0

## advanced report
def show_advanced_report(frek_val, cond_val, cost_val, meancost_val):
  df_vii = st.session_state.df_vi
  df_vii['COND_1'] = df_vii.apply(lambda row: cond_1(row, frek_val), axis=1)
  df_vii['COND_2'] = df_vii.apply(lambda row: cond_2(row, cond_val), axis=1)
  df_vii['COND_3'] = df_vii.apply(lambda row: cond_3(row, cost_val), axis=1)
  df_vii['COND_4'] = df_vii.apply(lambda row: cond_4(row, meancost_val), axis=1)
  
  df_cond = df_vii[(df_vii['COND_1']==1)|(df_vii['COND_2']==1)|(df_vii['COND_3']==1)|(df_vii['COND_4']==1)]
  
  st.session_state.df_cond=df_cond
  st.write(df_cond[['ID_BARANG','TAHUN_PR','PREDIKSI_ANGG','KODE_BARANG','NUP','NAMA_BARANG','MERK','KONDISI','FREK']])
  return df_cond
  
## detail report
def show_detail_report(condition_table):
  
  cond = st.session_state.str_cond
  
  st.subheader('Detail Barang:')
  for i, row in condition_table.iterrows():
    nm_barang = row['NAMA_BARANG']
    merk = row['MERK']
    barang = row['ID_BARANG']+', '+row['NAMA_BARANG']+', '+row['MERK']+', '+row['KONDISI']
    st.markdown(f'**_{barang}_**')
    
    st.markdown(f'<a href="https://e-katalog.lkpp.go.id/id/search-produk?q={nm_barang} {merk}" target="_blank">Cari di E-Katalog</a>', unsafe_allow_html=True)
    for j in range(1,5):
        if row['COND_'+str(j)]:
            st.write(cond[j])
    st.write('')
  
# quick report
def show_quickreport(satker):
  quick_btn = st.button(
    'Generate',
  )
  if quick_btn:
    rep1(satker)
    rep2(satker)
    rep3(satker)
    rep4(satker)

def rep1(satker):
  df_iii = df[df['KODE_SATKER']==satker]
  
  df_iii = df_iii.reset_index(drop=True)
  
  dfiii_1 = df_iii.groupby(['ID_BARANG',df_iii['TGL_SP2D'].dt.year]).agg({'NILAI':'mean'}).reset_index()
  dfiii_1 = dfiii_1.groupby([df_iii['TGL_SP2D'].dt.year]).agg({'NILAI':'sum'}).reset_index()
  
  dfiii_1 = dfiii_1.rename(columns={'TGL_SP2D':'TAHUN_PEMELIHARAAN'})
  
  iii1a = int(dfiii_1.mean()['NILAI'])
  iii1b = int(dfiii_1['NILAI'].max())
  iii1c = int(dfiii_1.sort_values(['NILAI'],ascending=False).reset_index()['TAHUN_PEMELIHARAAN'][0])

  # id barang
  id_barang = df_iii.groupby(['ID_BARANG',df_iii['TGL_SP2D'].dt.year]).agg({'KODE_BARANG':'min','NAMA_BARANG':'min','NUP':'min'}).reset_index()
  id_barang = id_barang.astype(str)
  # id_barang = set((id_barang['KODE_BARANG']+id_barang['NUP']+', '+id_barang['NAMA_BARANG']).tolist())
  id_barang = set((id_barang['ID_BARANG']))
  
  ## Biaya pemeliharaan rata-rata (per barang)
  dfiii_2 = df_iii.groupby(['ID_BARANG',df_iii['TGL_SP2D'].dt.year]).agg({'NILAI':'mean'}).reset_index()
  dfiii_2 = dfiii_2.rename(columns={'TGL_SP2D':'TAHUN'})
  
  iii2a = int(dfiii_2['NILAI'].mean())
  iii2b = int(dfiii_2['NILAI'].max())
  iii2c = dfiii_2.sort_values(['NILAI'],ascending=False).reset_index()['ID_BARANG'][0]
  iii2c_nama = df_iii[df_iii['ID_BARANG']==iii2c].reset_index()['NAMA_BARANG'][0]
  iii2c_merk = df_iii[df_iii['ID_BARANG']==iii2c].reset_index()['MERK'][0]
  
  ## Frekuensi Pemeliharaan (per barang)
  dfiii_3 = df_iii.groupby(['ID_BARANG',df_iii['TGL_SP2D'].dt.year]).agg({'NILAI':'count'}).reset_index()
  dfiii_3 = dfiii_3.rename(columns={'TGL_SP2D':'TAHUN','NILAI':'FREK'})
  
  iii3a = int(dfiii_3['FREK'].mean())
  iii3b = int(dfiii_3['FREK'].max())
  iii3c = dfiii_3.sort_values(['FREK'],ascending=False).reset_index()['ID_BARANG'][0]
  iii3c_nama = df_iii[df_iii['ID_BARANG']==iii3c].reset_index()['NAMA_BARANG'][0]
  iii3c_merk = df_iii[df_iii['ID_BARANG']==iii3c].reset_index()['MERK'][0]
  
  ## Show - Biaya pemeliharaan (dalam tahun)
  st.subheader('Biaya Pemeliharaan ')
  
  ### chart - biaya pemeliharaan (dalam tahun)
  mean_nilai = alt.Chart(dfiii_1).mark_rule(color='red').encode(
    y = 'mean(NILAI)',
    size=alt.value(2)
  )
  dfiii1_chart = alt.Chart(dfiii_1).mark_line().encode(
    x = 'TAHUN_PEMELIHARAAN:O',
    y = 'NILAI:Q',
  ).properties(
      height=400,
      title = 'Grafik Biaya Pemeliharaan Tiap Tahun'
  )
  
  st.altair_chart(dfiii1_chart+mean_nilai, use_container_width=True)
  st.write('- Rata-rata biaya pemeliharaan per tahun sebesar',fmt(iii1a))
  st.write('- Biaya pemeliharaan tertinggi terdapat pada tahun', str(iii1c), 'sebesar', fmt(iii1b)) 
  
  st.write('- Rata-rata biaya pemeliharaan per barang sebesar',fmt(iii2a))
  st.write('- Rata-rata biaya pemeliharaan tertinggi terdapat pada barang', str(iii2c),'yaitu',iii2c_nama,'merk',iii2c_merk, 'sebesar', fmt(iii2b))  
  
  st.write('- Rata-rata frekuensi pemeliharaan per barang sebanyak',str(iii3a),'kali')
  st.write('- Frekuensi pemeliharaan tertinggi pada barang', iii3c,'yaitu',iii3c_nama,'merk',iii3c_merk, 'sebanyak', str(iii3b), 'kali')

def rep2(satker):
  df_iii = df[df['KODE_SATKER']==satker]
  st.subheader('Daftar Pencilan (Outlier)')

def rep3(satker):
  df_iii = df[df['KODE_SATKER']==satker]
  st.subheader('Prediksi Anggaran')
  
  df_iv = df_iii[['ID_BARANG','TGL_SP2D','NILAI','TANGGAL_PEROLEHAN']].reset_index(drop=True)
  df_iv['TAHUN_PM'] = df_iv['TGL_SP2D'].dt.year
  df_iv['TAHUN_PR'] = df_iv['TANGGAL_PEROLEHAN'].dt.year
  df_iv = df_iv.rename(columns={'NILAI':'NILAI_PM'})
  df_iv.drop(['TGL_SP2D','TANGGAL_PEROLEHAN'],axis=1,inplace=True)
  enc_barang = LabelEncoder()
  df_iv['ID_BARANG'] = enc_barang.fit_transform(df_iv['ID_BARANG'])
  df_iv['NILAI_PM'] = df_iv.groupby(['TAHUN_PM','ID_BARANG'])['NILAI_PM'].transform('mean')
  df_iv = df_iv.drop_duplicates(subset=['ID_BARANG','NILAI_PM']).reset_index(drop=True)
  
  X_train = df_iv.drop(['NILAI_PM'],axis=1)
  y_train = df_iv['NILAI_PM']
  
  dfiii_1 = df_iii[['ID_BARANG','TGL_SP2D','NILAI','TANGGAL_PEROLEHAN']].reset_index(drop=True)
  dfiii_1['TAHUN_PM'] = dfiii_1['TGL_SP2D'].dt.year
  dfiii_1['TAHUN_PR'] = dfiii_1['TANGGAL_PEROLEHAN'].dt.year
  dfiii_1 = dfiii_1.rename(columns={'NILAI':'NILAI_PM'})
  
  dfiii_1['NILAI_PM'] = dfiii_1.groupby(['TAHUN_PM','ID_BARANG'])['NILAI_PM'].transform('mean')
  dfiii_1 = dfiii_1.drop_duplicates(subset=['ID_BARANG','NILAI_PM'])
  
  try:
    model = jbl.load('./files/models/rf_'+satker+'.pkl')
  except:
    fit_model('rf_', X_train, y_train, satker)
    model = jbl.load('./files/models/rf_'+satker+'.pkl')

  result = pred_result(X_train, model, enc_barang, df_iii)
  
  
  
  if 'prediction_res' in st.session_state:
    df_prediction = st.session_state.prediction_res
    st.write(df_prediction)
    st.write(df_prediction.groupby(['NAMA_BARANG']).agg({'PREDIKSI_ANGG':'sum'}).reset_index())
    st.write('Total Kebutuhan Anggaran Pemeliharaan tahun depan = ', fmt(st.session_state.prediction_res['PREDIKSI_ANGG'].sum()))
    
def rep4(satker):
  frek_val = round(st.session_state.rata_frek,2)
  cond_val = ['Rusak Ringan','Rusak Berat']
  cost_val = 100.0
  meancost_val = 200.0

  str_cond = ['-','','','','']
  str_cond[1] = '- Rata-rata frekuensi pemeliharaan > '+ str(frek_val)
  str_cond[2] = '- Kondisi Barang : ' + ', '.join(cond_val)
  str_cond[3] = '- Total biaya pemeliharaan telah mencapai/melebihi '+ str(cost_val) + '% biaya perolehan'
  str_cond[4] = '- Perkiraan biaya pemeliharaan mencapai/melebihi ' + str(meancost_val) + '% biaya pemeliharaan rata-rata per tahun'
        
  st.write('_Berikut tabel barang yang memenuhi minimal 1 kondisi di bawah:_')
  for i in range(1,5):
    st.write(str_cond[i])
  str_cond[2] = ''
  st.session_state.str_cond = str_cond
  
  condition_table = show_advanced_report2(frek_val, cond_val, cost_val, meancost_val)
  show_detail_report(condition_table)

def cond_2a(row, cond_val):
  if (row['KONDISI']==cond_val[0]) or (row['KONDISI']==cond_val[1]):
      return 1
  return 0

## advanced report
def show_advanced_report(frek_val, cond_val, cost_val, meancost_val):
  df_vii = st.session_state.df_vi
  df_vii['COND_1'] = df_vii.apply(lambda row: cond_1(row, frek_val), axis=1)
  df_vii['COND_2'] = df_vii.apply(lambda row: cond_2(row, cond_val), axis=1)
  df_vii['COND_3'] = df_vii.apply(lambda row: cond_3(row, cost_val), axis=1)
  df_vii['COND_4'] = df_vii.apply(lambda row: cond_4(row, meancost_val), axis=1)
  
  df_cond = df_vii[(df_vii['COND_1']==1)|(df_vii['COND_2']==1)|(df_vii['COND_3']==1)|(df_vii['COND_4']==1)]
  
  st.write(df_cond[['ID_BARANG','TAHUN_PR','PREDIKSI_ANGG','KODE_BARANG','NUP','NAMA_BARANG','MERK','KONDISI','FREK']])
  return df_cond

def show_advanced_report2(frek_val, cond_val, cost_val, meancost_val):
  df_vii = st.session_state.df_vi
  df_vii['COND_1'] = df_vii.apply(lambda row: cond_1(row, frek_val), axis=1)
  df_vii['COND_2'] = df_vii.apply(lambda row: cond_2a(row, cond_val), axis=1)
  df_vii['COND_3'] = df_vii.apply(lambda row: cond_3(row, cost_val), axis=1)
  df_vii['COND_4'] = df_vii.apply(lambda row: cond_4(row, meancost_val), axis=1)
  
  df_cond = df_vii[(df_vii['COND_1']==1)|(df_vii['COND_2']==1)|(df_vii['COND_3']==1)|(df_vii['COND_4']==1)]
  
  st.write(df_cond[['ID_BARANG','TAHUN_PR','PREDIKSI_ANGG','KODE_BARANG','NUP','NAMA_BARANG','MERK','KONDISI','FREK']])
  return df_cond

## detail report
def show_detail_report(condition_table):
  
  cond = st.session_state.str_cond
  
  st.subheader('Detail Barang:')
  for i, row in condition_table.iterrows():
    nm_barang = row['NAMA_BARANG']
    merk = row['MERK']
    barang = row['ID_BARANG']+', '+row['NAMA_BARANG']+', '+row['MERK']+', '+row['KONDISI']
    st.markdown(f'**_{barang}_**')
    
    st.markdown(f'<a href="https://e-katalog.lkpp.go.id/id/search-produk?q={nm_barang} {merk}" target="_blank">Cari di E-Katalog</a>', unsafe_allow_html=True)
    for j in range(1,5):
        if row['COND_'+str(j)]:
            st.write(cond[j])
    st.write('')

# main layout
if select_satker!='Pilih Satker':
  nav = navbar()
  visual(nav, select_satker)
else:
  show_raw()