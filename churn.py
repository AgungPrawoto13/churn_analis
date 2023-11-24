import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import psycopg2
conn = psycopg2.connect(
    dbname="web",
    options="endpoint=ep-white-hall-00289384",
    user="josebenaya",
    password="PWo3ZhTkEA1a",
    host="ep-white-hall-00289384.us-east-2.aws.neon.tech",
    port="5432",
)


# Read data from PostgreSQL into DataFrames
df_query = "SELECT * FROM telco_customer_churn_adapted_v2"  # Replace with your actual query

df = pd.read_sql(df_query, conn)
# df = pd.read_excel('data/Telco_customer_churn_adapted_v2.xlsx')
churn = df.loc[df['Churn Label'] == 'Yes']

def cek(x):
    if x == 'Yes':
        return 1
    else:
        return 0

def kategori_monthly(x):
    if x < 20:
        return 20
    elif x < 40:
        return 40
    elif x < 60:
        return 60
    elif x < 80:
        return 80
    elif x < 100:
        return 100
    elif x < 120:
        return 120
    elif x < 140:
        return 140
    elif x < 160:
        return 160
    else:
        return 180
    
def kategori_cltv(x):
    if x < 1000:
        return '0 jt ~ 1 jt'
    elif x < 2000:
        return '1 jt ~ 2 jt'
    elif x < 3000:
        return '2 jt ~ 3 jt'
    elif x < 4000:
        return '3 jt ~ 4 jt'
    elif x < 5000:
        return '4 jt ~ 5 jt'
    elif x < 6000:
        return '5 jt ~ 6 jt'
    elif x < 7000:
        return '6 jt ~ 7 jt'
    elif x < 8000:
        return '7 jt ~ 8 jt'
    else:
        return '> 8jt '

kategori_tenure =  {
        0: "Less than 1 year",
        1: "1-2 years",
        2: "2-3 years",
        3: "3-4 years",
        4: "4-5 years",
        5: "5-6 years",
        6: "6-7 years"
    }

def pie_churn():
	pie = df[['Churn Label', 'Customer ID']].groupby('Churn Label').count().reset_index()
	pie.columns = ['Churn Label', 'Percentage']
	pie['Percentage'] = round(pie['Percentage']/len(df), 3)
	fig = px.pie(values=pie['Percentage'], names=pie['Churn Label'],
	            title="<b>Churn Vs Not Churn</b>")
	fig.update_layout(legend=dict(orientation="h"), 
	                  height=250, width=250)
	fig.update_layout(legend=dict(orientation="h"), margin=dict(l=0, r=0, t=30, b=0), height=250)
	fig.update_traces(textposition='inside', textinfo='percent+label')

	return fig

def device_class():
	dc = churn.groupby('Device Class')['Customer ID'].count().sort_values(ascending=True)
	fig = px.bar(dc, x='Customer ID', text_auto=True)
	fig.update_layout(title=dict(text='<b>Jumlah Pengguna Churn berdasarkan Device Class'), height=250, width=400,
	                 plot_bgcolor="#fff", xaxis=dict(title=" "), yaxis=dict(title=" "),
	                 margin=dict(l=0, r=0, t=30, b=0))
	fig.update_traces(textfont_size=12, textposition="outside")
	fig.update_xaxes(range=[0, 1500])

	return fig

def category_product():
    
	churn['game'] = churn['Games Product'].apply(cek)
	churn['music'] = churn['Music Product'].apply(cek)
	churn['education'] = churn['Education Product'].apply(cek)
	churn['video'] = churn['Video Product'].apply(cek)

	product_x = ['game', 'music', 'education', 'video']
	product_y = [churn['game'].sum(), churn['music'].sum(), churn['education'].sum(), churn['video'].sum()]
	fig = px.bar(y=product_x, x=product_y, text_auto=True,
	             title='<b>Jumlah Pengguna Churn berdasarkan Kategori Produk</b>')
	fig.update_traces(textfont_size=12, textposition="outside")
	# fig.update_xaxes(range=[0, 1000])
	fig.update_layout(plot_bgcolor='#fff', xaxis=dict(title=''), yaxis=dict(title=''),
                  height=250, width=400,margin=dict(l=0, r=0, t=30, b=0),)	
	return fig

def tenure_churn():
	churn['kategori_tenure'] = churn['Tenure Months'].apply(lambda x: x//12)
	churn['kategori_tenure'] = churn['kategori_tenure'].map(kategori_tenure)
	tenure = churn.groupby(['kategori_tenure'])['Customer ID'].count().sort_values(ascending=False)
	fig = px.bar(x = tenure.index, y = tenure.values, text_auto=True,
	             title='<b>Jumlah Pengguna Churn berdasarkan Tenure</b>')
	fig.update_traces(textfont_size=13, textposition="outside")
	fig.update_layout(plot_bgcolor='#fff', xaxis=dict(title=''), yaxis=dict(title=''),)

	return fig

def monthly_purchase_churn():
	churn['kategori_monthly'] = churn['Monthly Purchase (Thou. IDR)'].apply(kategori_monthly)
	y_monthly = churn.groupby(['kategori_monthly'])['Customer ID'].count().sort_index().values
	x_monthly = np.array(['20k ~ 40k', '40k ~ 60k', '60k ~ 80k', '80k ~ 100k', '100k ~ 120k', '120k ~ 140k', '140k ~ 160k'])

	fig = px.bar(y=y_monthly, x=x_monthly, text_auto=True, title='<b>Jumlah Pengguna Churn berdasarkan Pembayaran tiap Bulan</b>')

	fig.update_traces(textfont_size=12, textposition="outside")
	fig.update_layout(plot_bgcolor='#ffffff', xaxis=dict(title=''), yaxis=dict(title=''))

	return fig

def cltv_churn():
	churn['kategori_cltv'] = churn['CLTV (Predicted Thou. IDR)'].apply(kategori_cltv)
	result = churn.groupby(['kategori_cltv'])['Customer ID'].count().sort_index()
	fig = px.bar(x=result.index, y=result.values, text_auto=True,
	             title='<b>Jumlah Pengguna Churn berdasarkan CLTV</b>')

	fig.update_traces(textfont_size=12, textposition="outside")
	fig.update_layout(plot_bgcolor='#fff')

	return fig

def preprocessing(tabel):
	map_label = { 'Yes' : 1,'No' : 0}
	numerik = ['Tenure Months', 'Monthly Purchase (Thou. IDR)']
	kategorik = ['Location', 'Device Class', 'Games Product','Music Product', 'Education Product',
				'Call Center', 'Video Product','Use MyApp', 'Payment Method']

	df_clean = df.drop(['Customer ID', 'Longitude', 'Latitude', 'CLTV (Predicted Thou. IDR)'],axis=1)
	df_clean['Churn Label'] = df_clean['Churn Label'].map(map_label)

	x = df_clean.drop(['Churn Label'], axis=1)
	y = df_clean['Churn Label']
				
	ohe = OneHotEncoder().fit(x[kategorik])
	scaler = MinMaxScaler().fit(x[numerik])

	tabel[numerik] = scaler.transform(tabel[numerik])
	hasil_ohe = ohe.transform(tabel[kategorik])
	df_ohe = pd.DataFrame(hasil_ohe.toarray(), columns=ohe.get_feature_names_out())
	x_full = pd.concat([tabel[numerik], df_ohe], axis=1)

	return x_full