import streamlit as st
import pandas as pd
from utils import (
    dataset_description, 
    pie,
    churn_rate, 
    cum_monthly_purchase, 
    cum_cltv, 
    tenure_months_distribution,
    payment_method,
    cols, plot_distribution,
    products_figure, chi2_test, cramers_v,
    cltv_hypothesis_testing
)
from churn import (
    pie_churn,
    device_class,
    category_product,
    tenure_churn,
    monthly_purchase_churn,
    cltv_churn,
    preprocessing
)
import customer_segmentation as cs
from PIL import Image
import association_rules_util as ar
import plotly.express as px
import joblib
import psycopg2

st.set_page_config(layout="wide")

# conn_string = "postgresql://josebenaya:PWo3ZhTkEA1a@ep-white-hall-00289384.us-east-2.aws.neon.tech/web"
# engine = sqlalchemy.create_engine(conn_string)

# Connect to your PostgreSQL database
# from sqlalchemy import create_engine

# Set up SQLAlchemy engine
conn = psycopg2.connect(
    dbname="web",
    options="endpoint=ep-white-hall-00289384",
    user="josebenaya",
    password="PWo3ZhTkEA1a",
    host="ep-white-hall-00289384.us-east-2.aws.neon.tech",
    port="5432",
)
# conn_string = "postgresql://josebenaya:************@ep-white-hall-00289384.us-east-2.aws.neon.tech/web"
# engine = sqlalchemy.create_engine(conn_string)


# Read data from PostgreSQL into DataFrames
df_query = "SELECT * FROM telco_customer_churn_adapted_v2"  # Replace with your actual query
segment_df_query = "SELECT * FROM segmentation_lengkap"  # Replace with your actual query

df = pd.read_sql(df_query, conn)
segment_df = pd.read_sql(segment_df_query, conn)
# df = pd.read_excel("./data/Telco_customer_churn_adapted_v2.xlsx")
# segment_df = pd.read_excel("data/segmentation_lengkap.xlsx")

st.title("📊 Customer Behavior Analysis 📈")
st.markdown("Selamat datang di Dashboard Customer Behavior Analysis 🚀 Dibangun untuk memberikan gambaran menyeluruh tentang perilaku pelanggan, dashboard ini memiliki fitur-fitur keren seperti **Analisis Penggunaan Produk, Analisis Cross-Selling dengan Association Rule, Segmentasi Pelanggan, dan Analisis Churn Pelanggan**. Mulai dari cara pelanggan berinteraksi dengan layanan kami hingga temuan menarik melalui aturan asosiasi, kami siap membantu perusahaan **mengoptimalkan strategi pemasaran, meningkatkan pendapatan, dan mengurangi churn**. Selamat mengeksplorasi dunia wawasan yang menginspirasi! 🌐✨")

tab1, tab2, tab3, tab4 = st.tabs(["Dashboard", "Dataset Overview", "Update", "Delete"])

# Dashboard
with tab1:
    st.header("General Statistics")

    # Metrics
    with st.container():
        col1, col2, col3, col4 = st.columns([0.2, 0.2, 0.3, 0.3])
        col1.metric("**Num of Customer**", f"{len(df['Customer ID'].unique())}", help="Lorem Ipsum")
        col2.metric("**Churn Rate**", f"{churn_rate}%", help="Lorem Ipsum")
        col3.metric("**Cumulative Monthly Purchase (Thou.)**", f"IDR {cum_monthly_purchase:,}")
        col4.metric("**Cumulative CLTV (Thou.)**", f"IDR {cum_cltv:,}")

    st.divider()

    # Jumlah Customer dan Tenure
    with st.container():
        col1, col2 = st.columns([0.4, 0.6])

        with col1:
            col1_1, col1_2 = st.columns([0.4, 0.3])

            with col1_2:
                names = st.radio("Select Names", options=["Location", "Device Class"],index=0)

            with col1_1:
                fig = pie(df, names = names, hover_variable = "Monthly Purchase (Thou. IDR)")
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            col2_1, col2_2 = st.columns([0.6, 0.3])
            
            with col2_2:
                x = st.radio("Select x-axis", options=["Count", "Monthly Purchase (Thou.)", "CLTV (Thou.)"],index=0)

                if x != "Count":
                    agg = st.radio("Select Aggregation", options=["Mean", "Median", "Sum"],index=0)

            with col2_1:
                if x != "Count":
                    fig = tenure_months_distribution(x, agg)
                else:
                    fig = tenure_months_distribution(x)
                st.plotly_chart(fig, use_container_width=True)
                
    st.divider()

    # Payment dan Distribution
    with st.container():
        col1, col2, col3 = st.columns([0.45, 0.4, 0.05])

        with col1:
            col1_1, col1_2 = st.columns([0.65, 0.35])
            
            with col1_2:
                x1 = st.radio("Select y-axis", options=["Count", "Monthly Purchase (Thou.)", "CLTV (Thou.)"], index=0)
                hue = st.radio("Select hue", options=["None", "Location", "Device Class"], index=0)

                if x1 != "Count":
                    agg1 = st.radio("Select Aggregation", options=["Sum", "Mean", "Median"], index=0)

            with col1_1:
                if x1 != "Count" and hue:
                    fig = payment_method(df, y=x1, agg=agg1, hue=hue)
                elif x1 != "Count" and not hue:
                    fig = payment_method(df, y=x1, agg=agg1)
                elif x1 == "Count" and hue:
                    fig = payment_method(df, y=x1, hue=hue)
                else:
                    fig = payment_method(df, y=x1)
                st.plotly_chart(fig, use_container_width=True)          


        with col2:

            option = st.selectbox(
                '**Distribution Chart**',
                cols, index=0)
            fig = plot_distribution(option)
            st.plotly_chart(fig, use_container_width=True)

# Dataset Overview
with tab2:
    st.subheader("Data Examples")
    st.write(df.head())
    st.write("**Variables**")
    st.write(dataset_description)

st.success("""**Insights:**
1. Kita mengetahui jika customer terbesar berasal dari kota Jakarta dengan persentase 71.4%, disusul dengan Kota Bandung dengan 28.6%.
2. Jika dilihat dari Tenure, pelanggan terbanyak ada berada tahun pertama, dan diikuti tahun-tahun selanjutnya. 
3. Kita mengetahui bahwasanya, metode payment paling digemari yaitu dengan metode pulsa, lalu diikuti metode digital wallet. 
4. Dari distribusi CLTV predicted, kita bisa mengetahui bahwa customer paling banyak diprediksi disekitar nilai 7000 atau 7 jutaan.
""")

for i in range(3):
    st.write("")


tab1, tab2, tab3, tab4 = st.tabs(["Product Usage Analysis", "Association Rule", "Customer Segmentation", "Customer Churn Analysis"])

# Product Usage Analysis
with tab1:
    st.header("📊 Product Usage Analysis 🚀")
    st.markdown("Dalam section ini, kita memahami perilaku pelanggan dalam menggunakan produk dan dampaknya terhadap keputusan churn serta nilai CLTV. Analisis mendalam ini memungkinkan perusahaan untuk mengoptimalkan performa produk, menyusun strategi yang cermat untuk mengurangi churn, dan meningkatkan nilai CLTV melalui produk yang digunakan. Data-driven insights membuka peluang baru untuk meningkatkan kualitas layanan dan pengalaman pelanggan.")
    with st.expander("❓ Methods"):
        st.markdown("""Dalam menganalisis penggunaan produk, kita menggunakan dua jenis uji statistik yang berbeda untuk variabel dependen yang berbeda. Pertama, untuk variabel dependen "Churn Label," kita menerapkan uji Chi-square (χ²). Hipotesisnya adalah bahwa tidak ada hubungan antara penggunaan produk dan keputusan pelanggan untuk berhenti berlangganan (churn). Uji ini bertujuan untuk mengidentifikasi apakah ada asosiasi yang signifikan antara variabel ini, memberikan wawasan tentang produk yang mungkin mempengaruhi keputusan churn.

Kedua, untuk variabel dependen "CLTV," kita menggunakan uji Kruskal-Wallis. Hipotesisnya adalah bahwa tidak ada perbedaan yang signifikan dalam nilai CLTV antara kelompok pengguna produk yang berbeda. Uji ini bertujuan untuk menentukan apakah terdapat perbedaan yang signifikan dalam Customer Lifetime Value (CLTV) antara kelompok-kelompok berdasarkan penggunaan produk, membantu perusahaan memahami dampak produk terhadap nilai jangka panjang pelanggan.

Dengan menggunakan kombinasi uji ini, kita dapat mengidentifikasi pengaruh penggunaan produk pada keputusan churn dan nilai CLTV, membimbing strategi perusahaan untuk meningkatkan retensi pelanggan dan nilai tambah jangka panjang. 📈✨
""")
    
    dependent = st.selectbox("Select Dependent Variable", ["Churn Label", "CLTV (Predicted Thou. IDR)"])

    products = st.multiselect(
        'Select Products',
        ["Games Product", "Music Product", "Education Product", "Video Product", "Call Center", "Use MyApp"],
        ["Games Product", "Music Product", "Education Product"])

    # percentage = st.toggle("Use Percentage")
    percentage=False

    # with st.expander("Help"):
    #     st.write("**Chi2 Test**\nLorem ipsum dolor sit amet, consectetur adipiscing elit. Nullam viverra justo nec metus hendrerit, in vestibulum augue pellentesque. Sed euismod ante et justo varius, vel feugiat nisl lacinia. In hac habitasse platea dictumst. Fusce ullamcorper, risus eget facilisis scelerisque, libero metus condimentum nulla, vel euismod est odio nec est. Nulla id augue ac metus dictum accumsan.")
    
    kesimpulan_cramers_v = [
        "Small Effect: Jika nilai Cramer's V mendekati 0.1, ini menunjukkan bahwa hubungan antara penggunaan layanan game dan keputusan churn memiliki efek kecil. Meskipun ada hubungan statistik, pengaruh praktisnya mungkin tidak begitu kuat.",
        "Medium Effect: Jika nilai Cramer's V berada di kisaran 0.3, ini menunjukkan bahwa hubungan memiliki efek menengah. Penggunaan layanan game mungkin memiliki pengaruh yang cukup terlihat terhadap keputusan churn, dan strategi bisnis dapat dipertimbangkan untuk mengatasi hal ini.",
        "High Effect: Jika nilai Cramer's V mendekati atau melebihi 0.5, ini menunjukkan bahwa hubungan antara penggunaan layanan game dan keputusan churn memiliki efek besar. Ini adalah sinyal yang kuat bahwa penggunaan layanan game secara substansial mempengaruhi keputusan pelanggan untuk churn, dan langkah-langkah bisnis lebih lanjut mungkin diperlukan untuk memitigasi churn di antara pengguna layanan game."
    ]

    if products:
        if dependent == "Churn Label":
            for i in range(len(products)):
                st.divider()

                col1, col2, col3 = st.columns([0.4, 0.05, 0.55])

                with col1:
                    st.write(f"**Jumlah Customer Churn dan Tidak Churn berdasarkan {products[i]}**")
                    fig = products_figure(products[i], percentage, "Churn Label")
                    st.plotly_chart(fig, use_container_width=True)

                with col3:
                    with st.expander("**Hypothesis Test**", expanded=False):
                        test = chi2_test(products[i])
                        st.write(test)

                    with st.expander("**Effect size in the Chi-square test**", expanded=False):
                        cramers_v_output = cramers_v(products[i])
                        st.write(cramers_v_output)

                        if "Small" in cramers_v_output["Effect Size"].iloc[0]:
                            kesimpulan_cv = kesimpulan_cramers_v[0]
                        elif "Medium" in cramers_v_output["Effect Size"].iloc[0]:
                            kesimpulan_cv = kesimpulan_cramers_v[1]
                        else:
                            kesimpulan_cv = kesimpulan_cramers_v[2]


                    st.success(f"""**Insights:**\n\n
1. Berdasarkan analisis statistik, terdapat perbedaan yang signifikan dalam hal keputusan churn di antara pelanggan yang menggunakan {products[i]} dan yang tidak menggunakan {products[i]}.
2. {kesimpulan_cv}
3. Pemahaman ini dapat membantu kita menyesuaikan strategi retensi pelanggan. Misalnya, jika pelanggan yang menggunakan {products[i]} lebih cenderung churn, mungkin perlu dipertimbangkan peningkatan nilai atau penawaran eksklusif untuk mempertahankan mereka.""")


        elif dependent == "CLTV (Predicted Thou. IDR)":
            for i in range(len(products)):
                st.divider()

                col1, col2, col3 = st.columns([0.4, 0.05, 0.55])

                with col1:
                    st.write(f"**Jumlah Customer Churn dan Tidak Churn berdasarkan {products[i]}**")
                    fig = products_figure(products[i], percentage, "Churn Label")
                    st.plotly_chart(fig, use_container_width=True)

                with col3:
                    assumptions, statistic, pvalue, posthoc_result, conclusion = cltv_hypothesis_testing(products[i])
                    # with st.expander("**Assumptions for ANOVA**", expanded=True):
                    #     st.write(assumptions)

                    if '❌' in assumptions.iloc[0].values:
                        label = "Kruskal-Wallis Test"
                        result = pd.DataFrame({
                            "Kruskal Statistic" : [statistic],
                            "P-Value" : [pvalue],
                            "Conclusion" : [conclusion]
                        })
                    else:
                        label = "ANOVA One-way"
                        result = pd.DataFrame({
                            "F Statistic" : [statistic],
                            "P-Value" : [pvalue],
                            "Conclusion" : [conclusion]
                        })
                    with st.expander(f"**{label}**", expanded=True):
                        st.write(result)

                    st.success(f"""**Insights:**
1. Hasil ini dapat diartikan bahwa penggunaan {products[i]} dapat mempengaruhi nilai CLTV pelanggan secara signifikan.
2. Implikasinya adalah perusahaan dapat mengoptimalkan strategi bisnis dengan fokus pada pengembangan layanan yang memiliki dampak positif terhadap nilai CLTV. Meningkatkan nilai CLTV dapat mencakup up-selling, cross-selling, atau meningkatkan retensi pelanggan.
""")


    else:
        st.warning("Silahkan pilih produk yang ingin dianalisis")


# Cross-selling Analysis
with tab2:
    st.header("🔗 Association Rule Analysis 🔄")
    st.markdown("Section ini fokus pada analisis cross-selling, mengungkap hubungan antar produk dan kombinasi yang potensial. Informasi yang dihasilkan dapat digunakan untuk meningkatkan performa produk melalui promosi yang tepat dan pembuatan paket bundling. Sebuah langkah inovatif untuk memaksimalkan potensi pendapatan dan memberikan nilai tambah kepada pelanggan.")
    with st.expander("❓ Methods"):
        st.markdown("""Pada bagian analisis asosiasi, kita menerapkan teknik Association Rule Mining untuk merinci hubungan antar produk dan menganalisis peluang cross-selling. Minimum support yang diatur pada 0.15 digunakan untuk mengidentifikasi kombinasi produk yang signifikan. Misalnya, jika kita memiliki aturan seperti:

- **Antecedents (Education Products) dan Consequents (Call Center):**
  - **Support (0.17166):** Menunjukkan seberapa sering kombinasi ini muncul dalam dataset.
  - **Confidence (0.499174):** Mencerminkan seberapa sering pelanggan yang menggunakan produk pendidikan juga menggunakan layanan Call Center.
  - **Lift (1.720002):** Menunjukkan seberapa besar kemungkinan pelanggan akan menggunakan Call Center setelah menggunakan produk pendidikan, dibandingkan jika kedua kejadian itu tidak terkait.

**Cara Membaca Rule untuk Cross-Selling Analysis:**
Contoh ini mengindikasikan bahwa pelanggan yang menggunakan produk pendidikan memiliki kemungkinan sekitar 50% untuk menggunakan layanan Call Center. Dengan lift lebih dari 1, hal ini menunjukkan bahwa peluang cross-selling antara produk pendidikan dan layanan Call Center lebih tinggi daripada kebetulan. Analisis ini memberikan wawasan berharga untuk merancang strategi cross-selling yang lebih efektif, meningkatkan penjualan lintas produk, dan memberikan pengalaman pelanggan yang lebih baik. 🔄📊
""")

    st.subheader("Top 10 Rules")
    rules = pd.read_csv("data/rules.csv")

    # Menerapkan fungsi pada kolom "antecedents" dan "consequents"
    rules['antecedents'] = rules['antecedents'].apply(lambda x: ar.frozenset_to_string(eval(x)))
    rules['consequents'] = rules['consequents'].apply(lambda x: ar.frozenset_to_string(eval(x)))

    st.table(rules)

    st.success("""**Rekomendasi:**
1. **Maksimalkan Penggunaan MyApp:**\n
    Implementasikan program insentif khusus untuk pelanggan yang sudah menggunakan layanan Video Product dan Education Product agar mereka lebih aktif menggunakan MyApp. Misalnya, penawaran diskon eksklusif atau akses ke konten premium melalui MyApp.
2. **Optimalkan Layanan Video dan Edukasi:**\n
    Perluas dan tingkatkan konten eksklusif pada layanan Video Product dan Education Product. Sertakan fitur yang memikat dan dapat menjadi pembeda dari layanan pesaing. Lakukan promosi yang lebih agresif untuk meningkatkan kesadaran pelanggan terhadap manfaat unik yang ditawarkan oleh kedua layanan ini.
3. **Perkuat Hubungan melalui Call Center:**\n
    Tingkatkan pelatihan dan kapabilitas agen Call Center untuk memberikan solusi yang lebih efektif dan cepat. Sertakan inisiatif penghargaan atau program pengenalan bagi agen yang memberikan layanan pelanggan yang luar biasa. Selain itu, pertimbangkan penggunaan teknologi canggih seperti chatbot untuk meningkatkan efisiensi dalam menangani permintaan pelanggan.
""")
    
    st.divider()
    st.header("Interactive Tools")

    col1, col2 = st.columns([0.25, 0.75])

    with col1:
        st.markdown("**Set the Parameters**")
        products = st.multiselect(
            'Select Products ',
            ["Games Product", "Music Product", "Education Product", "Video Product", "Call Center", "Use MyApp"],
            ["Games Product", "Music Product", "Education Product", "Video Product", "Call Center", "Use MyApp"])

        min_support = st.slider("Min Support", min_value=1e-5, max_value=1.0,
                                value=0.15, step=0.001)
        
        min_confidence = st.slider("Min Confidence", min_value=1e-5, max_value=1.0,
                                value=0.65, step=0.001)
        
    with col2:
        st.markdown("**Top 10 Rules**")
        sort_key = st.selectbox("Sort Values", ["Confidence", "Lift", "Support"]).lower()        
        df_ar = ar.index(products, min_support, min_confidence).head(10)
        output = df_ar[["antecedents", "consequents","support", "confidence", "lift"]].sort_values(sort_key, ascending=False).head(10)
        st.table(output)
    

# Customer Segmentation
with tab3:
    st.header('🎯 Customer Segmentation 🎭')
    st.markdown("Dengan menggunakan dua model - Value-based segmentation dan Needs-based segmentation - section ini memanfaatkan faktor seperti monthly-purchase, CLTV, tenure months, dan variabel produk untuk mengidentifikasi segmen pelanggan. Metode clustering dengan algoritma K-Means memberikan pemahaman mendalam tentang preferensi dan kebutuhan pelanggan, membimbing strategi pemasaran yang terfokus.")

    model = st.selectbox(
        'Select Segmentation Model',
        ["Value-based Segmentation", "Needs-based Segmentation"], index=0)
    
    filter = st.selectbox(
        'Filter',
        ["Show All", 'Churners', 'Non Churners']
    )

    model = "Value" if "Value" in model else "Needs"

    if filter == "Show All":
        data_segment = segment_df[segment_df["Model"] == model]
    elif filter == "Churners":
        data_segment = segment_df[(segment_df["Model"] == model) & (segment_df["Churn Label"] == "Yes")]
    else:
        data_segment = segment_df[(segment_df["Model"] == model) & (segment_df["Churn Label"] == "No")]
    
    
    # Read the data
    df = pd.read_excel("data/Telco_customer_churn_adapted_v2.xlsx")
    if model == "Needs":
        columns = ['Games Product', 'Music Product', 'Education Product', 'Video Product']
        X = df.copy()[columns]
        churners = X.loc[df[df["Churn Label"] == "Yes"].index]
        non_churners = X.loc[df[df["Churn Label"] == "No"].index]

        churners = churners[churners["Video Product"] != 'No internet service']
        non_churners = non_churners[non_churners["Video Product"] != 'No internet service']

        ###
        for col in columns:
            non_churners[col] = non_churners[col].replace({"Yes": 1, "No":0})
            churners[col] = churners[col].replace({"Yes": 1, "No":0})
        
        if filter == "Show All":
            resulting_figures1 = cs.visualize_needs_based_clustering(churners, k=2)
            resulting_figures2 = cs.visualize_needs_based_clustering(non_churners, k=3)

            with st.expander("📊 Clustering Result Visualization"):
                st.subheader("Churners Needs-based Segmentation")
                for fig in resulting_figures1:
                    st.pyplot(fig)
                st.divider()
                st.subheader("Non-Churners Needs-based Segmentation")
                for fig in resulting_figures2:
                    st.pyplot(fig)
        elif filter == "Churners":
            resulting_figures1 = cs.visualize_needs_based_clustering(churners, k=2)
            with st.expander("📊 Clustering Result Visualization"):
                st.subheader("Churners Needs-based Segmentation")
                for fig in resulting_figures1:
                    st.pyplot(fig)
        else:
            resulting_figures2 = cs.visualize_needs_based_clustering(non_churners, k=3)
            with st.expander("📊 Clustering Result Visualization"):
                st.subheader("Non-Churners Needs-based Segmentation")
                for fig in resulting_figures2:
                    st.pyplot(fig)
    
    else: # VALUE BASED
        columns = ["Tenure Months", "Monthly Purchase (Thou. IDR)", "CLTV (Predicted Thou. IDR)"]
        X = df.copy()[columns]
        churners = X.loc[df[df["Churn Label"] == "Yes"].index]
        non_churners = X.loc[df[df["Churn Label"] == "No"].index]

        if filter == "Show All":
            fig1 = cs.visualize_value_based_clustering(churners, k=3)
            fig2 = cs.visualize_value_based_clustering(non_churners, k=3)

            with st.expander("📊 Clustering Result Visualization"):
                st.subheader("Churners Value-based Segmentation")
                st.pyplot(fig1)
                st.divider()
                st.subheader("Non-Churners Value-based Segmentation")
                st.pyplot(fig2)
        elif filter == "Churners":
            fig1 = cs.visualize_value_based_clustering(churners, k=3)
            with st.expander("📊 Clustering Result Visualization"):
                st.subheader("Churners Value-based Segmentation")
                st.pyplot(fig1)
        else:
            fig2 = cs.visualize_value_based_clustering(non_churners, k=3)
            with st.expander("📊 Clustering Result Visualization"):
                st.subheader("Non-Churners Value-based Segmentation")
                st.pyplot(fig2)



    for i in range(2):
        st.write("")

    img_idx = pd.Series(range(1, 12)).sample(len(data_segment)).values

    col1, col2, col3 = st.columns(3)

    with col1:
        for i in range(0, len(data_segment), 3):
            segment = data_segment["Segment"].iloc[i]
            karakter = data_segment["Characteristics"].iloc[i]
            marketing = data_segment["Marketing"].iloc[i]
            st.image(Image.open(f"images/{img_idx[i]}.png"))
            st.success(f"""**{segment}**\n\n**Karakteristik**\n\n{karakter}\n\n**Rekomendasi Strategi Pemasaran**\n\n{marketing}""")
    
    with col2:
        for i in range(1, len(data_segment), 3):
            segment = data_segment["Segment"].iloc[i]
            karakter = data_segment["Characteristics"].iloc[i]
            marketing = data_segment["Marketing"].iloc[i]
            st.image(Image.open(f"images/{img_idx[i]}.png"))
            st.warning(f"""**{segment}**\n\n**Karakteristik**\n\n{karakter}\n\n**Rekomendasi Strategi Pemasaran**\n\n{marketing}""")

    with col3:
        for i in range(2, len(data_segment), 3):
            segment = data_segment["Segment"].iloc[i]
            karakter = data_segment["Characteristics"].iloc[i]
            marketing = data_segment["Marketing"].iloc[i]
            st.image(Image.open(f"images/{img_idx[i]}.png"))
            st.info(f"""**{segment}**\n\n**Karakteristik**\n\n{karakter}\n\n**Rekomendasi Strategi Pemasaran**\n\n{marketing}""")


# Churn Analysis
with tab4:
    with st.container():
        st.header('🔄 Customer Churn Analysis 🕵‍♂')
        st.markdown("Mengungkap alasan di balik keputusan churn, section ini menampilkan visualisasi karakteristik pelanggan yang churn dengan berbagai faktor. Selain itu, fitur interaktif memungkinkan prediksi churn, memberikan kesempatan proaktif untuk menjaga kepuasan pelanggan dan mengurangi tingkat churn. Sebuah langkah cerdas untuk mempertahankan basis pelanggan yang stabil.")
        st.divider()

        col1, col2, col3= st.columns([0.2 , 0.3, 0.3])

        with col1:
            fig = pie_churn()
            st.plotly_chart(fig)

        with col2:
            fig = device_class()
            st.plotly_chart(fig)

        with col3:
            fig = category_product()
            st.plotly_chart(fig)

    st.divider()
    with st.container():  

        col1, col2 = st.columns([0.25, 0.75])

        with col1:
            var = st.selectbox("Select variabel", 
                ['Monthly Purchase (Thou. IDR)', "CLTV (Predicted Thou. IDR)", 'Tenure Months'])

        with col2:
            if var == 'CLTV (Predicted Thou. IDR)':
                fig = cltv_churn()
            elif var == 'Monthly Purchase (Thou. IDR)':
                fig = monthly_purchase_churn()
            elif var == 'Tenure Months':
                fig = tenure_churn()

            st.plotly_chart(fig)

    st.success("""**Insight**:
1. Kita bisa mengetahui bahwasanya persentase churn rate sekitar 26.5%
2. Jika dilihat dari Device, kita bisa menyimpulkan bahwa customer yang churn yaitu customer yang menggunakan device high end
3. Jika dilihat dari product, customer yang mengalami churn ada pada mereka yang berlangganan video.
5. Customer yang memiliki churn paling tinggi ada pada mereka yang masa berlangganannya kurang dari satu tahun.
6. Mayoritas customer yang churn adalah mereka yang mengeluarkan biaya berlangganan bulanan dari 80,0000 - 140,000.
""")

    st.divider()
    with st.container():
        st.header('Predictive Modeling')
        st.markdown("""
Dalam predictive modelling ini, model yang telah dikembangkan akan mengevaluasi karakteristik data yang 
diinputkan untuk mengklasifikasikan apakah seorang pelanggan cenderung melakukan churn atau tidak. Pengguna dapat memanfaatkan analisis prediktif ini untuk mengidentifikasi faktor-faktor yang mempengaruhi 
kecenderungan churn dan mengambil langkah-langkah proaktif dalam mempertahankan basis pelanggan.
        """)
        with st.expander("❓ Methods"):
            st.markdown("""
Kami menggunakan algoritma Random Forest Classifier dimana algoritma ini memiliki beberapa keunggulan diantaranya sebagai berikut:
1. Model cenderung stabil.
2. Model juga cenderung memiliki akurasi yang tinggi.
3. Memiliki skalabilitas yang baik dimana waktu komputasi yang digunakan tergolong sedikit atau cepat.
            """)
    # Membagi layout menjadi 2 kolom
    col1, col2, col3 = st.columns([0.3, 0.3, 0.4])

    # Widget input untuk setiap kolom
    with col1:
        customer_id = st.number_input("Customer ID:")
        tenure = st.number_input('Tenure Months:')
        lokasi = st.selectbox('Select Location', ['Jakarta', 'Bandung'])
        device = st.selectbox('Select Device Class', ['Mid End', 'High End'])

    with col2:
        games = st.selectbox('Select Game', ['Yes', 'No', 'No internet service'])
        music = st.selectbox('Select Music', ['Yes', 'No', 'No internet service'])
        education = st.selectbox('Select Education', ['Yes', 'No', 'No internet service'])
        call = st.selectbox('Select Call', ['Yes', 'No', 'No internet service'])

    with col3:
        video = st.selectbox('Select Video', ['Yes', 'No', 'No internet service'])
        app = st.selectbox('Select App', ['Yes', 'No', 'No internet service'])
        payment = st.selectbox('Payment Method', ['Digital Wallet', 'Pulsa', 'Debit', 'Credit'])
        monthly = st.number_input('Monthly Purchase')

    # Menambahkan tombol "Predict!"
    if st.button("Predict!"):
        # Membuat DataFrame dari input
        data_input = {
            'Customer ID': [customer_id],
            'Tenure Months': [tenure],
            'Location': [lokasi],
            'Device Class': [device],
            'Games Product': [games],
            'Music Product': [music],
            'Education Product': [education],
            'Call Center': [call],
            'Video Product': [video],
            'Use MyApp': [app],
            'Payment Method': [payment],
            'Monthly Purchase (Thou. IDR)': [monthly]
        }

        # Menerapkan preprocessing
        table = pd.DataFrame(data_input)
        table = preprocessing(table)

        # Memuat model
        model = joblib.load('model.joblib')

        # Melakukan prediksi
        result = model.predict(table)

        jawaban = None
        if result == 0:
            st.error(f"""
Hasil prediksi : Customer {customer_id} diprediksi akan **Churn**
""")
        else:
            st.success(f"""
Hasil prediksi : Customer {customer_id} diprediksi **tidak akan Churn**
""")
        
        st.write(" ")
        st.write("#### Silahkan upload file.excel jika ingin melakukan prediksi secara kolektif")
        file_upload = st.file_uploader("Pilih file.excel", type=["xlsx"])

        if(file_upload != None):
            awal = pd.read_excel(file_upload)
            file_df = preprocessing(awal)
            y_pred = model.predict(file_df)
            awal = pd.read_excel(file_upload)
            awal['Churn Label predicted'] = y_pred
            awal['Churn Label predicted'] = awal['Churn Label predicted'].map({1:"Churn", 0:"Not Churn"})

            st.write(awal)
