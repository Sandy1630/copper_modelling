import streamlit as st
import re
import numpy as np
from sklearn.preprocessing import power_transform
from streamlit_option_menu import option_menu
import pandas as pd
from sklearn.metrics import r2_score,accuracy_score,f1_score,roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
import warnings

warnings.filterwarnings("ignore")

with st.sidebar:
    selected=option_menu("Home",["Home",'selling_price',"status","About"])
country_option=[ 28.,  25.,  30.,  32.,  38.,  78.,  27.,  77., 113.,  79.,  26.,
        39.,  40.,  84.,  80., 107.,  89.]
application_option=[10., 41., 28., 59., 15.,  4., 38., 56., 42., 26., 27., 19., 20.,
       66., 29., 22., 40., 25., 67., 79.,  3., 99.,  2.,  5., 39., 69.,
       70., 65., 58., 68.]
product_option=[1670798778, 1668701718,     628377,     640665,     611993,
       1668701376,  164141591, 1671863738, 1332077137,     640405,
       1693867550, 1665572374, 1282007633, 1668701698,     628117,
       1690738206,     628112,     640400, 1671876026,  164336407,
        164337175, 1668701725, 1665572032,     611728, 1721130331,
       1693867563,     611733, 1690738219, 1722207579,  929423819,
       1665584320, 1665584662, 1665584642]
status_option=['Won', 'Draft', 'To be approved', 'Lost', 'Not lost for AM',
       'Wonderful', 'Revised', 'Offered', 'Offerable']
item_option=['W', 'WI', 'S', 'Others', 'PL', 'IPL', 'SLAWR']

if selected=="Home":
    st.title("Industrial Copper Modeling")
    
    st.markdown("Welcome to our data analytics and lead management solution for the copper industry.")
    
    
    st.write("""Our Mission:
At the core of our mission is to revolutionize how the copper industry manages sales, pricing, and lead acquisition. 

We are committed to:""")
    
    st.markdown("""1.Enhancing Pricing Precision: Our state-of-the-art machine learning regression model is dedicated to tackling complex sales and pricing data. By utilizing advanced techniques such as data normalization, feature scaling, and outlier detection, we empower our industry experts to make data-driven pricing decisions with unrivaled accuracy and efficiency.""")
    
    st.markdown("""2.Streamlining Lead Management: Our lead classification model is the linchpin in our lead management strategy. It evaluates and classifies leads based on their potential to become valued customers. With a meticulous focus on "WON" and "LOST" statuses, we ensure that every lead is handled with the utmost efficiency and precision.""")

if selected=="About":
    st.title("About")
    st.write(
    "This Streamlit app is designed to address challenges in the copper industry by providing solutions for data analytics and lead management. "
    "Enter your data to classify leads and optimize pricing decisions.")
    
    st.write("""**Copper Modeling** is a cutting-edge data-driven solution tailored to meet the unique challenges faced by the copper industry. We empower businesses in the copper sector to harness the power of advanced analytics and machine learning for enhanced decision-making, improved pricing strategies, and streamlined lead management.

 Our Mission

At Copper Modeling, our mission is to revolutionize how the copper industry leverages data to drive growth and efficiency.
We are committed to:

 **Precision in Pricing**: We utilize state-of-the-art machine learning techniques to analyze sales and pricing data, ensuring that pricing decisions are made with unparalleled accuracy. Say goodbye to manual guesswork and hello to data-driven pricing excellence.

 **Efficient Lead Management**: Our lead classification model helps you identify the most promising leads, enabling your team to focus their efforts on leads with the highest conversion potential. This results in higher conversion rates and increased revenue.

Key Features

 **Data Analytics**: Our platform offers robust data analytics tools that allow you to explore and visualize your sales data, uncovering valuable insights that drive smarter business decisions.

 **Machine Learning**: We employ advanced machine learning algorithms to handle complex data challenges, such as skewness and noisy data, delivering more accurate predictions and pricing recommendations.

 **Lead Classification**: Our lead classification model is designed to evaluate leads based on their likelihood to convert into customers. This feature streamlines your lead management process and maximizes your sales team's efficiency.

 How It Works

1. **Lead Classification**: Our lead classification model identifies leads as 'WON' (success) or 'LOST' (failure) with precision. This helps you prioritize your sales efforts effectively.

2. **Pricing Optimization**: Use our machine learning models to optimize pricing decisions. We provide data normalization, feature scaling, and outlier detection to ensure your pricing is competitive and profitable.""")
    st.markdown('__<p style="text-align:left; font-size: 20px; color: #FAA026">For feedback/suggestion, connect with me on</P>__',
                unsafe_allow_html=True)
    st.subheader("Email ID")
    st.write("santhoshkumar.e2000@gmail.com")
    st.subheader("Github")
    st.write("https://github.com/Sandy1630")
    st.balloons()
if selected=="selling_price":
    
    st.title("Price Prediction") 
    country=st.selectbox("country",sorted(country_option))
    application=st.selectbox("application",sorted(application_option))
    product_ref=st.selectbox("product ref",product_option)
    status=st.selectbox("status",status_option)
    item_type=st.selectbox("item type",item_option)
    quantity_tons = st.text_input("Enter Quantity Tons (Min:611728 & Max:1722207579)")
    thickness = st.text_input("Enter thickness (Min:0.18 & Max:400)")
    width = st.text_input("Enter width (Min:1, Max:2990)")
    customer = st.text_input("customer ID (Min:12458, Max:30408185)")
    submit_button = st.button(label="selling_price")
    st.markdown("""<style>
                    div.stButton > button:first-child {
                        background-color: #990000;
                        color: white;
                        width: 100%; }
                    </style>
                """, unsafe_allow_html=True)

    flag = 0
    pattern = "^(?:\d+|\d*\.\d+)$"
    for i in [quantity_tons, thickness, width, customer]:
        if re.match(pattern, i):
            pass
        else:
            flag = 1
            break

    if submit_button and flag == 1:
        if len(i) == 0:
            st.write("please enter a valid number space not allowed")
        else:
            st.write("You have entered an invalid value: ", i)

    if submit_button and flag==0:
        import pickle
        with open(r"C:\Users\santh\model.pkl",'rb') as file:
            model=pickle.load(file)
        with open(r"labenc.pkl","rb") as f :
            le=pickle.load(f)
        with open(r"scaler.pkl","rb") as f :
            sc=pickle.load(f)
        with open(r"labenc1.pkl","rb") as f:
            le1=pickle.load(f)
            
            if thickness and quantity_tons and customer and width :
                quantity_tons=float(quantity_tons)
                thickness=float(thickness)
                width=float(width)
                customer=float(customer)

                new_sample=np.array([[quantity_tons,float(customer),float(country),float(application),
                thickness,width,float(product_ref),status,item_type]])
                new_sample_le=le.transform(new_sample[:,[7]])
                new_sample_le1=le1.transform(new_sample[:,[8]])
                new_sample_le=new_sample_le.reshape(-1,1)
                new_sample_le1=new_sample_le1.reshape(-1,1)
                new_sample_sc=np.concatenate((new_sample[:,[0,1,2,3,4,5,6]],new_sample_le,new_sample_le1),axis=1)
                new_sample1=sc.transform(new_sample_sc)
                new_pred=model.predict(new_sample1)[0]
                st.write("## :green[price:]",round(np.exp(new_pred)))


if selected=="status":
    
    st.title("Status Prediction")
    country=st.selectbox("country",sorted(country_option))
    application=st.selectbox("application",sorted(application_option))
    product_ref=st.selectbox("product ref",product_option)
    selling_price=st.text_input("Enter Selling Price (Min:0 & Max:100001015")
    item_type=st.selectbox("item type",item_option)
    quantity_tons = st.text_input("Enter Quantity Tons (Min:611728 & Max:1722207579)")
    thickness = st.text_input("Enter thickness (Min:0.18 & Max:400)")
    width = st.text_input("Enter width (Min:1, Max:2990)")
    customer = st.text_input("customer ID (Min:12458, Max:30408185)")
    submit_button = st.button(label="status")
    st.markdown("""<style>
                    div.stButton > button:first-child {
                        background-color: #990000;
                        color: white;
                        width: 100%; }
                    </style>
                """, unsafe_allow_html=True)

    flag = 0
    pattern = "^(?:\d+|\d*\.\d+)$"
    for i in [quantity_tons, thickness, width, customer,selling_price]:
        if re.match(pattern, i):
            pass
        else:
            flag = 1
            break

    if submit_button and flag == 1:
        if len(i) == 0:
            st.write("please enter a valid number space not allowed")
        else:
            st.write("You have entered an invalid value: ", i)

    if submit_button and flag==0:
        import pickle
        with open(r"class_model.pkl",'rb') as file:
            model=pickle.load(file)
        with open(r"lableenc2.pkl","rb") as f :
            le=pickle.load(f)
        with open(r"scaler2.pkl","rb") as f :
            sc=pickle.load(f)
            
            if thickness and quantity_tons and customer and width and selling_price:
                quantity_tons=float(quantity_tons)
                thickness=float(thickness)
                width=float(width)
                customer=float(customer)

                new_sample=np.array([[quantity_tons,float(customer),float(country),float(application),
                thickness,width,float(product_ref),np.log(float(selling_price)),item_type]])
                new_sample_le1=le.transform(new_sample[:,[8]])
                new_sample_le1=new_sample_le1.reshape(-1,1)
                new_sample_sc=np.concatenate((new_sample[:,[0,1,2,3,4,5,6,7]],new_sample_le1),axis=1)
                new_sample1=sc.transform(new_sample_sc)
                new_pred=model.predict(new_sample1)[0]
                if new_pred==1:
                    st.write("## :green[Status is Won]")
                else:
                    st.write("##:red[Status is Lost]")
st.write( f'<h6 style="color:rgb(0, 153, 153,0.35);">App Created by Sandy</h6>', unsafe_allow_html=True )






