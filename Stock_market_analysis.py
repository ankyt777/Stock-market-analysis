import streamlit as st
from streamlit_option_menu import option_menu

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

import yfinance as yf
import datetime as dt
from datetime import date
from datetime import timedelta

start = "2015-01-01"
end =date.today().strftime("%Y-%m-%d")

st.title('Stock Market Analysis')

with st.sidebar:
    selected = option_menu(
        menu_title="Main Menu",
        options=("Home","Stock Prediction","About"),
        icons=("house","graph-up-arrow","file-person-fill"),
        
    )
if selected=="Home":
    st.title(f"Stock Detail")
    user_input=st.text_input('Enter Stock Ticker','GOOG')
    

    stock_info = yf.Ticker(user_input).info
    company_name = stock_info['shortName']
    st.subheader(company_name)

    company = yf.Ticker(user_input) 
    
    ticker_info = yf.download(user_input, start="2021-10-01", end=date.today().strftime("%Y-%m-%d"))
    data1 = company.history(period="1y")



    # summary 
    st.write(company.info['longBusinessSummary'])
    st.write(ticker_info)

    # graph
    st.line_chart(data1.values)

   

    
if selected=="Stock Prediction":
    st.title(f"Stock Prediction")
    user_input=st.text_input('Enter Stock Ticker','GOOG')
    df=data.DataReader(user_input,'yahoo',start,end)
    #data
    st.subheader("Data from 2015-2022")
    st.write(df.describe())

   #visualization
    st.subheader("Closing price vs time chart")
    fig=plt.figure(figsize=(12,6))
    plt.plot(df.Close)
    st.pyplot(fig)

    
    
    
    
    
    


    st.subheader("Closing price vs time chart with 100MA and 200MA")
    ma100= df.Close.rolling(100).mean()
    ma200= df.Close.rolling(200).mean()
    fig=plt.figure(figsize=(12,6))
    plt.plot(ma100,'r')
    plt.plot(ma200,'g')
    plt.plot(df.Close,'b')
    st.pyplot(fig)

 #splitting data into training and testing
    data_training=pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
    data_testing=pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

    scaler=MinMaxScaler(feature_range=(0,1))

    data_training_array=scaler.fit_transform(data_training)


   #load my model
    model=load_model('keras_model.h5')

 #testing part
    past_100_days=data_training.tail(100)
    final_df=past_100_days.append(data_testing, ignore_index=True)
    input_data = scaler.fit_transform(final_df)

    x_test=[]
    y_test=[]

 
    for i in range(100,input_data.shape[0]):
        x_test.append(input_data[i-100:i])
        y_test.append(input_data[i,0])

    x_test,y_test=np.array(x_test),np.array(y_test)
    y_predicted=model.predict(x_test)
    scaler=scaler.scale_

    scale_factor=1/scaler[0]
    y_predicted=y_predicted * scale_factor
    y_test=y_test * scale_factor


 #final graph

    st.subheader("Prediction VS Original")
    fig2=plt.figure(figsize=(12,6))
    plt.plot(y_test,'b',label='Original price')
    plt.plot(y_predicted,'r',label='Predicted price')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    st.pyplot(fig2)
    
 # future Prediction 
    st.subheader('Stock Price Prediction by Date')
    
    df1=df.reset_index()['Close']
    scaler=MinMaxScaler(feature_range=(0,1))
    df1=scaler.fit_transform(np.array(df1).reshape(-1,1))
    datemax=dt.datetime.strftime(dt.datetime.now() - timedelta(1), "%d/%m/%Y")
    datemax =dt.datetime.strptime(datemax,"%d/%m/%Y")
    x_input=df1[:].reshape(1,-1)
    temp_input=list(x_input)
    temp_input=temp_input[0].tolist()
    date1 = st.date_input("Enter Date in this format yyyy-mm-dd")

    result = st.button("Predict")
    if result:

        from datetime import datetime
        my_time = datetime.min.time()
        date1 = datetime.combine(date1, my_time)
    
    
        nDay=date1-datemax
        nDay=nDay.days

        date_rng = pd.date_range(start=datemax, end=date1, freq='D')
        date_rng=date_rng[1:date_rng.size]
        lst_output=[]
        n_steps=x_input.shape[1]
        i=0

        while(i<=nDay):
            
            if(len(temp_input)>n_steps):
                
                x_input=np.array(temp_input[1:]) 
                
                x_input=x_input.reshape(1,-1)
                x_input = x_input.reshape((1, n_steps, 1))

                yhat = model.predict(x_input, verbose=0)
                
                temp_input.extend(yhat[0].tolist())
                temp_input=temp_input[1:]
                lst_output.extend(yhat.tolist())
                i=i+1
            else:
                x_input = x_input.reshape((1, n_steps,1))
                yhat = model.predict(x_input, verbose=0)
                
                temp_input.extend(yhat[0].tolist())
                
                lst_output.extend(yhat.tolist())
                i=i+1
        res =scaler.inverse_transform(lst_output)


        output = res[nDay]

        st.write("*Predicted Price for Date :*", date1, "*is*", (np.round_(output[0], 2)))

    

        predictions=res[res.size-nDay:res.size]
        
        predictions=predictions.ravel()

        fig =plt.figure(figsize=(10,6))
        xpoints = date_rng
        ypoints =predictions
        plt.xticks(rotation = 90)
        plt.plot(xpoints, ypoints)
        st.pyplot(fig)

if selected=="About":
    st.title(f"About")
    st.write(f'Thanks for visiting my webapp')
    st.write(f'Developed by Ankit Ghildiyal')
    