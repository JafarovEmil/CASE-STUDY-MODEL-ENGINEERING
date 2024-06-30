#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Import libraries to find duplicated rows. Duplicated transactions are called transactions within one minute, with the same amount of money and from the same country
 
import pandas as pd import numpy as np
from datetime import datetime
 
df1= pd.read_csv('/Users/emilcafarov/Desktop/PSP_Jan_Feb_2019.csv')
drop_list=[]
 
for i in range(len(df1)-1):
    if df1.loc[i,'country']==df1.loc[i+1,'country'] and df1.loc[i,'amount']==df1.loc[i+1,'amount']:
        dt1 = df1.loc[i,'tmsp'] 
        dt2 = df1.loc[i+1,'tmsp'] 
        delta=dt2-dt1
        if delta.seconds<=60: 
            drop_list.append(i)
#Eliminating duplicated rows to see how many rows remained
raw_dataset = df1.drop(df1.index[drop_list]) 
raw_dataset=raw_dataset.reset_index() 
raw_dataset.shape[0]

#Eliminating unnecessary column with the name:'Unnamed: 0'
raw_dataset = raw_dataset.drop('Unnamed: 0', axis=1)
raw_dataset.head(10)
 
#Seperating the card column into these 3 new columns:Is_Visa,Is_Diners,Is_Master 
for i in range(len(raw_dataset)):
    if raw_dataset.iloc[i,7]=='Visa':
        prepros_dataset.loc[i,'Is_Visa']=1 
        prepros_dataset.loc[i,'Is_Diners']=0 
        prepros_dataset.loc[i,'Is_Master']=0 
    elif raw_dataset.iloc[i,7]=='Diners': 
        prepros_dataset.loc[i,'Is_Visa']=0 
        prepros_dataset.loc[i,'Is_Diners']=1 
        prepros_dataset.loc[i,'Is_Master']=0 
    elif raw_dataset.iloc[i,7]=='Master': 
        prepros_dataset.loc[i,'Is_Visa']=0 
        prepros_dataset.loc[i,'Is_Diners']=0 
        prepros_dataset.loc[i,'Is_Master']=1

#Segregating the country column into 3 columns which names are: Is_Germany,Is_Austia,Is_Switzerland 
prepros_dataset=pd.DataFrame()
for i in range(len(raw_dataset)):
    if raw_dataset.iloc[i,2]=='Germany': 
        prepros_dataset.loc[i,'Is_Germany']=1 
        prepros_dataset.loc[i,'Is_Austria']=0 
        prepros_dataset.loc[i,'Is_Switzerland']=0
    elif raw_dataset.iloc[i,2]=='Austria': 
        prepros_dataset.loc[i,'Is_Germany']=0 
        prepros_dataset.loc[i,'Is_Austria']=1 
        prepros_dataset.loc[i,'Is_Switzerland']=0
    elif raw_dataset.iloc[i,2]=='Switzerland': 
        prepros_dataset.loc[i,'Is_Germany']=0 
        prepros_dataset.loc[i,'Is_Austria']=0 
        prepros_dataset.loc[i,'Is_Switzerland']=1
prepros_dataset.head()
 
#describing the amount column to see it's mean and standard deviation 
raw_dataset['amount'].describe(percen tiles=[.05,.95])
 
#Seperating the amount column into these three categories: amount0,amount1,amount2
for i in range(len(raw_dataset)):
    if raw_dataset.iloc[i,3]<100:
        prepros_dataset.loc[i,'amount0']=1 
        prepros_dataset.loc[i,'amount1']=0 
        prepros_dataset.loc[i,'amount2']=0
    elif 100<= raw_dataset.iloc[i,3] <=300: 
        prepros_dataset.loc[i,'amount0']=0 
        prepros_dataset.loc[i,'amount1']=1 
        prepros_dataset.loc[i,'amount2']=0
    elif raw_dataset.iloc[i,3] >300: 
        prepros_dataset.loc[i,'amount0']=0 
        prepros_dataset.loc[i,'amount1']=0 
        prepros_dataset.loc[i,'amount2']=1

prepros_dataset.head()
 
#Determining day of week and digitizing the day into 7 days of a week according to the transaction.
for i in range(len(raw_dataset)):
    weekday=datetime.weekday(raw_data set.at[i,'tmsp'])
    if weekday==0: 
        prepros_dataset.loc[i,'Monday']=1 
        prepros_dataset.loc[i,'Tuesday']=0 
        prepros_dataset.loc[i,'Wednesday']=0 
        prepros_dataset.loc[i,'Thursday']=0 
        prepros_dataset.loc[i,'Friday']=0 
        prepros_dataset.loc[i,'Saturday']=0 
        prepros_dataset.loc[i,'Sunday']=0
    elif weekday==1: 
        prepros_dataset.loc[i,'Monday']=0 
        prepros_dataset.loc[i,'Tuesday']=1 
        prepros_dataset.loc[i,'Wednesday']=0 
        prepros_dataset.loc[i,'Thursday']=0 
        prepros_dataset.loc[i,'Friday']=0 
        prepros_dataset.loc[i,'Saturday']=0 
        prepros_dataset.loc[i,'Sunday']=0
    elif weekday==2: 
        prepros_dataset.loc[i,'Monday']=0 
        prepros_dataset.loc[i,'Tuesday']=0 
        prepros_dataset.loc[i,'Wednesday']=1 
        prepros_dataset.loc[i,'Thursday']=0 
        prepros_dataset.loc[i,'Friday']=0 
        prepros_dataset.loc[i,'Saturday']=0 
        prepros_dataset.loc[i,'Sunday']=0
    elif weekday==3: 
        prepros_dataset.loc[i,'Monday']=0 
        prepros_dataset.loc[i,'Tuesday']=0 
        prepros_dataset.loc[i,'Wednesday']=0 
        prepros_dataset.loc[i,'Thursday']=1 
        prepros_dataset.loc[i,'Friday']=0 
        prepros_dataset.loc[i,'Saturday']=0 
        prepros_dataset.loc[i,'Sunday']=0
    elif weekday==4:
        prepros_dataset.loc[i,'Monday']=0 
        prepros_dataset.loc[i,'Tuesday']=0 
        prepros_dataset.loc[i,'Wednesday']=0 
        prepros_dataset.loc[i,'Thursday']=0 
        prepros_dataset.loc[i,'Friday']=1 
        prepros_dataset.loc[i,'Saturday']=0 
        prepros_dataset.loc[i,'Sunday']=0
    elif weekday==5: 
        prepros_dataset.loc[i,'Monday']=0 
        prepros_dataset.loc[i,'Tuesday']=0 
        prepros_dataset.loc[i,'Wednesday']=0 
        prepros_dataset.loc[i,'Thursday']=0 
        prepros_dataset.loc[i,'Friday']=0 
        prepros_dataset.loc[i,'Saturday']=1 
        prepros_dataset.loc[i,'Sunday']=0
    elif weekday==6: 
        prepros_dataset.loc[i,'Monday']=0 
        prepros_dataset.loc[i,'Tuesday']=0 
        prepros_dataset.loc[i,'Wednesday']=0 
        prepros_dataset.loc[i,'Thursday']=0 
        prepros_dataset.loc[i,'Friday']=0 
        prepros_dataset.loc[i,'Saturday']=0 
        prepros_dataset.loc[i,'Sunday']=1

prepros_dataset.head()
 
#Dividing 24 hours of a day into 24 groups which begins from hour 0 to hour23
for i in range(len(raw_dataset)): 
    hour=raw_dataset.iloc[i]['tmsp'].hour 
    for j in range (24):
        if j==hour: 
            prepros_dataset.loc[i,'hour'+ str(j)]=1 
        else:
            prepros_dataset.loc[i,'hour'+ str(j)]=0 
prepros_dataset.tail(10)

#transfering the 3D_secured column into secured column in the new dataset
for i in range(len(raw_dataset)): 
    prepros_dataset.loc[i,'secured']=raw_d ataset.loc[i,'3D_secured'] 
prepros_dataset.head() 

#This    section    of    code    defines    4 columns as our model targets which 1column belongs to one of the PSPs. # If a transaction is successful on any
#of PSPs, we consider as 1 in the corresponding PSP column and we mention 0 into three others.
#if a transaction is failed, We put 0 in all these four columns
for i in range(len(raw_dataset)):
    if raw_dataset.iloc[i,5]=='UK_Card' and raw_dataset.iloc[i,4]==1 :
        prepros_dataset.loc[i,'UK_Card_succe ss']=1 
        prepros_dataset.loc[i,'Simplecard_suc cess']=0 
        prepros_dataset.loc[i,'Moneycard_suc cess']=0 
        prepros_dataset.loc[i,'Goldcard_succe ss']=0
    elif raw_dataset.iloc[i,5]=='Simplecard' and raw_dataset.iloc[i,4]==1 : 
        prepros_dataset.loc[i,'UK_Card_succe ss']=0 
        prepros_dataset.loc[i,'Simplecard_suc cess']=1 
        prepros_dataset.loc[i,'Moneycard_suc cess']=0 
        prepros_dataset.loc[i,'Goldcard_succe ss']=0
    elif raw_dataset.iloc[i,5]=='Moneycard' and raw_dataset.iloc[i,4]==1 : 
        prepros_dataset.loc[i,'UK_Card_succe ss']=0 
        prepros_dataset.loc[i,'Simplecard_suc cess']=0 
        prepros_dataset.loc[i,'Moneycard_suc cess']=1 
        prepros_dataset.loc[i,'Goldcard_succe ss']=0
    elif raw_dataset.iloc[i,5]=='Goldcard' and raw_dataset.iloc[i,4]==1 : 
        prepros_dataset.loc[i,'UK_Card_succe ss']=0 
        prepros_dataset.loc[i,'Simplecard_suc cess']=0
        prepros_dataset.loc[i,'Moneycard_suc cess']=0 
        prepros_dataset.loc[i,'Goldcard_succe ss']=1
    elif raw_dataset.iloc[i,4]==0 : 
        prepros_dataset.loc[i,'UK_Card_succe ss']=0 
        prepros_dataset.loc[i,'Simplecard_suc cess']=0 
        prepros_dataset.loc[i,'Moneycard_suc cess']=0 
        prepros_dataset.loc[i,'Goldcard_succe ss']=0

prepros_dataset.head()
 

#Adding another column to the last of our dataset to separate successful and failed transactions.
# if a transaction failed, We put 0 in the failed column to determine this transaction failed
# if a transaction successed, We put 1 in in the success column
for i in range(len(raw_dataset)): 
    if raw_dataset.iloc[i,4]==0 :
        prepros_dataset.loc[i,'Success']=1 
    else: 
        prepros_dataset.loc[i,'Failed']=0
prepros_dataset.head()
 
#save the final processed data into a CSV file 
prepros_dataset.to_csv('preprocessed')
 
#visualizing the data using bar graph 
import matplotlib.pyplot as plt 
names=['UK_Card','Simplecard','Mone ycard','Goldcard'] 
values=[(raw_dataset.PSP=='UK_Card ').sum(),(raw_dataset.PSP=='Simpleca rd').sum(),(raw_dataset.PSP=='Money card').sum()
,(raw_dataset.PSP=='Goldcard').sum()] 
plt.bar(names,values)

#visualizing the data 
[su,ss,sm,sg]=[0,0,0,0]
[fu,fs,fm,fg]=[0,0,0,0]
for i in range(len(raw_dataset)):
    if raw_dataset.iloc[i,5]=='UK_Card':
        if raw_dataset.iloc[i,4]==1:
            su +=1 
        else:
            fu +=1
    elif raw_dataset.iloc[i,5]=='Simplecard': 
        if raw_dataset.iloc[i,4]==1:
            ss +=1 
        else:
            fs +=1
    elif raw_dataset.iloc[i,5]=='Moneycard': 
        if raw_dataset.iloc[i,4]==1:
            sm +=1 
        else:
            fm +=1
    elif raw_dataset.iloc[i,5]=='Goldcard': 
        if raw_dataset.iloc[i,4]==1:
            sg +=1 
        else:
            fg +=1

labels = ['UK_Card', 'Simplecard', 'Moneycard', 'Goldcard']
successful = [su,ss,sm,sg] 
failed = [fu,fs,fm,fg]
 
x = np.arange(len(labels)) # the label locations
width = 0.35 # the width of the bars
 
fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, successful, width, label='success')
rects2 = ax.bar(x + width/2, failed, width, label='fail')
 
# Add some text for labels, title and custom x-axis tick labels, etc. 
ax.set_ylabel('Scores') 
ax.set_title('Scores by PSP and success')
ax.set_xticks(x, labels) 
ax.legend()
 
ax.bar_label(rects1, padding=3)

ax.bar_label(rects2, padding=3) 
fig.tight_layout()
plt.show()


#splitting the dataframe into two groups:80% for training and 20% for testing
val_dataframe = dataframe.sample(frac=0.2, random_state=1337)
train_dataframe = dataframe.drop(val_dataframe.index)
 
print(
    "Using %d samples for training and %d for validation"
    % (len(train_dataframe), len(val_dataframe))
)



#Import libraries to design the model import pandas as pd
import numpy as np import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
 
#reading the processed data 
dataframe= pd.read_csv('preprocessed') 
dataframe.shape
 

#defining target columns for the model which are the last five columns:UK_Card_success,Simplecar d_success, 
#Moneycard_success,Goldcard_succe ss and failed
def dataframe_to_dataset(dataframe): 
    dataframe = dataframe.copy()
    labels =dataframe[dataframe.columns[-5:]]
ds = tf.data.Dataset.from_tensor_slices((dict(dataframe[dataframe.columns[:-5]]), labels))
ds = ds.shuffle(buffer_size=len(dataframe))

return ds
 
train_ds = dataframe_to_dataset(train_dataframe) 
val_ds = dataframe_to_dataset(val_dataframe)
 
#showing Input and Target columns
for x, y in train_ds.take(1): 
    print("Input:", x)
    print("Target:", y)

#Dataset batching
train_ds = train_ds.batch(32) 
val_ds = val_ds.batch(32)

# Numerical features 
Is_Germany=keras.Input(shape=(1,),n ame="Is_Germany") 
Is_Austria=keras.Input(shape=(1,),na me="Is_Austria") 
Is_Switzerland=keras.Input(shape=(1,), name="Is_Switzerland") 
Is_Visa=keras.Input(shape=(1,),name="Is_Visa") 
Is_Diners=keras.Input(shape=(1,),name="Is_Diners")
Is_Master=keras.Input(shape=(1,),name="Is_Master") 
amount0=keras.Input(shape=(1,),name="amount0") 
amount1=keras.Input(shape=(1,),name="amount1") 
amount2=keras.Input(shape=(1,),name="amount2") 
Monday=keras.Input(shape=(1,),name="Monday") 
Tuesday=keras.Input(shape=(1,),name="Tuesday") 
Wednesday=keras.Input(shape=(1,),name="Wednesday") 
Thursday=keras.Input(shape=(1,),name="Thursday")
Friday=keras.Input(shape=(1,),name=" Friday") 
Saturday=keras.Input(shape=(1,),name="Saturday") 
Sunday=keras.Input(shape=(1,),name="Sunday") 
hour0=keras.Input(shape=(1,),name=" hour0") 
hour1=keras.Input(shape=(1,),name=" hour1") 
hour2=keras.Input(shape=(1,),name=" hour2") 
hour3=keras.Input(shape=(1,),name=" hour3") 
hour4=keras.Input(shape=(1,),name=" hour4") 
hour5=keras.Input(shape=(1,),name=" hour5") 
hour6=keras.Input(shape=(1,),name=" hour6") 
hour7=keras.Input(shape=(1,),name=" hour7") 
hour8=keras.Input(shape=(1,),name=" hour8") 
hour9=keras.Input(shape=(1,),name=" hour9") 
hour10=keras.Input(shape=(1,),name= "hour10") 
hour11=keras.Input(shape=(1,),name= "hour11") 
hour12=keras.Input(shape=(1,),name= "hour12") 
hour13=keras.Input(shape=(1,),name= "hour13") 
hour14=keras.Input(shape=(1,),name= "hour14") 
hour15=keras.Input(shape=(1,),name= "hour15") 
hour16=keras.Input(shape=(1,),name= "hour16") 
hour17=keras.Input(shape=(1,),name= "hour17") 
hour18=keras.Input(shape=(1,),name= "hour18") 
hour19=keras.Input(shape=(1,),name= "hour19") 
hour20=keras.Input(shape=(1,),name= "hour20") 
hour21=keras.Input(shape=(1,),name= "hour21")
hour22=keras.Input(shape=(1,),name= "hour22") 
hour23=keras.Input(shape=(1,),name= "hour23") 
secured=keras.Input(shape=(1,),name="secured")
 
all_inputs = [ Is_Germany, Is_Austria, Is_Switzerland, Is_Visa, Is_Diners, Is_Master, 
              amount0, amount1, amount2, Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday,
hour0, hour1, hour2, hour3, hour4, hour5, hour6, hour7, hour8, hour9, hour10, 
              hour11, hour12, hour13, hour14, hour15, hour16, hour17, hour18, 
              hour19, hour20, hour21, hour22, hour23, secured,]
 
all_features = layers.concatenate( [
Is_Germany, Is_Austria, Is_Switzerland, Is_Visa, Is_Diners, Is_Master, amount0, amount1, amount2,
    Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday,
hour0, hour1, hour2, hour3, hour4, hour5, hour6, hour7, hour8, hour9,
    hour10, hour11, hour12, hour13, hour14, hour15, hour16, hour17, 
    hour18, hour19, hour20, hour21, hour22, hour23, secured,]
)


#designing the model
x = layers.Dense(1024, activation="relu")(all_features)
x = layers.Dropout(0.2)(x) 
x=layers.Dense(1024,activation="relu") (x)
x=layers.Dropout(0.2)(x) 
x=layers.Dense(512,activation="relu")( x)
x=layers.Dropout(0.2)(x)
output = layers.Dense(5, activation="softmax")(x)
model = keras.Model(all_inputs, output) 
optimizer=keras.optimizers.Adam(lear ning_rate=0.01) 
model.compile(optimizer, "categorical_crossentropy", metrics=["accuracy"])
 
#training the model
model.fit(train_ds,epochs=10, validation_data=val_ds)

 
# this function converts any input transaction into an input for the model 
def record_to_input_data(passed_record): 
    record=passed_record
    input_data=[]
    if record[1]=='Germany': 
        temp_list=[1,0,0] 
        input_data.extend(temp_list)
    elif record[1]=='Austria': 
        temp_list=[0,1,0] 
        input_data.extend(temp_list) 
    elif record[1]=='Switzerland': 
        temp_list=[0,0,1] 
        input_data.extend(temp_list)
    if record[5]=='Visa': 
        temp_list=[1,0,0] 
        input_data.extend(temp_list)
    elif record[5]=='Diners': 
        temp_list=[0,1,0] 
        input_data.extend(temp_list) 
    elif record[5]=='Master': 
        temp_list=[0,0,1] 
        input_data.extend(temp_list)

    if record[2]<100: 
        temp_list=[1,0,0]
        input_data.extend(temp_list) 
    elif 100<= record[2] <=300: 
        temp_list=[0,1,0] 
        input_data.extend(temp_list)
    elif record[2] >300: 
        temp_list=[0,0,1] 
        input_data.extend(temp_list)

    datetime_object = datetime.strptime(record[0],'%Y-%m-%d %H:%M:%S')
    weekday=datetime.weekday(datetime_object)
    if weekday==0: 
        temp_list=[1,0,0,0,0,0,0] 
        input_data.extend(temp_list) 
    elif weekday==1: 
        temp_list=[0,1,0,0,0,0,0] 
        input_data.extend(temp_list) 
    elif weekday==2: 
        temp_list=[0,0,1,0,0,0,0] 
        input_data.extend(temp_list) 
    elif weekday==3: 
        temp_list=[0,0,0,1,0,0,0] 
        input_data.extend(temp_list) 
    elif weekday==4: 
        temp_list=[0,0,0,0,1,0,0] 
        input_data.extend(temp_list)
    elif weekday==5: 
        temp_list=[0,0,0,0,0,1,0] 
        input_data.extend(temp_list) 
    elif weekday==6: 
        temp_list=[0,0,0,0,0,0,1] 
        input_data.extend(temp_list)

    hour=datetime_object.hour 
    for j in range (24):
        if j==hour: 
            input_data.append(1) 
        else: input_data.append(0)

    input_data.append(record[4]) 
    return input_data
 
#Providing a test record to our model to predict
record=['2019-01-22 11:02:11',"Germany",310,"UK_Card",1,"Visa"] 
test_record=record_to_input_data(record)
sample={ "Is_Germany":test_record[0],
        "Is_Austria" :test_record[1],
        "Is_Switzerland":test_record[2], 
        "Is_Visa":test_record[3], 
        "Is_Diners":test_record[4], 
        "Is_Master":test_record[5], 
        "amount0":test_record[6],
        "amount1":test_record[7],
        "amount2":test_record[8], 
        "Monday":test_record[9], 
        "Tuesday":test_record[10],
        "Wednesday":test_record[11]
        "Thursday":test_record[12],
        "Friday":test_record[13], 
        "Saturday":test_record[14],
        "Sunday":test_record[15],
        "hour0":test_record[16],
        "hour1":test_record[17], 
        "hour2":test_record[18],
        "hour3":test_record[19],
        "hour4":test_record[20], 
        "hour5":test_record[21],
        "hour6":test_record[22], 
        "hour7":test_record[23], 
        "hour8":test_record[24],
        "hour9":test_record[25],
        "hour10":test_record[26],
        "hour11":test_record[27],
        "hour12":test_record[28],
        "hour13":test_record[29],
        "hour14":test_record[30],
        "hour15":test_record[31],
        "hour16":test_record[32], 
        "hour17":test_record[33],
        "hour18":test_record[34],
        "hour19":test_record[35],
        "hour20":test_record[36],
        "hour21":test_record[37],
        "hour22":test_record[38], 
        "hour23":test_record[39], 
        "secured":test_record[40],
 
}
 
input_dict = {name: tf.convert_to_tensor([value]) for name, value in sample.items()}
predictions = model.predict(input_dict) 
print("The percent for UK_card PSP to be successful is: %.1f percent. " % (100 * predictions[0][0])
)
print(
"The percent for Simplecard PSP to be successful is: %.1f percent. " % (100 * predictions[0][1])
)
print(
"The percent for Moneycard PSP to be successful is: %.1f percent. " % (100 * predictions[0][2])
)
print(
"The percent for Goldcard PSP to be successful is: %.1f percent. " % (100 * predictions[0][3])
)
print(
"The percent for transaction to be failed is: %.1f percent. " % (100 * predictions[0][4])
)
 
# Selecting the best PSP for the test transaction
predicted = [predictions[0][0], predictions[0][1], predictions[0][2], predictions[0][3], predictions[0][4]]
multiple = 100
predicted = [x * multiple for x in predicted]
predicted = [round(x, 2) for x in predicted]

max_value = max(predicted)

if max_index == 0:
    print("UK_Card PSP should be selected.")
elif max_index == 1:
    print("Simplecard PSP should be selected.")
elif max_index == 2:
    print("Moneycard PSP should be selected.")
elif max_index == 3:
    print("Goldcard PSP should be selected.")
elif max_index == 4:
    print("Simplecard PSP should be selected.")

max_value = max(predicted) max_index = predicted.index(max_value)
 
if max_index == 1:
    print("Card A should be selected.")
elif max_index == 2:
    print("Card B should be selected.")
elif max_index == 3:
    print("Goldcard PSP should be selected.")
elif max_index == 4:
    print("Simplecard PSP should be selected.")
else:
    print("No valid card selected.")


# In[ ]:




