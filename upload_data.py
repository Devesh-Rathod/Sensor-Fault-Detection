from pymongo.mongo_client import MongoClient
import pandas as pd
import json

#url
import certifi

uri = "mongodb+srv://devesh:devesh@cluster0.rnxdtyn.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(uri, tlsCAFile=certifi.where())



#create a new client and connectt to server


#create database name and collection name
DATABASE_NAME="Dev"
COLLECTION_NAME='wafer fault dataset'

df=pd.read_csv("/Users/deveshrathod/Desktop/Data Analytics/Projects/Sensor Fault Detect/notebook/sensor-fault-detection.csv")
#df.head()
#df=df.drop("Unnamed: 0",axis=1)

json_record=list(json.loads(df.T.to_json()).values())

client[DATABASE_NAME][COLLECTION_NAME].insert_many(json_record)