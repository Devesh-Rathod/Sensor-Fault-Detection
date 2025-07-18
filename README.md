**Sensor Fault Detection on Silicon Wafers** project. It provides an overview of the project, instructions for setup, usage, and other relevant details based on the context of your code and the goal of classifying silicon wafers as "good" or "bad" using sensor data.

---

# Sensor Fault Detection on Silicon Wafers

## Project Overview
This project focuses on detecting faults in silicon wafers using sensor data. The goal is to classify wafers as **"good"** or **"bad"** based on sensor readings collected during testing. The dataset is stored in a MongoDB database, and the project uses Python for data processing, storage, and analysis. The pipeline involves loading sensor data from a CSV file, storing it in MongoDB Atlas, and preparing it for machine learning-based classification.

### Objectives
- **Data Storage**: Store wafer sensor data in a MongoDB Atlas database for efficient retrieval and processing.
- **Data Processing**: Clean and preprocess sensor data for analysis.
- **Classification**: Develop a model to classify wafers as "good" or "bad" based on sensor readings.
- **Scalability**: Handle large datasets by inserting records in batches to avoid timeouts.

## Dataset
The dataset (`sensor-fault-detection.csv`) contains sensor readings from silicon wafer tests. Each row represents a wafer with various sensor measurements, and the target variable indicates whether the wafer is "good" or "bad." The dataset is loaded into a MongoDB collection for further analysis.

## Prerequisites
To run this project, ensure you have the following installed:
- **Python**: Version 3.6 or higher
- **MongoDB Atlas Account**: A free or paid MongoDB Atlas cluster for storing the dataset
- **Dependencies**:
  - `pymongo`: For MongoDB connectivity
  - `pandas`: For data manipulation
  - `certifi`: For SSL certificate verification
  - `numpy` (optional): For numerical operations
  - `scikit-learn` (optional): For machine learning model development

Install the required Python packages:
```bash
pip install pymongo pandas certifi numpy scikit-learn
```

## Setup Instructions

### 1. MongoDB Atlas Configuration
1. **Create a MongoDB Atlas Cluster**:
   - Sign up or log in to [MongoDB Atlas](https://www.mongodb.com/cloud/atlas).
   - Create a cluster (e.g., free tier M0) and note the cluster name (e.g., `Cluster0`).
2. **Set Up Database Access**:
   - In the MongoDB Atlas dashboard, go to **Database Access** and create a user (e.g., username: `devesh`, password: `devesh`).
   - Grant the user `readWrite` permissions for the database.
3. **Configure Network Access**:
   - In **Network Access**, add your IP address or `0.0.0.0/0` (allow all, for testing only) to the IP allowlist.
4. **Get Connection String**:
   - Go to **Connect** in your cluster, select "Connect your application," and copy the Python connection string (e.g., `mongodb+srv://devesh:devesh@cluster0.rnxdtyn.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0`).

### 2. Project Structure
```plaintext
sensor-fault-detection/
│
├── notebook/
│   ├── sensor-fault-detection.csv  # Input dataset
│   ├── data_upload.py              # Script to upload data to MongoDB
│
├── README.md                       # Project documentation
├── requirements.txt                # Required Python packages
└── models/                         # (Optional) Directory for trained models
```

### 3. Data Upload to MongoDB
The `data_upload.py` script loads the CSV dataset and inserts it into a MongoDB Atlas collection in batches to avoid timeouts. Update the script with your MongoDB Atlas connection string and file path.

Example `data_upload.py`:
```python
from pymongo import MongoClient
import pandas as pd
import json
import certifi

# MongoDB connection URI
uri = "mongodb+srv://<username>:<password>@cluster0.rnxdtyn.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

# Create a new client with SSL certificate and increased timeouts
client = MongoClient(uri, tlsCAFile=certifi.where(), serverSelectionTimeoutMS=30000, connectTimeoutMS=30000)

# Test connection
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(f"Connection failed: {e}")
    exit()

# Database and collection names
DATABASE_NAME = "Dev"
COLLECTION_NAME = "wafer fault dataset"

# Read CSV file
try:
    df = pd.read_csv("./notebook/sensor-fault-detection.csv")
    print("CSV loaded successfully")
except Exception as e:
    print(f"Error loading CSV: {e}")
    exit()

# Replace NaN with None for MongoDB compatibility
df = df.where(pd.notnull(df), None)

# Convert DataFrame to list of JSON records
try:
    json_record = list(json.loads(df.T.to_json()).values())
    print(f"Converted {len(json_record)} records to JSON")
except Exception as e:
    print(f"Error converting to JSON: {e}")
    exit()

# Insert records in batches
batch_size = 1000
try:
    for i in range(0, len(json_record), batch_size):
        client[DATABASE_NAME][COLLECTION_NAME].insert_many(json_record[i:i + batch_size])
        print(f"Inserted {i + batch_size} records")
    print("Data insertion completed successfully!")
except Exception as e:
    print(f"Error during insertion: {e}")

# Close the client
client.close()
```

Update the `<username>`, `<password>`, and file path in the script.

### 4. Running the Script
1. Place the `sensor-fault-detection.csv` file in the `notebook/` directory.
2. Run the data upload script:
   ```bash
   python notebook/data_upload.py
   ```
3. Verify the data in MongoDB Atlas:
   - Go to **Collections** in the Atlas dashboard and check the `Dev.wafer fault dataset` collection.

## Usage
1. **Data Preprocessing**:
   - Clean the dataset (e.g., handle missing values, normalize sensor readings).
   - Example: Use `pandas` to drop irrelevant columns or encode the target variable ("good" or "bad").

2. **Model Development**:
   - Retrieve data from MongoDB using PyMongo queries.
   - Train a machine learning model (e.g., using `scikit-learn`) to classify wafers as "good" or "bad."
   - Example models: Logistic Regression, Random Forest, or Neural Networks.

3. **Evaluation**:
   - Evaluate the model using metrics like accuracy, precision, recall, and F1-score.
   - Use cross-validation to ensure robustness.

4. **Prediction**:
   - Use the trained model to predict the quality of new wafers based on sensor data.

Example code to retrieve and process data:
```python
from pymongo import MongoClient
import pandas as pd
import certifi

uri = "mongodb+srv://<username>:<password>@cluster0.rnxdtyn.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(uri, tlsCAFile=certifi.where())

# Retrieve data from MongoDB
db = client["Dev"]
collection = db["wafer fault dataset"]
data = pd.DataFrame(list(collection.find()))

# Preprocess and train model (example)
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# X = data.drop("target", axis=1)  # Assuming 'target' is the column for good/bad
# y = data["target"]
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# model = RandomForestClassifier()
# model.fit(X_train, y_train)
# print(model.score(X_test, y_test))

client.close()
```

## Troubleshooting
- **SSL Certificate Error**:
  - Ensure `certifi` is installed (`pip install certifi`).
  - Add `tlsCAFile=certifi.where()` to the `MongoClient`.
  - Update your system's CA certificates (see OS-specific instructions).
- **Timeout Error**:
  - Increase timeouts in `MongoClient` (`serverSelectionTimeoutMS=30000, connectTimeoutMS=30000`).
  - Insert data in smaller batches (e.g., `batch_size = 1000`).
- **Connection Issues**:
  - Verify your MongoDB Atlas connection string, IP allowlist, and database user credentials.
  - Check network connectivity and disable VPNs/proxies if necessary.

## Future Improvements
- Implement feature engineering to improve model accuracy.
- Add data visualization (e.g., using `matplotlib` or `seaborn`) to analyze sensor patterns.
- Deploy the model as an API for real-time wafer classification.
- Optimize MongoDB queries for faster data retrieval.

## Contributing
Contributions are welcome! Please:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/YourFeature`).
3. Commit changes (`git commit -m "Add YourFeature"`).
4. Push to the branch (`git push origin feature/YourFeature`).
5. Open a pull request.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact
For questions or support, contact deveshrathod15@gmail.com or open an issue in the repository.

---