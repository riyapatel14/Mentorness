from flask import Flask, render_template, request
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

app = Flask(__name__)

# Sample data preprocessing (you should adjust this according to your actual preprocessing steps)
def preprocess_data(df):
    # Convert categorical variables
    df['Gender'] = df['Gender'].map({"F": 0, "M": 1})
    df['Marital status'] = df['Marital status'].map({"Single / not married": 0, "Married": 1, "Civil marriage": 2, "Widow": 3, "Separated": 4})
    df['Employment status'] = df['Employment status'].map({"Commercial associate": 0, "Pensioner": 1, "State servant": 2, "Student": 3, "Working": 4})
    df['Education level'] = df['Education level'].map({"Academic degree": 0, "Higher education": 1, "Incomplete higher": 2, "Lower secondary": 3, "Secondary / secondary special": 4})
    
    # Drop unnecessary columns
    df.drop(columns=['ID', 'Job title'], inplace=True)
    
    # Perform one-hot encoding if necessary
    df = pd.get_dummies(df)
    
    return df

# Sample function to load data and preprocess
def load_and_preprocess_data():
    df = pd.read_csv(r"train_data.csv")
    df1 = pd.read_csv(r"test_data.csv")
    
    # Save original copies if needed
    train_original = df.copy()
    test_original = df1.copy()
    
    # Preprocess training and test data
    x = preprocess_data(df)
    xtest = preprocess_data(df1)
    
    # Assuming y and other variables are defined in your actual code
    y = df["Is high risk"]
    ytest = df1["Is high risk"]
    
    return x, y, xtest, ytest

# Assuming you have loaded your data and processed it
x, y, xtest, ytest = load_and_preprocess_data()

# Instantiate KNN Classifier
knn_classifier = KNeighborsClassifier()

# Oversample the minority class using SMOTE
smote = SMOTE()
x_resampled, y_resampled = smote.fit_resample(x, y)

# Fit the classifier on the resampled data
knn_classifier.fit(x_resampled, y_resampled)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    income = float(request.form['income'])
    age = int(request.form['age'])
    employment_length = int(request.form['employment_length'])
    family_member_count = int(request.form['family_member_count'])
    account_age = int(request.form['account_age'])
    gender = int(request.form['gender'])
    employment_status = int(request.form['employment_status'])
    education_level = int(request.form['education_level'])
    marital_status = int(request.form['marital_status'])
    
    # Create a DataFrame with the user input
    user_input = pd.DataFrame([[income, age, employment_length, family_member_count, account_age, gender, employment_status, education_level, marital_status]],
                              columns=['Income', 'Age', 'Employment length', 'Family member count', 'Account age', 'Gender', 'Employment status', 'Education level', 'Marital status'])

    # Ensure the user_input matches the feature set used for training
    user_input = pd.get_dummies(user_input)
    
    # Add missing columns if any
    missing_cols = set(x.columns) - set(user_input.columns)
    for col in missing_cols:
        user_input[col] = 0
    
    # Ensure the order of columns matches
    user_input = user_input[x.columns]
    
    # Use your trained model to make predictions
    pred_knn = knn_classifier.predict(user_input)
    
    # Determine prediction result
    if pred_knn[0] == 1:
        prediction = "The credit card will be approved."
    else:
        prediction = "The credit card will be rejected."
    
    return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
