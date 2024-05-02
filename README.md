# Heart-Disease

Random Forest Classifier for Heart Disease Prediction
Introduction
This Python script uses a Random Forest Classifier to predict heart disease based on input data. It employs the numpy, pandas, matplotlib, and scikit-learn libraries for data manipulation, visualization, model training, and evaluation.

Requirements
Python 3.x
numpy
pandas
matplotlib
scikit-learn
You can install these dependencies using pip:

bash
Copy code
pip install numpy pandas matplotlib scikit-learn
Usage
Ensure you have Python and the required libraries installed.
Download the heart-disease.csv dataset and place it in the same directory as the script.
Run the script random_forest_classifier.py.
The script will load the dataset, split it into training and testing sets, train a Random Forest Classifier, and evaluate its performance.
The trained model will be saved as random_forest_model.pkl using pickle.
Files
random_forest_classifier.py: Main Python script for training and evaluating the Random Forest Classifier.
heart-disease.csv: Dataset containing heart disease data.
random_forest_model.pkl: Pickled file containing the trained Random Forest Classifier model.
Script Overview
Imports necessary libraries:
numpy for numerical operations
pandas for data manipulation
matplotlib for plotting
sklearn for machine learning functionalities
Loads the dataset heart-disease.csv using pandas.
Prepares the data by splitting it into features (X) and target variable (y).
Splits the data into training and testing sets using train_test_split from sklearn.
Initializes a Random Forest Classifier with 100 estimators.
Trains the classifier on the training data and evaluates its performance on the testing data.
Prints classification report, confusion matrix, and accuracy score of the model.
Performs a hyperparameter tuning loop to test the model with varying numbers of estimators.
Saves the final trained model using pickle.
Additional Notes
Feel free to modify hyperparameters, such as the number of estimators, test size, etc., to optimize the model's performance.
For larger datasets or more complex models, consider using parallel processing or cloud-based resources for faster computations.
