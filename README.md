# Diabetes Prediction Portal

This project is a *Diabetes Prediction Portal*, designed to help users predict the likelihood of diabetes based on input data. It uses a Machine Learning model integrated with a user-friendly interface for predictions. The project is built with a **Flask** backend, an interactive **HTML/CSS** frontend, and employs **scikit-learn** for model training and predictions.

## System Architecture
1. *Frontend*: The user interface is built using HTML and styled with CSS. It provides a form to input medical data (e.g., glucose levels, BMI, age).
2. *Backend*: Flask-based application handles form submissions, preprocesses data, and serves predictions using the trained ML model.
3. *Machine Learning*:
   - SVM model trained on the Pima Indians Diabetes Dataset.
   - Scikit-learn used for model training and scaling of input features.
4. *Storage*:
   - Model and scaler are saved locally as .pkl files using Python's pickle library for easy reusability.

## Key Decisions and Trade-offs
1. *Model Choice*:
   - *SVM*: Chosen for its ability to handle smaller datasets effectively and its robustness with high-dimensional spaces.
   - *Logistic Regression* was considered but not chosen due to potentially lower accuracy in the given dataset's feature space.
2. *Storage*:
   - Local storage of the model and scaler was preferred for simplicity and quick prototyping.
   - An alternative like an S3 bucket could provide better scalability but was not necessary for this project.
3. *Framework*:
   - Flask was chosen for its lightweight and simple integration capabilities.
   - A more extensive framework like Django was deemed unnecessary for the project's scope.

## Steps to Run Locally

1. *Clone the Repository*:
bash
git clone https://github.com/SREEKAR2499/ML_DIABETES.git
cd Diabetes_Prediction


2. *Install Dependencies*:
Ensure you have Python installed. Then, run:
bash
pip install -r requirements.txt


3. *Run the Application*:
bash
python app.py

Access the application at http://127.0.0.1:5000/ in your web browser.

## How the Model was Trained
1. *Dataset*:
   - The Pima Indians Diabetes Dataset was used.
   - Features include medical test results and demographic information.
2. *Preprocessing*:
   - Standardization of features using StandardScaler to ensure all inputs are on the same scale.
   - Data was split into training (80%) and testing (20%) sets.
3. *Model Training*:
   - An SVM with a linear kernel was trained using scikit-learn's SVC.
   - Probabilities were enabled for better interpretability.
4. *Saving Model*:
   - The trained model and scaler were serialized using Python's pickle library and saved as diabetes_model.pkl and scaler.pkl respectively.
