# House Price Prediction

A simple end-to-end Machine Learning project that predicts California house prices based on various features. It uses Scikit-Learn for the model and Streamlit for the user interface.

## Project Structure
- `src/train.py`: The script to train the model.
- `app.py`: The Streamlit web application.
- `models/`: Directory where the trained model is saved.
- `requirements.txt`: Python dependencies.

-->Setup Instructions

1. Install Dependencies
Make sure you have Python installed, then run:

```bash
pip install -r requirements.txt
```

### 2. Train the Model
Before running the app, you need to train the model and generate the `.joblib` file:

```bash
python src/train.py
```

### 3. Run the App
Launch the Streamlit web application:

```bash
streamlit run app.py
```


###after this all code should work 