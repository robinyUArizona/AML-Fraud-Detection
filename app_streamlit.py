import streamlit as st
import dill
from aml_fraud_detector.pipeline.prediction_pipeline import CustomData, PredictionPipeline
from aml_fraud_detector.exception import CustomerException
from aml_fraud_detector.logger import logging
import pandas as pd

def main():
    logging.info(f"Starting Streamlit App")
    st.write(
    """
    # Anti-Money Laundering (AML) Fraud Detection
            
    This app predicts the given transaction is *fraud* or *not fraud*
    """
    )
    st.write("-------")

    # Sidebar
    # Header of specify Input Parameters
    st.sidebar.header("Specify Input Features")


    def user_input_features():
        data = CustomData(
                from_bank = st.sidebar.number_input("from_bank"),
                account = st.sidebar.text_input("account"),
                to_bank = st.sidebar.number_input("to_bank"),
                account_1 = st.sidebar.text_input("account_1"),
                amount_received =  st.sidebar.number_input("amount_received"),
                receiving_currency = st.sidebar.text_input("receiving_currency"),
                payment_currency = st.sidebar.text_input("payment_currency"),
                payment_format = st.sidebar.text_input("payment_format")
            )
        features_df = data.get_data_as_DataFrame()
        # print(features_df)
        return features_df

    df = user_input_features()

    # Print specified input parameters
    st.header("Specified Input parameters")
    st.write(df)
    st.write("----------")

    # Call best model
    predict_pipeline = PredictionPipeline()
    # Apply best model to make prediction
    prediction = predict_pipeline.predict(df)
    prediction_proba = predict_pipeline.predict_proba(df)

    st.subheader('Fraud detector class labels')
    class_labels_df = pd.DataFrame({"Not Fraud" : [0], "Fraud" : [1]})
    class_labels_df.index = ["Class labels"]
    st.write(class_labels_df.T)

    st.header("Prediction of a given transaction")
    st.write(prediction)
    st.write("-------")

    st.header("Prediction probabilities of a given transaction")
    st.write(prediction_proba)
    st.write("-------")
    logging.info(f"Streamlit app execution completed")


if __name__ == "__main__":
    main()

# Explaining the model's predictions using SHAP values
# https://github.com/slundberg/shap
# explainer = shap.TreeExplainer(model)
# shap_values = explainer.shap_values(X)

# def main():
#     st.set_page_config(
#         page_title="Anti-Money Laundering (AML) Fraud Detection",
#         page_icon=":bank:",
#         layout="wide",

#     )

#     with st.container():
#         st.title("Anti-Money Laundering (AML) Fraud Detection")
#         st.write("Please predict the given transaction is fraud or not")

#     col1, col2 = st.columns([4, 1])

#     with col1:
#         st.write("This is column 1")

#     with col2:
#         st.write("This is column 2")
    

