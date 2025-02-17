import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd

from src.data_management import download_dataframe_as_csv
from src.machine_learning.predictive_analysis import (
                                                    load_model_and_predict,
                                                    resize_input_image,
                                                    plot_predictions_probabilities
                                                    )

def page_malaria_detector_body():
    st.info(
        f"* The client is interested in telling whether a given cell contains a malaria parasite "
        f"or not."
        )

    st.write(
        f"* You can download a set of parasitised and uninfected cells for live prediction. "
        f"You can download the images from [here](https://www.kaggle.com/codeinstitute/cell-images-test)."
        )

    st.write("---")

    images_buffer = st.file_uploader('Upload blood smear samples. You may select more than one.',
                                        type='png',accept_multiple_files=True)
    # We have used st.file_uploader here to upload images from the view where
    # we have set multiple file uploads and allowed image type as a png image file.
    if images_buffer is not None:
        df_report = pd.DataFrame([])
        for image in images_buffer:

            img_pil = (Image.open(image))
            st.info(f"Blood Smear Sample: **{image.name}**")
            img_array = np.array(img_pil)
            st.image(img_pil, caption=f"Image Size: {img_array.shape[1]}px width x {img_array.shape[0]}px height")
            # We are using the st.image function to display a pil image on the view and passing the uploaded
            # image through the three functions which we have discussed before to resize the images,
            # these custom functions are inside the machine_learning/predictive_analysis.py file in this workspace
            # that we imported above.
            version = 'v1'
            resized_img = resize_input_image(img=img_pil, version=version)
            pred_proba, pred_class = load_model_and_predict(resized_img, version=version)
            plot_predictions_probabilities(pred_proba, pred_class)
            # predict the labelled result over the cell images with our model and custom function,
            # and plot probabilities for the result using Plotly bar plot with also our custom function.

            df_report = df_report._append({"Name":image.name, 'Result': pred_class },
                                        ignore_index=True)
            # We are also creating a report with all the predictions and
            # displaying this report in a table over the page.
        
        if not df_report.empty:
            st.success("Analysis Report")
            st.table(df_report)
            st.markdown(download_dataframe_as_csv(df_report), unsafe_allow_html=True)
            
# To download this report we are using st.markdown to create a downloadable link and passing our
# report through the download_dataframe_as_csv function. Try to look into this function to
# understand the logic which makes our report change into CSV and allow users to download it.


