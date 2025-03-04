import streamlit as st
from app_pages.multipage import MultiPage

# load pages scripts
# Here we import the page views from the app_pages folder
from app_pages.page_summary import page_summary_body
from app_pages.page_cells_visualizer import page_cells_visualizer_body
from app_pages.page_malaria_detector import page_malaria_detector_body
from app_pages.page_project_hypothesis import page_project_hypothesis_body
from app_pages.page_ml_performance import page_ml_performance_metrics

app = MultiPage(app_name="Malaria Detector")  # Create an instance of the app

# Below are our page views that we made inside the app_pages folder with streamlit components.
# This file acts as a controller for the app. Similar to django views and urls.
# Add your app pages here using .add_page()
app.add_page("Quick Project Summary", page_summary_body)
app.add_page("Cells Visualiser", page_cells_visualizer_body)
app.add_page("Malaria Detection", page_malaria_detector_body)
app.add_page("Project Hypothesis", page_project_hypothesis_body)
app.add_page("ML Performance Metrics", page_ml_performance_metrics)

app.run()  # Run the app
