import streamlit as st
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.image import imread

import itertools
import random

def page_cells_visualizer_body():
    st.write("### Cells Visualizer")
    st.info(
        f"* The client is interested in having a study that visually "
        f"differentiates a parasitised from an uninfected cell.")
    
    version = 'v1'
    if st.checkbox("Difference between average and variability image"):
    # We have used st.checkbox to display the checkbox for different visualization operations.
    # We are also using st.image to display the saved plots we generated during the model evaluation on the page.
      
      avg_parasitized = plt.imread(f"outputs/{version}/avg_var_Parasitized.png")
      avg_uninfected = plt.imread(f"outputs/{version}/avg_var_Uninfected.png")

      st.warning(
        f"* We notice the average and variability images did not show "
        f"patterns where we could intuitively differentiate one from another. " 
        f"However, a small difference in the colour pigment of the average images is seen for both labels.")

      st.image(avg_parasitized, caption='Parasitised Cell - Average and Variability')
      st.image(avg_uninfected, caption='Uninfected Cell - Average and Variability')
      st.write("---")

    if st.checkbox("Differences between average parasitised and average uninfected cells"):
          diff_between_avgs = plt.imread(f"outputs/{version}/avg_diff.png")

          st.warning(
            f"* We notice this study didn't show "
            f"patterns where we could intuitively differentiate one from another.")
          st.image(diff_between_avgs, caption='Difference between average images')

    if st.checkbox("Image Montage"): 
      st.write("* To refresh the montage, click on the 'Create Montage' button")
      my_data_dir = 'inputs/malaria_dataset/cell_images'
      labels = os.listdir(my_data_dir+ '/validation')
      label_to_display = st.selectbox(label="Select label", options=labels, index=0)
      if st.button("Create Montage"):      
        image_montage(dir_path= my_data_dir + '/validation',
                      label_to_display=label_to_display,
                      nrows=8, ncols=3, figsize=(10,25))
      st.write("---")

# One special case on this
# page is image montage for which, if you remember, we have already developed the pseudo-code in the
# data visualization section of the project. We have used that code here in this section and used
# st.pyplot to display this figure instead of plot.show which we did in the jupyter notebooks
# in this workspace.

def image_montage(dir_path, label_to_display, nrows, ncols, figsize=(15,10)):
  sns.set_style("white")
  labels = os.listdir(dir_path)

  # subset the class you are interested to display
  if label_to_display in labels:

    # checks if your montage space is greater than subset size
    # how many images in that folder
    images_list = os.listdir(dir_path+'/'+ label_to_display)
    if nrows * ncols < len(images_list):
      img_idx = random.sample(images_list, nrows * ncols)
    else:
      print(
          f"Decrease nrows or ncols to create your montage. \n"
          f"There are {len(images_list)} in your subset. "
          f"You requested a montage with {nrows * ncols} spaces")
      return
    

    # create list of axes indices based on nrows and ncols
    list_rows= range(0,nrows)
    list_cols= range(0,ncols)
    plot_idx = list(itertools.product(list_rows,list_cols))


    # create a Figure and display images
    fig, axes = plt.subplots(nrows=nrows,ncols=ncols, figsize=figsize)
    for x in range(0,nrows*ncols):
      img = imread(dir_path + '/' + label_to_display + '/' + img_idx[x])
      img_shape = img.shape
      axes[plot_idx[x][0], plot_idx[x][1]].imshow(img)
      axes[plot_idx[x][0], plot_idx[x][1]].set_title(f"Width {img_shape[1]}px x Height {img_shape[0]}px")
      axes[plot_idx[x][0], plot_idx[x][1]].set_xticks([])
      axes[plot_idx[x][0], plot_idx[x][1]].set_yticks([])
    plt.tight_layout()
    
    st.pyplot(fig=fig)
    # we use the above streamlit function instead of = plt.show()


  else:
    print("The label you selected doesn't exist.")
    print(f"The existing options are: {labels}")