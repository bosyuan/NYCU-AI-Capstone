# Facial Age Recognition
  This project implements multiple facial age classifiers implementing ML approaches including K-Means, Logistic Regression, and Res-Net.

## Dataset
  The given dataset collected contains 800 facial images of 160 subjects (public figures). With a labeled age span ranging from 20 to 60 years old, each subject is captured five times, and labeled according to the corresponding age of the subject.

## To Run
  1. Pre collected dataset is already collected and stored under directory <code>/dataset</code>, age is labeled under direcory name:
       dataset
     
          |---21
          |---22
          .
          .
          .
          |---59
  3. After checking the correct configuration of the dataset, tests of different methods could be run by below commands:
     ```
     python KMean.py
     python LogisticRegression.py
     python ResNet.py
Further detail of results and corresponding figures could be seen in [report.docx](https://github.com/bosyuan/NYCU-AI-Capstone/blob/d8f684d269d86190b8788c3d32a162d2065c004f/facial%20age%20detection/report.docx)
