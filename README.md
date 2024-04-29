**Bernardo Marquez** (<a href="http://www.linkedin.com/in/bernardo-marquez-1266476" target="_blank" rel="noopener noreferrer">LinkedIn</a>)

This page lists notebooks that I have created in Google Colab as well as scripts that I have created locally to demonstrate various concepts and algorithms in Artificial Intelligence/Machine Learning/Data Science. The notebooks and scripts are currently listed in the order that they were created. When there are already sufficiently many notebooks and scripts in this page, I will organize them by themes.

1. **Pretrained Image Recognition Models**  
Load and run two pretrained image recognition models. The models are called *AlexNet* and *ResNet*. These models were trained on a subset of the *ImageNet* dataset. In the notebook, each model is run on an image that was taken from a *Kaggle* dataset.  
(<a href="https://colab.research.google.com/drive/1sLkTThmtt-2VllfF-XnZ-Wi8rpcYZ26p?usp=sharing" target="_blank" rel="noopener noreferrer">Colab</a> | <a href="https://github.com/bern429/AI-ML-DS-Excursions/blob/main/000_pretrainedImageRecognitionModels/pretrainedImageRecognitionModels.ipynb" target="_blank" rel="noopener noreferrer">GitHub</a>)
   
2. **Cycle GAN**  
Load and run a pretrained *CycleGAN* model. The model was trained on a dataset of horse images and zebra images extracted from the *ImageNet* dataset. In the notebook, the model is run on an image that was taken from a *Kaggle* dataset.  
(<a href="https://colab.research.google.com/drive/13DJuCbOFmlbBpzy4ogVPwEUYRMrDbqYA?usp=sharing" target="_blank" rel="noopener noreferrer">Colab</a> | <a href="https://github.com/bern429/AI-ML-DS-Excursions/blob/main/001_cycleGAN/cycleGAN.ipynb" target="_blank" rel="noopener noreferrer">GitHub</a>)

3. **Table Extraction from PDF**  
Extract tabular data from a pdf document using `tabula-py` and `camelot-py`.  
(<a href="https://colab.research.google.com/drive/1rhVzA4qBoNf4z0NZ0Qb3GOXXjPkKPqmx?usp=sharing" target="_blank" rel="noopener noreferrer">Colab</a> | <a href="https://github.com/bern429/AI-ML-DS-Excursions/blob/main/002_tableExtractionFromPDF/tableExtractionFromPDF.ipynb" target="_blank" rel="noopener noreferrer">GitHub</a>)
 
4. **AutoML**  
Train and test predictive models on the heart disease prediction dataset from *Kaggle* using the following AutoML libraries: `auto-sklearn`, `H2O AutoML`, `PyCaret`, and `TPOT`.  
(<a href="https://colab.research.google.com/drive/1HXI8TcurZD3pB1rhjLWmIeadlNtVPSXq?usp=sharing" target="_blank" rel="noopener noreferrer">Colab</a> | <a href="https://github.com/bern429/AI-ML-DS-Excursions/blob/main/003_autoML/autoML.ipynb" target="_blank" rel="noopener noreferrer">GitHub</a>)

5. **Data Standardization**  
Standardize data using basic `pandas` operations or using the `fit`, `transform`, and `fit_transform` methods of `sklearn.preprocessing.StandardScaler()`.  
(<a href="https://colab.research.google.com/drive/1J_WzeYM6ySVN0JyjgvQwoRLp-MAPtwhx?usp=sharing" target="_blank" rel="noopener noreferrer">Colab</a> | <a href="https://github.com/bern429/AI-ML-DS-Excursions/blob/main/004_dataStandardization/dataStandardization.ipynb" target="_blank" rel="noopener noreferrer">GitHub</a>)
   
6. **Tensor Representation**  
Load different types of data and represent them as tensors.  
(<a href="https://colab.research.google.com/drive/1anyc1ZZyqe41KDik8Mo95i9kLdNtUcao?usp=sharing" target="_blank" rel="noopener noreferrer">Colab</a> | <a href="https://github.com/bern429/AI-ML-DS-Excursions/blob/main/005_tensorRepresentation/tensorRepresentation.ipynb" target="_blank" rel="noopener noreferrer">GitHub</a>)

7. **Forward and Backward Propagation**  
Use a sequence of forward and backward propagation to estimate the intercept and coefficient of a simple linear regression model. Data is first generated from a specified simple linear regression model. The simulated data is then passed to a sequence of alternating forward and backward propagation.  
(<a href="https://colab.research.google.com/drive/10qo5H_ZvA4Lq5dKXN-YVYNv89x_WsMQF?usp=sharing" target="_blank" rel="noopener noreferrer">Colab</a> | <a href="https://github.com/bern429/AI-ML-DS-Excursions/blob/main/006_forwardBackwardPropagation/forwardBackwardPropagation.ipynb" target="_blank" rel="noopener noreferrer">GitHub</a>)
  
8. **Neural Network on Synthetic Data**  
Fit a neural network to synthetic training data. Data is first generated from a specified univariate function. The synthetic data is split into training and test sets, which are then passed to a sequence of alternating forward and backward propagation.  
(<a href="https://colab.research.google.com/drive/1XGyd3w9uujVBdEOY95ocCOzBnh3_2Fn0?usp=sharing" target="_blank" rel="noopener noreferrer">Colab</a> | <a href="https://github.com/bern429/AI-ML-DS-Excursions/blob/main/007_neuralNetworkOnSyntheticData/neuralNetworkOnSyntheticData.ipynb" target="_blank" rel="noopener noreferrer">GitHub</a>)
  
9. **Feature Importance**  
Display different types of feature importance after training predictive models on the bank customer churn dataset from *Kaggle*. The feature importance scores are calculated based on one of the following: absolute value of standardized coefficients, mean decrease in Gini impurity, permutation importance, and `SHAP` values.  
(<a href="https://colab.research.google.com/drive/1V7_oIL2TZvZkAdQbMvQtkgspHvsStgoY?usp=sharing" target="_blank" rel="noopener noreferrer">Colab</a> | <a href="https://github.com/bern429/AI-ML-DS-Excursions/blob/main/008_featureImportance/featureImportance.ipynb" target="_blank" rel="noopener noreferrer">GitHub</a>)
  
10. **SHAP**  
Display different types of `SHAP` plots after training an XGBoost regression model on the Boston house prices dataset from *Kaggle*.  
(<a href="https://colab.research.google.com/drive/1DPyuYi0A1mLqzJsQ594_zFz3nVoeF76s?usp=sharing" target="_blank" rel="noopener noreferrer">Colab</a> | <a href="https://github.com/bern429/AI-ML-DS-Excursions/blob/main/009_shap/shap.ipynb" target="_blank" rel="noopener noreferrer">GitHub</a>)

11. **Dominance Analysis**  
Determine the relative importance of predictors using dominance analysis when fitting a multiple regression model to the insurance premium prediction dataset from *Kaggle* to predict expenses.  
(<a href="https://colab.research.google.com/drive/19oVz5iEv36a-dyl1-ST-SUa8lnb6w4dH?usp=sharing" target="_blank" rel="noopener noreferrer">Colab</a> | <a href="https://github.com/bern429/AI-ML-DS-Excursions/blob/main/010_dominanceAnalysis/dominanceAnalysis.ipynb" target="_blank" rel="noopener noreferrer">GitHub</a>)

12. **Image Recognition on CIFAR-10**  
Train and test a fully connected neural network to recognize images in a subset of the *CIFAR-10* dataset.  
(<a href="https://colab.research.google.com/drive/1nMeL5ID6UBGzKAAWdB9MEB1K2oJ7Vy6I?usp=sharing" target="_blank" rel="noopener noreferrer">Colab</a> | <a href="https://github.com/bern429/AI-ML-DS-Excursions/blob/main/011_imageRecognitionOnCIFAR/imageRecognitionOnCIFAR.ipynb" target="_blank" rel="noopener noreferrer">GitHub</a>)

13. **Anomaly Detection on Synthetic Data**  
Run anomaly detection algorithms on synthetic data using `scikit-learn` and `pyod`.  
(<a href="https://colab.research.google.com/drive/1KO9yFX2ap08z91AmCb9OcxZVdO8CA_LA?usp=[sharing](https://colab.research.google.com/drive/11aF1g7O86iB9VvIglxZwHWQmkanSw8cm?usp=sharing)" target="_blank" rel="noopener noreferrer">Colab</a> | <a href="https://github.com/bmarquez429/AI-ML-DS-Excursions/blob/main/012_anomalyDetectionOnSyntheticData/anomalyDetectionOnSyntheticData.ipynb" target="_blank" rel="noopener noreferrer">GitHub</a>)

14. **Anomaly Detection on Healthcare Providers Data**  
Run anomaly detection algorithms on healthcare providers data from *Kaggle*.  
(<a href="https://colab.research.google.com/drive/1JIqCZJ1q9Kt4XMe01WCjbdFieeveSaeI?usp=sharing" target="_blank" rel="noopener noreferrer">Colab</a> | <a href="https://github.com/bmarquez429/AI-ML-DS-Excursions/blob/main/013_anomalyDetectionOnHealthcareProvidersData/anomalyDetectionOnHealthcareProvidersData.ipynb" target="_blank" rel="noopener noreferrer">GitHub</a>)

15. **Regular Expressions**  
Provide examples of working with regular expressions using `re`.  
(<a href="https://colab.research.google.com/drive/1YTfuakqArnW37PXgwu2lfarS5DXw40j5?usp=sharing" target="_blank" rel="noopener noreferrer">Colab</a> | <a href="https://github.com/bmarquez429/AI-ML-DS-Excursions/blob/main/014_regularExpressions/regularExpressions.ipynb" target="_blank" rel="noopener noreferrer">GitHub</a>)

16. **Classification App**  
Walk the user through the steps in training and testing one or more binary classifiers using a selection of algorithms that are implemented in scikit-learn.  
(<a href="https://scikit-learnclassification.streamlit.app/" target="_blank" rel="noopener noreferrer">Streamlit</a> | <a href="https://github.com/bmarquez429/Classification-App" target="_blank" rel="noopener noreferrer">GitHub</a>)

