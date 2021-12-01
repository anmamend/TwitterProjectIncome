# Income Predictor for Colombian tweets

This project test different Machine Learning algorithms to predict if a tweet in Colombia is for a high-income user or for a low-income user.
The data used corresponds to tweets made in different zones in Colombia, labeled as high income zones or low income zones. The data was collecte between september and november of 2020.


## Getting Started 🚀

Feel free to download a copy of this project or the data contained in the project to recreate it.
See deployment for notes on how to deploy the project on a live system.


### Prerequisites 📋

This project needs the following Python libraries in order of being executed correctly:
Sagemaker
xgboost
nltk
beautifulsoup4
html5lib
pandas
numpy
sklearn
torch
emoji
sentiment_analysis_spanish
wordcloud
matplotlib
collections
PIL
re
squarify
matplotlib
seaborn

To install this libraries is necesarry to import them directly from pip

```
!pip install squarify
import matplotlib.pyplot as plt
import squarify #
```


## Deployment📦

To deploy the final model, it was created an endpoint with AWS Sagemaker. This endpoint needs to be connected to lambda function and AWS Gateway Proxy in order to be able to built the web-app.

## Built With 🛠️

* Sagemaker (https://us-east-2.console.aws.amazon.com/sagemaker/home?region=us-east-2#/notebook-instances) 


## Wiki 📖

Read Report.pdf to understan what is made in the project


## Authors✒️


* **Andrés Mendoza** - *Project* -



## License📄

Feel free to download and improve the problem

## Acknowledgments🎁

* Thanks for all people who developed the libraries used in this project


