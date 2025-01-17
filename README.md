<a href="https://anaconda.org/"><img src="https://img.shields.io/static/v1?label=win-64&message=Anaconda.org&color=lightgreen&link=https://anaconda.org/"/></a>
<a href="https://github.com/AlexYoussef/RapiD_AI/blob/main/LICENSE"><img src="https://img.shields.io/static/v1?label=LICENSE&message=MIT&color=informational&link=https://github.com/AlexYoussef/RapiD_AI/blob/main/LICENSE"/>
<a href="https://keras.io/getting_started/"><img src="https://img.shields.io/static/v1?label=TensorFlow&message=Keras v2.6.0 &color=red&link=https://keras.io/getting_started/"/>
<a href="https://scikit-learn.org/stable/index.html"><img src="https://img.shields.io/static/v1?label=scikit-learn&message=v0.24.2&color=blue&link=https://scikit-learn.org/stable/index.html" /></a>
<a href="https://pandas.pydata.org/"><img src="https://img.shields.io/static/v1?label=pandas&message=v1.1.3.0&color=blueviolet&link=https://pandas.pydata.org/"/></a>
<a href="https://scipy.org/"><img src="https://img.shields.io/static/v1?label=SciPy&message=v1.5.2&color=informational&link=https://scipy.org/"/></a>
<a href="https://matplotlib.org/"><img src="https://img.shields.io/static/v1?label=Matplotlib&message=v3.2.1&color=yellow&link=https://matplotlib.org/"/></a>
<a href="https://pytorch.org/get-started/locally/"><img src="https://img.shields.io/static/v1?label=PyTorch&message=v1.12.1&color=red&link=https://pytorch.org/get-started/locally/"/></a>
<a href="https://pypi.org/project/pytorch-tabnet/"><img src="https://img.shields.io/static/v1?label=pytorch_tabnet&message=v4.0&color=red&link=https://pypi.org/project/pytorch-tabnet/"/></a>
<a href="https://xgboost.readthedocs.io/en/latest/index.html"><img src="https://img.shields.io/static/v1?label=xgboost&message=v1.6.2&color=9cf&link=https://xgboost.readthedocs.io/en/latest/index.html"/></a>
 
# RapiD_AI: A framework for Rapidly Deployable AI for novel disease \& pandemic preparedness

## Overview
COVID-19 is unlikely to be the last pandemic that we face. According to an analysis of a global dataset of historical pandemics from 1600 to the present, the risk of a COVID-like pandemic has been estimated as 2.63\% annually or a 38\% lifetime probability. This rate may double over the coming decades. While we may be unable to prevent future pandemics, we can reduce their impact by investing in preparedness. In this study, we demonstrate the value of transfer learning and pretrained neural network models as a pandemic preparedness tool to enable healthcare system resilience and effective use of Machine Learning during future pandemics. RapiD\_AI demonstrates the utility of transfer learning to build high-performing ML models using data collected in the first weeks of the pandemic and provides an approach to adapt the models to the local populations and healthcare needs. The motivation is to enable healthcare systems to overcome data limitations that prevent the development of effective ML in the context of novel diseases. We digitally recreated the first 20 weeks of the COVID-19 pandemic and experimentally demonstrated the utility of transfer learning using domain adaptation and inductive transfer. We (i) pretrain two neural network models (Deep Neural Network and TabNet) on a large Electronic Health Records dataset representative of a general in-patient population in Oxford, UK, (ii) fine-tune using data from the first weeks of the pandemic, and (iii) simulate local deployment by testing the performance of the models on a held-out test dataset of COVID-19 patients. Our approach has demonstrated an average relative/absolute gain of 4.92/4.21\% AUC compared to an XGBoost benchmark model trained on COVID-19 data only. Moreover, we show our ability to identify the most useful historical pretraining samples through clustering and to expand the task of deployed models through inductive transfer to meet the emerging needs of a healthcare system without access to large historical pretraining datasets.

## System Requirements
 ### Hardware requirements
   `RapiD_AI` requires a computer device with enough RAM and GPU to support the in-memory operations and model training and inferance scenarios.
 ### Software requirements
  The package has been tested on the following systems:
   * Microsoft Windows 10 64 bit
 ### Python Dependencies
 `RapiD_AI` runs on `Python 3.7+`
 
 `RapiD_AI` deepnds on the following python libraries and APIs 
   * `numpy`
   * `pandas`
   * `scikit-learn`
   * `scipy`
   * `Keras`
   * `matplotlib`
   * `PyTorch`
   * `pytorch_tabnet`
   * `xgboost`
 

## Installation Guide
### Clone the `RapiD_AI` repo to your own machine using:
   `git clone https://github.com/AlexYoussef/RapiD_AI.git`
### Create a new envronemnet in `Anaconda`: To run all the codes and notebooks, use the `requirements.txt` file to create an environment as follows:
    `$ conda create --name <env_name> --file requirements.txt`

### If you come across any challenges finding some of packages in conda channels, please freeze the `requirements.txt` and install the packages using using `pip` tool as follows:
 
`pip freeze > requirements.txt`
 
`pip install -r requirements.txt`
 
If there are still missing packages, you will have to install them manually using `pip install <package_name>`

### When you clone the repo, make sure to add the repo project to your system path envireonment variable
  This can be done in each notebook as follows:
 
  `from sys import path as pylib`
 
 `import os`
 
 `pylib += [os.path.abspath('path/to/repo/location/on/your/local/device')]`

 Add the three lines at first of each notebook you run.
 
## Link to full paper:
   To be added upon publication

##  Documenation and Code notes
- Jupyter notebooks used to train and validate our proposed models are available in [main](main/)
- Scripts for pretraining `TabNet` and `DNN` models are  available [here](transfer_learning/) 
- Data used in our work is not publically available. For access requests please refer to: https://oxfordbrc.nihr.ac.uk/research-themes-overview/antimicrobial-resistance-and-modernising-microbiology/infections-in-oxfordshire-research-database-iord/

- To test the code, we provided template dataset (refer to `data` folder) and pretrained models on this data (`transfer_learning/pretrained_models`).  
- To run the code on the uploaded template data, pleas navigate to the `main` folder where you can find the three adopted scenarios: A, B and C.
- For better understanding the whole process, please refer to the provided pseudocode [here](https://github.com/AlexYoussef/RapiD_AI/blob/main/RapiD_AI%20PSEUDOCODE.txt)

## License
This project is covered under the [MIT License](https://github.com/AlexYoussef/RapiD_AI/blob/main/LICENSE).

