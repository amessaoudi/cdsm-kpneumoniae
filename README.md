CDCSM-KPNEUMONIAE

This repository contains the code and pre-trained models for antimicrobial resistance (AMR) prediction and clinical decision support in Klebsiella pneumoniae. It is part of the manuscript submitted to Journal of Infection and Public Health.

Cloning the Repository and Running the Project

To clone this repository to your local machine, open the terminal/command prompt and run the following command:

git clone https://github.com/amessaoudi/cdsm-kpneumoniae.git
cd cdsm-kpneumoniae
pip3 install -r requirements.txt
Running the Main Pipeline

To train and evaluate the models:

python3 src/train.py
python3 src/evaluate.py
Running the Clinical Decision Support Module (CDSM)
python3 src/cdsm.py
Running the Streamlit Web Application
streamlit run app/streamlit_app.py
