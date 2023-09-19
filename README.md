# Diamond Price Prediction
![Diamond](https://imgur.com/a/ifbwhvS)

Welcome to the Diamond Price Prediction project. This project focuses on predicting diamond prices based on a dataset of gemstone characteristics. Accurate diamond price predictions can be invaluable for both buyers and sellers in the jewelry market. This project utilizes machine learning techniques to develop a predictive model and provides a web application for users to make price predictions.

## Table of Contents

- [Data](#data)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Model Training](#model-training)
- [Web Application](#web-application)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Data

This project utilizes several datasets related to gemstone characteristics:

- `gemstone.csv`: The primary raw dataset containing a comprehensive set of gemstone characteristics, including carat weight, cut, color, clarity, depth, table, price, and more.

- `raw.csv`: This dataset is a slightly modified version of `gemstone.csv` and can be used for reference or alternative analysis.

- `test.csv`: A dataset designed for testing the trained prediction model. Users can input gemstone features, and the model will provide price predictions.

- `train.csv`: This dataset is used for training the machine learning model. It contains a subset of the gemstone data with corresponding price labels.

## Exploratory Data Analysis (EDA)

Explore and analyze the gemstone dataset in `EDA.ipynb`. This Jupyter Notebook provides insights into the data, including visualizations, summary statistics, and preprocessing steps. It helps prepare the dataset for model training.

## Model Training

In `Model Training.ipynb`, you'll find the detailed process of training the diamond price prediction model. This Jupyter Notebook covers various steps, such as data preprocessing, feature engineering, model selection, and evaluation. The model is saved as `model.pkl`, and the data preprocessing pipeline is saved as `preprocessor.pkl` within the `artifacts/` directory.

### Artifacts

- `model.pkl`: This is the trained prediction model. It can be used for making predictions on new data.

- `preprocessor.pkl`: The data preprocessing pipeline used to transform input data before making predictions with the model.

## Web Application

The Diamond Price Prediction project includes a web application that allows users to interact with the trained model. The application is implemented in Python using Flask and incorporates machine learning pipelines for real-time predictions.

### Source Code

- `src/` contains the source code for the web application.

  - `components/`: This directory houses custom components used in the web app.

  - `pipeline/`: It contains the data processing and prediction pipeline used by the web application.

  - `templates/`: HTML templates used for rendering the user interface of the web app.

- `application.py` is the main Python file for running the web application. To use the web app, simply run this script.

## Project Structure

The project directory structure is as follows:

- `src/`: Contains the source code for the project.

- `Document/`: Documentation related to the project.

- `.gitignore`: Specifies which files should be ignored by Git.

- `README.md`: You are currently reading this file.

- `git/`: Contains Git-related files, such as hooks or configuration.

- `requirements.txt`: Lists the Python dependencies required to run the project. You can install them using `pip install -r requirements.txt`.

- `setup.py`: Handles project setup and packaging configuration.

## Setup

To set up and run the Diamond Price Prediction project, follow these steps:

1. Clone the repository: `$ git clone https://github.com/yourusername/Diamond-Price-Prediction.git`

2. Navigate to the project directory: `$ cd Diamond-Price-Prediction`

3. Install project dependencies: `$ pip install -r requirements.txt`

4. Run the web application: `$ python application.py`

The web application should be accessible at `http://localhost:5000` in your web browser.

## Usage

To use the Diamond Price Prediction web application:

1. Access the web application by running `python application.py` as mentioned in the setup section.

2. Fill out the gemstone characteristics in the web form.

3. Click the "Predict" button to obtain a diamond price prediction based on the trained model.

Feel free to explore the Jupyter notebooks for EDA and model training for a deeper understanding of the project.

## Contributing

We welcome contributions to this project. If you'd like to contribute, please follow these guidelines:

- Submit bug reports or feature requests through the GitHub issue tracker.

- If you'd like to contribute code, fork the repository, make your changes, and submit a pull request.

- Please adhere to our code of conduct when participating in discussions or contributing to the project.

## License

This project is licensed under the MIT License. For details, please refer to the [LICENSE.md](LICENSE.md) file.
