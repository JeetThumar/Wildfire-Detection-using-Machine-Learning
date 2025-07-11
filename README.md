# Wildfire Detection Using Machine Learning

## Overview
This project develops a machine learning model to detect wildfires in real-time using satellite data. It leverages a Random Forest Classifier with features such as surface temperature, infrared (IR) bands, vegetation index, humidity, wind speed, and elevation, achieving an accuracy of 93.5% and an ROC-AUC score of 0.96.

## Features
- **Real-time Prediction**: Analyzes satellite data for timely wildfire detection.
- **Key Features**: IR bands, temperature, wind speed, elevation, slope.
- **Performance**: 93.5% accuracy, 94% recall, 0.96 ROC-AUC.
- **EDA**: Includes correlation heatmaps, boxplots, and KDE plots.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/JeetThumar/Wildfire-Detection-using-Machine-Learning.git
2. Install dependencies:
   pip install -r requirements.txt
3. Set up environment: Requires Python 3.11+.
4. Run the notebook: Open Wildfire_Detection.ipynb in Jupyter Notebook.
   
## Usage
- Data Preparation: Load and preprocess the dataset (e.g., synthetic_wildfire_data.csv).
- Model Training: Train the Random Forest Classifier with hyperparameter tuning.
- Prediction: Input new data to predict wildfire risk.
- See Wildfire_Detection.ipynb for detailed steps.
  
## Project Structure
Wildfire-Detection-using-Machine-Learning/
- Wildfire_Detection.ipynb       # Main notebook
- requirements.txt               # Dependencies
- README.md                      # This file
- .gitignore                     # Ignore files
- LICENSE                        # License
  
**Documentation**
- Project_On_Wildfire_Detection.pdf
- Wildfire_Detection_Documentation.pdf
   
## Results
- Accuracy: 93.5%
- Precision: 91%
- Recall: 94%
- F1-Score: 92%
- ROC-AUC: 0.96
- Key Features: ir_band_2, wind_speed, elevation, slope (via RFE).
  
## Future Work
- Integrate real-time satellite APIs.
- Develop a web or mobile dashboard.
- Explore time-series forecasting.
  
## Contributing
Contributions are welcome! Fork the repo, create a branch, and submit a Pull Request.

## License
MIT

## Author
Jeet Thumar
