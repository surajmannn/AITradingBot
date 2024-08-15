/* simple version of script for machine learning algorithm table in database
    Simplified version for bot principle rather than
    user software version */

/*
    For each model training period, 
    writes performance metrics to the table.

    mla - Machine Learning Algorithm
    mla = 1 for Multi Layer Perceptron
    mla = 2 for Support Vector Machine
    mla = 3 for Random Forest
    mla = 4 for Long Short Term Memory
*/

CREATE TABLE ml_metrics (
    ID INT NOT NULL AUTO_INCREMENT,
    userID INT NOT NULL,
    ticker TEXT,  
    ml_model INT,
    training_start_date TEXT,
    training_end_date TEXT,
    accuracy FLOAT,
    ROC_AUC FLOAT,
    FOREIGN KEY (userID) REFERENCES users (ID),
    PRIMARY KEY (ID)
);