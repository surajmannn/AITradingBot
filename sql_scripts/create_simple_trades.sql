/* simple version of script for trades table in database
    Simplified version for bot principle rather than
    user software version */

/*
      position = 0 if position closed
      position = 1 if buy order
      position = 2 if sell order

      mla - Machine Learning Algorithm
      mla = MLP for Multi Layer Perceptron
      mla = SVM for Support Vector Machine
      mla = LSTM for Long Short Term Memory
*/

CREATE TABLE simple_trades (
      ID INT NOT NULL AUTO_INCREMENT,  
      userID INT NOT NULL,
      ML_type TEXT,
      ticker TEXT,
      position INT,
      quantity INT,
      stock_price FLOAT,
      total_price FLOAT,
      purchase_date DATETIME DEFAULT CURRENT_TIMESTAMP,
      RSI FLOAT,
      BB_upper_band FLOAT,
      BB_lower_band FLOAT,
      ADX FLOAT,
      DI_Pos FLOAT,
      DI_neg FLOAT,
      confidence_rating FLOAT,
      FOREIGN KEY (userID) REFERENCES users (ID),
      PRIMARY KEY (ID)
);