/* simple version of script for trades table in database
    Simplified version for bot principle rather than
    user software version */

/*
      position = 0 if position closed
      position = 1 if buy order
      position = -1 if sell order
      position = -2 for stop loss close

      mla - Machine Learning Algorithm
      mla = 1 for Multi Layer Perceptron
      mla = 2 for Support Vector Machine
      mla = 3 for Random Forest
      mla = 4 for Long Short Term Memory
*/

CREATE TABLE simple_trades (
      ID INT NOT NULL AUTO_INCREMENT,  
      userID INT NOT NULL,
      ticker TEXT,
      mla INT,
      position_type INT,
      quantity INT,
      security_price FLOAT,
      total_price FLOAT,
      profit FLOAT,
      balance FLOAT,
      purchase_date TEXT,
      BB_upper_band FLOAT,
      BB_lower_band FLOAT,
      RSI FLOAT,
      ADX FLOAT,
      DI_Pos FLOAT,
      DI_neg FLOAT,
      volatility FLOAT,
      confidence_probability FLOAT,
      FOREIGN KEY (userID) REFERENCES users (ID),
      PRIMARY KEY (ID)
);