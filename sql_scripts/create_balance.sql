/* script for balance table in database*/

CREATE TABLE balance (
      ID INT NOT NULL AUTO_INCREMENT,  
      userID INT NOT NULL,
      initial_balance DECIMAL(15,2),
      current_balance DECIMAL(15,2),
      stock_balance DECIMAL(15,2) DEFAULT 0,
      FOREIGN KEY (userID) REFERENCES users (ID),
      PRIMARY KEY (ID)
);