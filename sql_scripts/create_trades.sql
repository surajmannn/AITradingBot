/* script for trades table in database */

/*
      position = 1 if buy order
      position = 2 if sell order

      status = 1 if position is open
      status = 2 if position is closed
*/

CREATE TABLE trades (
      ID INT NOT NULL AUTO_INCREMENT,  
      userID INT NOT NULL,
      ticker TEXT NOT NULL,
      position INT NOT NULL,
      quantity INT NOT NULL,
      entry_stock_price DECIMAL NOT NULL,
      entry_total_price DECIMAL NOT NULL,
      purchase_date DATETIME DEFAULT CURRENT_TIMESTAMP,
      status INT NOT NULL,
      close_date DATETIME,
      exit_stock_price DECIMAL,
      exit_total_price DECIMAL,
      FOREIGN KEY (userID) REFERENCES users (ID),
      PRIMARY KEY (ID)
);