/* script for users stock table in the database */

CREATE TABLE users_stock (
      ID INT NOT NULL AUTO_INCREMENT,  
      userID INT NOT NULL,
      ticker TEXT NOT NULL,
      total_shares INT,
      FOREIGN KEY (userID) REFERENCES users (ID),
      PRIMARY KEY (ID)
);