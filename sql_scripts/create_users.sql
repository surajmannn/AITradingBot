/* script for users table in database */

CREATE TABLE users (
      ID INT NOT NULL AUTO_INCREMENT,  
      username VARCHAR(16) UNIQUE NOT NULL,
      PRIMARY KEY (ID)
);

/* Insert users */
INSERT INTO users (username) VALUES
      ("test1");