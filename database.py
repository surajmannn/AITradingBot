""" Creates connection to database and funtion for sending queries to database """

import mysql.connector
from mysql.connector import Error

# Create connection to database
config = {'user' : 'root', 
            'password' : 'localhost',
            'host' : '127.0.0.1',
            'database' : 'AI_trading_bot'}


def run_value_query(query, values):
    try:
        cnx = mysql.connector.connect(**config)
        cursor = cnx.cursor()
        cursor.execute(query, (values,))
        data = cursor.fetchall()
        cnx.close()
        return data
    except Error as err:
        print(f"Error: '{err}'")


def run_multiple_value_query(query, values):
    try:
        cnx = mysql.connector.connect(**config)
        cursor = cnx.cursor()
        cursor.execute(query, values)
        data = cursor.fetchall()
        cnx.close()
        return data
    except Error as err:
        print(f"Error: '{err}'")


def run_all_query(query):
    try:
        cnx = mysql.connector.connect(**config)
        cursor = cnx.cursor()
        cursor.execute(query)
        data = cursor.fetchall()
        cnx.close()
        return data
    except Error as err:
        print(f"Error: '{err}'")


def run_alter_query(query, values):
    try:
        cnx = mysql.connector.connect(**config)
        cursor = cnx.cursor()
        cursor.execute(query, values)
        cnx.commit()
        cnx.close()
        return(query, " was completed successfully")
    except Error as err:
        print(f"Error: '{err}'")