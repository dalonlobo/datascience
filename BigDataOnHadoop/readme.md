About Data Set :

The NYSE stock Data set which can also be downloaded from the HDFS file browser from Hue in Big Data Lab. The path of file is : “/data/NYSE_daily_File”.

NYSE dataset Daily stock data of each company is available live on yahoo finance for each stock exchange worldwide. We have taken the NYSE stock exchange data for this study. The data set is composed of: stock exchange,company symbol, date, open price of the day, high of the day, low of the day, close of the day , volume and adjusted close price.

Problem Statement :

Please complete following tasks based on the data set mentioned above:

-  Create NYSE_Partition table based on the date field.
-  This table needs to store the data as ORC file.
-  Load data of only those records where open price of the day is greater than 68 and high price of the day is less than 70.

