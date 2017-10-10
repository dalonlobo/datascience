# I'm setting the Execution engine to mapreduce and not tez
SET hive.execution.engine=mr;

# First I'm creating a table called 'nyse_data', and into this table I'll pull the data from the files
CREATE TABLE nyse_data(stockexc string,
                       company string,
                       dt string,
                       openprice float, 
                       highprice float, 
                       lowprice float, 
                       closeprice float, 
                       volume int, 
                       adjustedprice float) ROW format delimited fields terminated BY '\t' stored AS textfile;

LOAD DATA inpath '/user/dalonlobo2857/BDH_Lab_R4' INTO TABLE nyse_data;

# Creating the following table which is partitioned on date field and also its stored as ORC file
CREATE TABLE nyse_data_partitioned(stockexc string,
                                   company string,
                                   openprice float, 
                                   highprice float, 
                                   lowprice float, 
                                   closeprice float, 
                                   volume int, 
                                   adjustedprice float) partitioned BY (dt string) ROW format delimited fields terminated BY '\t' stored AS orcfile;

# Here the data from staged table nyse_data is copied to nyse_data_partitioned table
FROM nyse_data
INSERT INTO TABLE nyse_data_partitioned partition(dt)
SELECT stockexc,
       company,
       dt,
       openprice,
       highprice,
       lowprice,
       closeprice,
       volume,
       adjustedprice
WHERE openprice > 68.0
  AND highprice < 70.0;

# Just a debugging step
SELECT count(*) FROM nyse_data_partitioned;
# The output of above query was 79

# Below query will save the output of the query to the specified path
INSERT OVERWRITE DIRECTORY '/user/dalonlobo2857/BDH_Lab_R4/output' SELECT * FROM nyse_data_partitioned;
