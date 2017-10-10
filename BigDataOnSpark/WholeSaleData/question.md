Data Set : Wholesale customers data.csvView in a new window

The data set refers to clients of a wholesale distributor. It includes the annual spending in monetary units (m.u.) on diverse product categories.

The data set is also available at [here](https://archive.ics.uci.edu/ml/datasets/Wholesale+customers).

 

Description of variables is as folllows:

    FRESH: annual spending (m.u.) on fresh products (Continuous);
    MILK: annual spending (m.u.) on milk products (Continuous);
    GROCERY: annual spending (m.u.)on grocery products (Continuous);
    FROZEN: annual spending (m.u.)on frozen products (Continuous)
    DETERGENTS_PAPER: annual spending (m.u.) on detergents and paper products (Continuous)
    DELICATESSEN: annual spending (m.u.)on and delicatessen products (Continuous);
    CHANNEL: customers Channel - Horeca (Hotel/Restaurant/Cafe) or Retail channel (Nominal)
    REGION: customers Region Lisnon, Oporto or Other (Nominal)

 

The dataset gives data about sales of 6 category of products across 3 regions via 2 channel. 

Please do the following task on dataset given above.

    - Read the csv file as a dataframe(DF). 
    - See the schema of the DF.
    - Use select to view a single column or a set of chosen columns.
    - Use filter to see records with fresh sales more than 50000 only.
    - Create aggregates on channels and regions variables.
    - Use describe to see summary statistics on dataframe.
    - Change datatype of Channels to Strings.
    - Perform rollups on channels and regions.

