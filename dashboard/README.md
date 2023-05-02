# Dashboard Deployment Instructions
In order to run the dashboard, you will first need to unzip the file 'dashboard_multivariate_set.zip' and ensure the file inside is saved in this directory with the title 'dashboard_multivariate_set.csv'.
The other csv files in this directory must also be included and have the correct permissions to be read by the dashboard script. As long as you then install of the dependencies listed below,
simply running the python script should provide you with a dashboard running on port 8050 on localhost. It has been tested to work on Python version 3.8.10.
## Dependencies
The following Python packages must be installed to use this dashboard:
* pandas
* dash
* dash_daq
* numpy
* dash_bootstrap_templates
* hausdorff
* math
* plotly
