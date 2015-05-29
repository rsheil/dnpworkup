# dnpworkup
python program for working up dnp data

Instructions:
copy in both python files (formatflip.py & dnp_rs.py) into output folder from returnintegrals.py (Ryan B's script)
copy over your t1series.csv from T10 data folder and rename t10series.csv
if you took multiple T1's without power copy over their t1series.csv and name it t12series.csv
open formatflip.py and enter in the number of t1 and enhancement experiments
run formatflip.py
open dnp_rs.py
follow instructions to add title of output and experiment info
enter if you have extra t10 or t1(0) data sets and how you want to calculate error for krho
run dnp_rs.py
output pdf will be created with relavent values calculated and displayed. 

Note: this set up is currently jenky, I will fix to run one script and to not create a bunch of txt files, but too lazy
