import re
from math import *

#for enhancement: enint: integrals, errorint: errors

# Enter in the number of enhancement expts + 1 on line 19 (where the 18's are)
# Enter int the number of T1 experiments + 1 on line

import csv



#Enhancement Powers and # Enhancement Integrals
list3=[]

with open('enhancementPowers.csv', 'rb') as csvfile2:
	reader2 = csv.reader(csvfile2)
	list2 = list(reader2)

#Enter in number of experiments + 1 for titles
	list3[0:18] = list2[0:18]
	


enpower,enint,exptnumber2 = zip(*list3)

enpower2= []

enint2 = []

enpower3 = []

enint2[0:16] = enint[1:18]


enpower2[0:16] = enpower[1:18]


outpower = open("en_power_final.txt","w")

outint = open("en_int_final.txt","w")

for item in enint2:
	outint.write("%s\n" % item)

for item in enpower2:
  outpower.write("%s\n" % item)
  

outint.close()


outpower.close()

csvfile2.close()

#T1's

t1list=[]
t1list2 = []

with open('t1Powers.csv', 'rb') as csvfile3:
	reader3 = csv.reader(csvfile3)
	t1list = list(reader3)

#Enter in number of experiments + 1 for titles
	t1list2[0:12] = t1list[0:12]

	


t1power,t1int,t1error, t1exptnumber = zip(*t1list2)

t1err = open('t1err.txt',"w")

t1err.write(t1error[-1])
t1err.close()

t1power2= []
t1int2 = []
t1int4 = []

t1power2[0:11] = t1power[1:12]
t1int2[0:11] = t1int[1:12]





t1power3 = t1power2[::-1]
t1int4 = t1int2[::-1]


outt1power = open("t1_power_final.txt","w")

for item in t1power3:
  outt1power.write("%s\n" % item)

outt1integral = open("t1_int_final.txt","w")

for item in t1int4:
	outt1integral.write("%s\n" % item)
	


	
	

outt1power.close()
outt1integral.close()

csvfile2.close()






		



