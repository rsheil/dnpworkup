# Python version of mathematica work-up of DNP data from Ryan's CNSI workbook
# Run format flip first to extract data in correct format
#enter in T10, title of output file, and SL concentration prior to running

from scipy import stats
from pylab import *

from reportlab.graphics.charts.linecharts import HorizontalLineChart
from reportlab.graphics.shapes import *
from reportlab.graphics.charts import *
from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.graphics.widgets.markers import makeMarker
from reportlab.graphics.charts.linecharts import HorizontalLineChart


from reportlab.platypus import *
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.rl_config import defaultPageSize
from reportlab.lib.units import inch
from reportlab.lib.styles import ParagraphStyle

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pylab import *
import matplotlib.image as mpimg
from scipy.optimize import curve_fit

import re
from math import *
import csv
import numpy


import time

localtime = time.asctime( time.localtime(time.time()) )
print "Local current time :", localtime

# set t1extra = True if you want to import extra T1(0)'s from T12series.csv
# set t1extra = False if you want to use fitting error for T1(0) error
# set t1extra = 1 if you want to use T10 error for T1(0)
# set t1extra = 2 if you want to manually enter in error for T1(0) (stdevt1 = "error" below)
t1extra = False
stdevt1 = 0


#---------T10---(extra T10's)------------------
#if you want to import T10series.csv then keep imp = True
#if you want to manually enter in T10 enter list in T10, but then make imp = False

t10 = [1.256]
imp = True



# Enter in date of experiment
date = '2015_05_18'

# Enter in title of pdf for output 

def go():
    doc = SimpleDocTemplate('2015_05_18_sigma_sio2_ph4p5.pdf') 
    doc.build(Elements)
    
# Enter in the experiment name

Exptname = '2015_05_18_sigma_sio2_ph4p5'
    

# set SL concentration in Molar
#concentration only affect ksigma,krho but cancels out in tau and couplingcnst
conc = float(500 * 10 ** (-6))

# Name
Name = 'Ryan Sheil'






#----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------


# open text file of enhancement power list without nopower
with open("en_power_final.txt") as enp:
    enpower = enp.read().splitlines()

enpower_f = []

for item in enpower:
	enpower_f.append(float(item))
	
    

# open text file of enhancement integrals without nopower
with open("en_int_final.txt") as enint:
    en_integral = enint.read().splitlines()

en_integral_f = []

for item in en_integral:
	en_integral_f.append(float(item))
	

enint.close()

#This gives the x and y of the enhancement data
enhancement_data = zip(enpower_f, en_integral_f)

print "----------" *5
print "----------" *5
print "This is the enhancement data:"
print "----------" *5
print "----------" *5
print "\n"



Lengthen = len(en_integral_f)



for c1, c2 in zip(enpower_f, en_integral_f):
    print "%-9s %s" % (c1, c2)


print "\n"
print "There are %r experiments..." % Lengthen
print "\n"


del enpower_f[0]
del en_integral_f[0]

#open text file of t1 power list and t1 time list

with open("t1_power_final.txt") as t1p:
	t1power = t1p.read().splitlines()

t1power_f = []
t1_integral_f = []

for item in t1power:
	t1power_f.append(float(item))
	
with open("t1_int_final.txt") as t1int:
	t1_integral = t1int.read().splitlines()

for item in t1_integral:
	t1_integral_f.append(float(item))
	
t1p.close()
t1int.close()

T1_data = zip(t1power_f, t1_integral_f)

print "----------" *5
print "----------" *5
print "This is the T1 data:"
print "----------" *5
print "----------" *5
print "\n"

for c3, c4 in zip(t1power_f, t1_integral_f):
    print "%-9s %s" % (c3, c4)


len3 = len(t1_integral_f)

print "\n"
print "There are %r experiments..." % len3


#This set the first T1 power to 0
t1power_f[0] = 0



#This finds the max of the T1 powers
maxt1p = max(t1power_f)



relT1power = []

#This divides all T1 powers by max t1 power
for i in t1power_f:
	relT1power.append(i)
	

relT1power_nop = []

relT1power_nop[0:] = relT1power[1:]

t1_integral_f_nop = []
t1int3 = []

t1powerplot = []
t1powerplot[0:] = t1power_f[0:]
	

t1_integral_f_nop[0:] = t1_integral_f[1:]
	
for i in t1_integral_f:	
	t1int3.append(float(1/i))


relT1data_nop = zip(relT1power_nop,t1_integral_f_nop)

#relative powers (not including no power) and all corresponding T1's
print "----------" *5
print "----------" *5
print "This is the T10 data:"
print "----------" *5
print "----------" *5
print "\n"

t10state = 'dummy'
t10_value = 'dummy'
stdev = 'dummy'

if imp == True:
	print "Importing T10data from T10series.csv"
	with open('T10series.csv', 'rb') as t10csv:
		reader2 = csv.reader(t10csv)
		t10list = list(reader2)
	
	

	t1val, t10err, expnum = zip(*t10list)
	
	for c8, c9 in zip(t1val, t10err):
		print "%-9s %s" % (c8, c9)




	T10li = []
	T10li[0:]= t1val[1:]
	T10vals = []

	for i in T10li:
		T10vals.append(float(i))

	t10_value = float(sum(T10vals)/(len(T10vals)))

	

	T10arr = numpy.array(T10vals)
	stdev = numpy.std(T10arr)
	
	


	t10csv.close()
	
elif imp == False:
	if len(t10)== 1:
		t10_value = t10[0]
		stdev = 0
	elif len(t10) > 1:
		t10_value = float(sum(t10)/len(t10))
		T10arr = numpy.array(t10)
		stdev = numpy.std(T10arr)
		for item in t10:
			print "%-\n" % item
		
	else: 
		print "no T1?.... /n enter in dummy value"
else:
	print "Specify if import is True or false... do you want to import T10 data/n or input yourself"
	
print "Average T10 value: %f" % t10_value
print "Standard Dev of T10 values: %f " % stdev



print """
 --------------------------------------------------------------------------
-----------------------------LINEAR FIT of T1 DATA-------------------------
---------------------------------------------------------------------------"""

print "\n"



gradient, intercept, r_value, p_value, std_err = stats.linregress(t1power_f,t1int3)

print "Gradient and intercept", gradient, intercept

grad1 = str(gradient)
grad2 = gradient


print "R-squared", r_value**2

print "p-value", p_value

#____________________enhancement stuff___________________








po2 = []
po1 = []
enpower_f.insert(0, 0)



po2[:] = enpower_f[:]

maxenp = max(po2)

for i in po2:
	po1.append(i)






en1 = []

en_integral_f.insert(0, 1)

en1[:] = en_integral_f[:]



T1cor = []

for item in po1:
	T1cor.append(gradient*item + intercept)

print "\n"
print "t1 corr list:"
print "\n"

print T1cor

print """
---------------------------------------------------------------------------
-------------------------------------K sigma S-----------------------------
---------------------------------------------------------------------------"""

Ksigma1  = []
Ksigma2 = []
Ksigma3 = []
Ksigmas = []
Ksigma4 = []
Ksigmas1 = []

for item in en1:
	Ksigma1.append(float(1) - float(item))
	
for i in Ksigma1:
	Ksigma2.append(float(i))

Ksigma3 = [float(b) * float(m) for b,m in zip(Ksigma2,T1cor)]

for b,m in zip(Ksigma2,T1cor):
	Ksigma4.append(float(b)*float(m))




for i in Ksigma4:
	Ksigmas1.append(float(i)/float(658.0))


for i in Ksigmas1:
	Ksigmas.append(float(i/conc))



print """
--------------------------------------------------------------------------
---------------------------------Ksigma(max)*s(max)-----------------------
--------------------------------------------------------------------------"""

import numpy as np
from scipy.optimize import curve_fit

xdata = np.array(po1)
ydata = np.array(Ksigmas)


# simKsigmaS[A_, B_, P_] := A*P/(B + P)


def func(x, p1,p2):
  return (p1 * x)/ (p2 + x )

#Enter in fitting guess
Result_ksigma_smax, pcov = curve_fit(func, xdata, ydata,p0=(33.0,66.0))

perr = np.sqrt(np.diag(pcov))

Ksig_error = perr[0]



ksig = Result_ksigma_smax[0]
print "\n"
print ksig
print "error: %r" % Ksig_error

print """
--------------------------------------------------------------------------
-------------------------Ksigma/concentration-----------------------------
------------------to check against ksigma from rbscript-------------------"""
xdata1 = np.array(po1)
ydata1 = np.array(Ksigmas1)



# simKsigmaS[A_, B_, P_] := A*P/(B + P)


def func(x, p1,p2):
  return (p1 * x)/ (p2 + x )

Result_ksigma_smax1, pcov1 = curve_fit(func, xdata1, ydata1,p0=(1.0,0.2))

perr1 = np.sqrt(np.diag(pcov1))

Ksig_error_rb = perr1[0]



ksig1 = Result_ksigma_smax1[0]
print "\n"
print " This is ksigma*conc check to see if same in ksigma.csv in workup"
print ksig1
print "\n"

print "error: %r" % Ksig_error_rb

#----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------
#t1 error (set above) if t1extra = True-> import extra T1(0) data
# if false then use integral fitting error for T1(0)
# if 1 then manually set t1(0) error (stdevt1)
# if 2 then set T1(0) error same as error in T10
t1_atnopower = t1_integral_f[0]

if t1extra == True:
	print "Importing extra T1data from T12series.csv"
	with open('T12series.csv', 'rb') as t12csv:
		reader2 = csv.reader(t12csv)
		t12list = list(reader2)
	
	

	t12val, t12err, expnum = zip(*t12list)




	T12li = []
	T12li[0:]= t12val[1:]
	T12vals = []

	


	for i in T12li:
		T12vals.append(float(i))
	
	T12vals.append(t1_atnopower)

	

	t12_value = float(sum(T12vals)/(len(T12vals)))

	

	T12arr = numpy.array(T12vals)
	stdevt1 = numpy.std(T12arr)
	


	t12csv.close()
	print stdevt1
	print t12_value
	
elif t1extra == False:
	t1error = []
	with open("t1err.txt") as t1err1:
		t1error = t1err1.read()
    	stdevt1 = float(t1error)
elif t1extra == 1:
	stdevt1 = stdev
elif t1extra == 2:
	var5 = 'do_nothing'
    	
else:
	print "error something wrong with your extra T1 answer... True if importing False if not"


print """--------------------------------------------------------------------------
--------------------------------------krho--------------------------------
--------------------------------------------------------------------------"""



print "\n"
print 'T1 at no power:  %r   ' % t1_atnopower


print 'T10 :  %r ' % t10_value

def krho(t1,t10):
	return ((1/float(t1)) - (1/float(t10)))/conc


#Resultkrho = ((1/T1nopowervalue) - (1/T10value))/Conc 

Result_krho = krho(t1_atnopower,t10_value)
print "\n"
print 'krho: '
print Result_krho

# calculating error 




T10value_err = [t10_value + stdev,t10_value + stdev, t10_value - stdev,t10_value - stdev]

T1value_err = [t1_atnopower +  stdevt1, t1_atnopower -  stdevt1, t1_atnopower +  stdevt1, t1_atnopower -  stdevt1]


Result_krho_error = []

for x,y in zip(T1value_err,T10value_err):
	Result_krho_error.append(krho(x,y))
	



Result_krho_error2 = []
for i in Result_krho_error:
	Result_krho_error2.append(abs(Result_krho - float(i)))

KHRO_pmerror = max(Result_krho_error2)

print "khro error: %r" % KHRO_pmerror
	
	







print """
--------------------------------------------------------------------------
----------------------------------Coupling Constant-----------------------
--------------------------------------------------------------------------"""

def cplcnst(ks, kr):
	return float(ks)/float(kr)
	


Coupling_cnst =  cplcnst(ksig, Result_krho)
print "\n"
print "the coupling constant is : % r "  % Coupling_cnst


#Coupling constant error

khrolist = [Result_krho + KHRO_pmerror, Result_krho + KHRO_pmerror, Result_krho - KHRO_pmerror, Result_krho - KHRO_pmerror]
ksigmalister = [ksig + Ksig_error, ksig - Ksig_error, ksig + Ksig_error, ksig - Ksig_error]

cplist = []

for c1,c2 in zip(ksigmalister,khrolist):
	cplist.append( cplcnst(c1,c2))

cp2 = []
for i in cplist:
	cp2.append(abs(Coupling_cnst- float(i)))

Coupling_cnst_error = max(cp2)

print "Coupling constant error: %r" % Coupling_cnst_error
	
	

print"""
--------------------------------------------------------------------------- 
----------------------------------------Tauc-------------------------------
---------------------------------------------------------------------------"""

a = float((1.761*10**7)/658)


print "\n"





from mpmath import *
from numpy import *

mp.dps = 6



g = float((1.761*10.0**7.0)/658.0)

h = 1.0* 1.0* 10.0 **(-12.0)



def jtb(w,t):
	return (float(1.0) + (float(5.0/8.0))* (float(2.0)* float(w) * float(t))**(float(1.0/2.0)) + (float(1.0/8.0))* float(2.0)* float(w) * float(t))/ float((float(1.0) + (float(2.0)* float(w) * float(t))**(float(1.0/2.0)) + float(w) * float(t) + (float(1.0/6.0))* (float(2.0) * float(w) * float(t))**(float(3.0/2.0)) + (float(4.0/81.0))*(float(2.0)*float(w)* float(t))**float(2.0) + (float(1.0/81.0))* (float(2.0)* float(w) * float(t))**(float(5.0/2.0)) + (float(1.0/648.000))*(float(2.0)*float(w)*float(t))**(float(3.0))))
	

def pB(B,t):
	return (6.0 * jtb(g*B + 658.0* g*B,t*float(h)) - jtb(g*B*658.0 - g*B, t*float(h)))/(6.0*jtb(g* B *658.0 + g*B,t*float(h))+ 3.0* jtb(g*B,t*float(h)) + jtb(g*B*658.0 - g*B, t*float(h)))

def jtb1(w,t):
	return (float(1.0) + (float(5.0/8.0))* (float(2.0)* w * t)**(float(1.0/2.0)) + (float(1.0/8.0))* float(2.0)* w * t)/ (float(1.0) + (float(2.0)* w * t)**(float(1.0/2.0)) + w * t + (float(1.0/6.0))* (float(2.0) * w * t)**(float(3.0/2.0)) + (float(4.0/81.0))*(float(2.0)*w* t)**float(2.0) + (float(1.0/81.0))* (float(2.0)* w * t)**(float(5.0/2.0)) + (float(1.0/648.000))*(float(2.0)*w*t)**(float(3.0)))
	
def jtb2(w,t):
	return (mpf(1.0) + (mpf(5.0/8.0))* (mpf(2.0)* w * t)**(mpf(1.0/2.0)) + (mpf(1.0/8.0))* mpf(2.0)* w * t)/ (mpf(1.0) + (mpf(2.0)* w * t)**(mpf(1.0/2.0)) + w * t + (mpf(1.0/6.0))* (mpf(2.0) * w * t)**(mpf(3.0/2.0)) + (mpf(4.0/81.0))*(mpf(2.0)*w* t)**mpf(2.0) + (mpf(1.0/81.0))* (mpf(2.0)* w * t)**(mpf(5.0/2.0)) + (mpf(1.0/648.000))*(mpf(2.0)*w*t)**(mpf(3.0)))
	

	
def pB1(B,t):
	return (6.0 * jtb1(g*B + 658.0* g*B,t*float(h)) - jtb1(g*B*658.0 - g*B, t*float(h)))/(6.0*jtb1(g* B *658.0 + g*B,t*float(h))+ 3.0* jtb1(g*B,t*float(h)) + jtb1(g*B*658.0 - g*B, t*float(h)))
	

var = jtb(1.0,1.0)

var2 = pB(3500.0,1.0)



var3 = jtb1(1.0,1.0)
var4 = pB1(3500.0,1.0)




g = float((1.761*10.0**7.0)/658.0)

h = 1.0* 1.0* 10.0 **(-12.0)



"print solve(pB1(3500,t) - 0.036,t,minimal=True)"
"print solve(jtb1(1,t) - 0.51,t, numerical =True,)"



fir = 0.512296092078

"Coupling_constant = 0.0357295389231958"
# to test to see if find root is working value = 383.99


tau = lambda t: pB1(3500, t) - Coupling_cnst



tauc = findroot(tau,300)

print "Tau: "
print tauc

print "\n"

#tau error

tau1 = lambda t: pB1(3500, t) - (Coupling_cnst+ Coupling_cnst_error)
tau2 = lambda t: pB1(3500, t) - (Coupling_cnst - Coupling_cnst_error)

tauc1 = findroot(tau1,300)
tauc2 = findroot(tau2,300)

taulist = [tauc1,tauc2]
taulist2 = []

for i in taulist:
	taulist2.append(abs(float(tauc) - float(i)))

tauc_error = max(taulist2)

print "error: plus&minus %r " % tauc_error





#----------------------------------------------------------------------------------------
#------------------------------figure plotting-------------------------------------------
#----------------------------------------------------------------------------------------

# K sigma !!!!!!!!

#this is not necessary should remove if above functions w/o floats work
def func2(x, p1,p2):
  return float((p1 * x)/ float((p2 + x )))


a = float(Result_ksigma_smax[0])
b = float(Result_ksigma_smax[1])


#y values for fit
fity = []


po2 = sort(po1)

for i in sorted(po2):
	fity.append(func2(i,a,b))


#this is ksigma* concnentration

c = Result_ksigma_smax1[0]
d = Result_ksigma_smax1[1]

#y values for fit of ksigma*conc
fityconc = []

for i in sorted(po2):
	fityconc.append(func2(i,c,d))



plt.figure(1)
plt.subplot(211)
plt.scatter(po1, Ksigmas, color = 'red')
plt.plot(po2,fity, color = 'blue',linestyle='dashed')
plt.ylabel('ksigma')
plt.title('KsigmaS')

plt.subplot(212)
plt.scatter(po1, Ksigmas1, color = 'green')
plt.plot(po2,fityconc, color = 'purple',linestyle='dashed')
plt.xlabel('Power (dBm)')
plt.ylabel('ksigma*conc')


plt.savefig("ksigma.png", box_inches='tight')


#-------------------------------------------------------------------------------------
#-----------------------------t1_fit_plot---------------------------------------------
#-------------------------------------------------------------------------------------


def truncate(f, n):
    '''Truncates/pads a float f to n decimal places without rounding'''
    s = '{}'.format(f)
    if 'e' in s or 'E' in s:
        return '{0:.{1}f}'.format(f, n)
    i, p, d = s.partition('.')
    return '.'.join([i, (d+'0'*n)[:n]])

rsq = r_value**2


gradplot = float(truncate(grad2, 4))
intercept1 = float(truncate(intercept, 4))
rsq2 = float(truncate(rsq,4))


eqstr = "y = %r * x + %r \n Rsq: %r " % (gradplot, intercept1, rsq2)

listy = []

for i in t1powerplot :
	listy.append(gradplot * i + intercept1)

maxp = max(t1powerplot)
maxt1 = max(listy)

xtmin = -0.5
xtmax = 5.5
ytmax = maxt1 +0.5
ytmin = 0

textplace = maxp - 0.6

plt.figure(2)
plt.scatter(t1powerplot, t1int3, color = 'red')
plt.plot(t1powerplot,listy, color = 'blue', linestyle = 'dashed')
plt.xlim([xtmin, xtmax])
plt.ylim([ytmin,ytmax])
plt.xlabel('Power (dB)')
plt.ylabel('1 / T1')
plt.title('T1 Fit')
plt.text(textplace, maxt1, eqstr)


plt.savefig("t1fit.png", box_inches='tight')





#----------------------------------------------------------------------------------------
#------------------------------output file printing--------------------------------------
#----------------------------------------------------------------------------------------


from PIL import Image
from reportlab.platypus import Image
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle


from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch, mm
from reportlab.platypus import (Flowable, Paragraph,
                                SimpleDocTemplate, Spacer)
 
 

 






Imksig = Image('ksigma.png',8*inch, 6*inch)
Imt1plot = Image('t1fit.png',8*inch, 6*inch)








stylesheet=getSampleStyleSheet()
normalStyle = stylesheet['Normal']

 
PAGE_HEIGHT=defaultPageSize[1]
styles = getSampleStyleSheet()
Title = "DNP Work-up"
SL = "<b>SL concentration (M):</b> %r" % conc
Name = "<b>Name:</b>   %s" % Name
expt = "<b>EXPT:</b>    %s" % Exptname

stringt1t10 = "<b>T1 (0):</b> %r <b>Std Dev:</b> %f <br></br> <b>T10 (0):</b> %r  <b>Std Dev:</b> %f" % (t1_atnopower,stdevt1, t10_value,stdev)

v1 = []
v2 = []
v3 = []
v4 = []

for l3,l4 in enhancement_data:
	v3.append(l3)
	v4.append(l4)

for l1,l2 in T1_data:
	v1.append(l1)
	v2.append(l2)

v1.insert(0,'Power')
v2.insert(0, 'T1(s)')
v3.insert(0, 'Power')
v4.insert(0, 'Int')

T1plt = zip(v1,v2)
enplt = zip(v3,v4)

t = Table(T1plt)

t.setStyle(TableStyle([('TEXTCOLOR',(0,0),(0,-1),colors.red)]))

t1_l = "There are %d T1 experiments" % len3
en_l = "There are %d enhancement experiments" % Lengthen


t2 = Table(enplt)

t2.setStyle(TableStyle([('TEXTCOLOR',(0,0),(0,-1),colors.red)]))


T1corrpr =[T1cor[x:x+4] for x in xrange(0, len(T1cor), 4)]


t3 = Table(T1corrpr)


	



t1f = "<b>Gradient and intercept :</b> %s and %s" % (grad1, intercept)
t1f2 = "<b>R-squared:</b> %s" % r_value**2
t1f3 = "<b>p-value:</b> %s"  % p_value


T1corr = '\n'.join(map(str,T1cor))

ks = "<b>Ksigma:</b> %r" % ksig

kse = "error: %r" % Ksig_error

kse1 = "error: %r" % Ksig_error_rb


krho_d = "<b>Krho:</b>  %r" % Result_krho

khro_e = "error: %r" % KHRO_pmerror

cplct = "<b>The Coupling constant is:</b> %r" % Coupling_cnst

cplct_e = "error: %r" % Coupling_cnst_error

taucval = "<b>Tau is:</b> %s (ps)"  % tauc

taucvale = "error: %r" % tauc_error



ksig2 = "<b>Ksigma * concentration:</b> %r <br></br> check against ksigma from ksigma.csv  in work-up" % ksig1

Runtime = "<b>Work-up Runtime:</b> %s" % localtime

Date = "<b>Date of experiment:</b>  %s" % date




 
########################################################################
class BoxyLine(Flowable):
    """
    Draw a box + line + text
 
    -----------------------------------------
    | foobar |
    ---------
 
    """
 
    #----------------------------------------------------------------------
    def __init__(self, x=0, y=-15, width=100, height=15, text=""):
        Flowable.__init__(self)
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.text = text
        self.styles = getSampleStyleSheet()
 
    #----------------------------------------------------------------------
    def coord(self, x, y, unit=1):
        """
        http://stackoverflow.com/questions/4726011/wrap-text-in-a-table-reportlab
        Helper class to help position flowables in Canvas objects
        """
        x, y = x * unit, self.height -  y * unit
        return x, y
 
    #----------------------------------------------------------------------
    def draw(self):
        """
        Draw the shape, text, etc
        """
        self.canv.rect(self.x, self.y, self.width, self.height)
        self.canv.line(self.x, 0, 500, 0)
 
        p = Paragraph(self.text, style=self.styles["Normal"])
        p.wrapOn(self.canv, self.width, self.height)
        p.drawOn(self.canv, *self.coord(self.x+2, 10, mm))
 

txt = "<font size=24>This is a 24 point font</font>"

   
Elements=[]
HeaderStyle = styles["Heading1"]
ParaStyle = styles["Normal"]
PreStyle = styles["Code"]
HeaderStyle3 = styles["Heading3"]

 
def header(txt, style=HeaderStyle, klass=Paragraph, sep=0.3):
    s = Spacer(0.2*inch, sep*inch)
    Elements.append(s)
    para = klass(txt, style)
    Elements.append(para)
    
def header3(txt, style=HeaderStyle3, klass=Paragraph, sep=0.2):
    s = Spacer(0.1*inch, sep*inch)
    Elements.append(s)
    para = klass(txt, style)
    Elements.append(para)
 
def p(txt):
    return header(txt, style=ParaStyle, sep=0.1)
    
def a(var):
	Elements.append(var)
    




box1 = BoxyLine(text = "Enhancement Data")
box2 = BoxyLine(text="T1 data")
box3 = BoxyLine(text="T1 linear fit")
box4 = BoxyLine(text="T1 corr list")
box5 = BoxyLine(text="Ksigma*smax")
box6 = BoxyLine(text="Krho")
box7 = BoxyLine(text="Coupling Constant")
box8 = BoxyLine(text="Tau")

    


#THIS is order of pdf output

header(Title)
p(Name)
p(expt)
p(SL)
p(Date)
p(Runtime)
a(box1)
Elements.append(t2)
p(en_l)
a(box2)
Elements.append(t)
p(t1_l)
a(box3)
Elements.append(Imt1plot)
p(t1f)
p(t1f2)
p(t1f3)
a(box4)
a(Spacer(0, 0.5*inch))
Elements.append(t3)
a(box5)
Elements.append(Imksig)
p(ks)
p(kse)
p(ksig2)
p(kse1)
a(box6)
a(Spacer(0, 0.5*inch))
p(krho_d)
p(khro_e)
if imp == True:
	p("<font size=8>***Caclulated from T10series.csv***</font>")
else:
	d1 = True
p(stringt1t10)
a(box7)
a(Spacer(0, 0.5*inch))
p(cplct)
p(cplct_e)
a(box8)
a(Spacer(0, 0.5*inch))
p(taucval)
p(taucvale)


 



 


go()
 







	



