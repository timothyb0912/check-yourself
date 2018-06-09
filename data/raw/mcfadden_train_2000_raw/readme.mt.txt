Daniel McFadden and Kenneth Train, "Mixed MNL Models for Discrete
Response",  Journal of Applied Econometrics, Vol. 15, No. 5, 2000, pp.
447-470. 

The file xmat.txt contains the data. The file dotable2.g is a Gauss command
file that takes the data in xmat.txt and creates the parameter estimates
given in Table 2 of the paper. Both files, which are in DOS format, are
zipped in the file mt-data.zip. Details of these two files are given below.

XMAT.TXT

Xmat.txt is an ascii file consisting of 181,506 lines of data with 4 numbers
in each line. Each choice experiment is an observation, and there are 39
lines of data for each choice experiment. The file therefore contains data
on 4654 observations (4654 = 181,506 / 39) and 156 pieces of information for
each observation (where 156 = 39 * 4).

This file with 181,506 lines and 4 numbers per line should be read, or
reconfigured, into a matrix of dimension 4654x156, such that each row
contains all the information about one choice experiment. In each
experiment, the respondent was presented with 6 alternative vehicles. The
columns contain the attributes of the six vehicles and the identification of
which vehicle the respondent chose. Each attribute consists of 6 columns,
giving the value of the attribute for each of the six vehicles. The
variables contained in each set of 6 columns are given below. Precise
definitions of the variables are given in Table 1 of the paper.

Variable	Columns:
1		1-6		Price divided by ln(income)
2		7-12		Range
3		13-18		Acceleration
4		19-24		Top speed
5		25-30		Pollution
6		31-36		Size
7		37-42		"Big enough"
8		43-48		Luggage space
9		49-54		Operating cost
10		55-60		Station availability
11		61-66		Sports utility vehicle
12		67-72		Sports car
13		73-78		Station wagon
14		79-84		Truck
15		85-90		Van
16		91-96		Constant for EV
17		97-102		Commute < 5 x EV
18		103-108	College x EV
19		109-114	Constant for CNG
20		115-120	Constant for methanol
21		121-126	College x methanol
22		127-132	These data are not used. 
23		133-138	Dependent variable. 1 for chosen vehicle, 0 otherwise
24		139-144	Non-EV dummy
25		145-150	Non-CNG dummy
26		151-156	These data are not used.


DOTABLE2.G

This text file contains the Gauss code that reads the data from xmat.txt and
calculates the mixed logit parameter estimates given in Table 2 of the
paper. A reader who has Gauss installed, along with the Gauss maxlik
routine, can run the program to reproduce the results. The only modification
that is needed is to change the file paths appropriately in specifying the
output file (output file=path/filename) and in loading the input data
(load[4564,(6*26)]=path/xmat.txt.)

This Gauss code is the cross-section version of the mixed logit estimation
routines contained at K. Train's website: http://elsa.berkeley.edu/~train,
with the model and data specified as needed for the model in Table 2. The
website contains a manual that describes how to modify the code for other
specifications and other data.

