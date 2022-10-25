import os
from matplotlib import pyplot
import numpy as np
pyplot.rcParams["font.family"] = "Times New Roman"
pyplot.rcParams["font.size"] = 20
pyplot.rcParams["mathtext.fontset"] = 'stix' # 'stix' 'cm'
pyplot.rcParams['mathtext.default'] = 'it'
xscale = 10 # scale of x

def plot(filename,legends=None):
	count = 0
	f, extension = os.path.splitext(filename)
	if extension is '':
		f2 = f + '.csv'
	elif extension is not '.csv':
		raise Exception("The file must have csv format.")
	else:
		f2 = f

	y = np.loadtxt(f2,delimiter=',',dtype=float)
	count = 0
	for i in range(1,3):
		x = np.linspace(0,len(y[i])*xscale,len(y[i]))
		if i == 0:
			linestyle = '-'
			linecolor = 'black'
		elif i == 1:
			linestyle = ':'
			linecolor = 'red'
		elif i == 2:
			linestyle = '--'
			linecolor = 'blue'
		elif i == 3:
			linestyle = '-.'
			linecolor = 'green'
		if legends is not None:
			pyplot.plot(x,y[i],linestyle,linewidth=0.8,color=linecolor,label=legends[count])
		else:
			pyplot.plot(x,y[i],linestyle,linewidth=0.8,color=linecolor)
		count += 1

def graph(filenames='score',legends=[r"$6\times2$-grid model","arch model"]):

	fig = pyplot.figure(figsize=(8,4),frameon=False,constrained_layout=True)
	plot(filenames,legends)
	fig.legend(loc='lower right', bbox_to_anchor=(0.98, 0.18))
	pyplot.xlabel("trained episodes",labelpad=5)
	pyplot.ylabel("cumulative rewards $\sum{r}$",rotation=90,labelpad=5)

	# save figure
	pyplot.show()
	pyplot.close()

# file_input = input("Please enter the name of csv file in the same directory from which you want to make a graph.\n")
graph()