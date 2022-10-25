import os
from matplotlib import pyplot
import numpy as np
pyplot.rcParams["font.family"] = "Times New Roman"
pyplot.rcParams["font.size"] = 22
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
	for i in range(2,0,-1):
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
			pyplot.plot(x,y[i],linestyle,linewidth=1.0,color=linecolor,label=legends[count])
		else:
			pyplot.plot(x,y[i],linestyle,linewidth=1.0,color=linecolor)
		count += 1

def graph(filenames='score',legends=["Fig. 2(b): dome model","Fig. 2(c): flat model"]):

	fig = pyplot.figure(figsize=(10,3.5),frameon=False,constrained_layout=True)
	plot(filenames,legends)
	fig.legend(loc='lower right', bbox_to_anchor=(0.98, 0.25))
	pyplot.xlabel("trained episodes",labelpad=5)
	pyplot.ylabel("$\sum{r}$",rotation=0,labelpad=10)
	pyplot.yticks([0,-250,-500,-750,-1000])

	# save figure
	pyplot.show()
	pyplot.close()

# file_input = input("Please enter the name of csv file in the same directory from which you want to make a graph.\n")
graph()