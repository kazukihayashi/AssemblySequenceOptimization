import colorsys
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D
import numpy as np
import pickle

def Draw(node, connectivity, section, node_color=None, node_vec=None, node_size=None, node_shape=None, front_node_index=None, edge_color=None, edge_annotation=None, scale=3.0, name=0, save=True, show=False):
	"""
	node[nk,3]or[nk,2]  :(float) nodal coordinates
	connectivity[nm,2]	:(int)   connectivity to define member
	section[nm]			:(float) cross-sectional area of member
	edge_annotation[nm]	:(float) axial stress of member
	"""
	fig = pyplot.figure(figsize=(8,8))

	if node.shape[1] == 2 or np.allclose(node[:,2],0): # 2D

		if node_shape is None:
			node_shape = ["o" for i in range(np.shape(node)[0])]
		if node_size is None:
			node_size = [3 for i in range(np.shape(node)[0])]
		if node_color is None:
			node_color = [(0.5,0.5,0.5) for i in range(np.shape(node)[0])]
		if edge_color is None:
			edge_color = [(0.5,0.5,0.5) for i in range(np.shape(connectivity)[0])]
		
		ax = pyplot.subplot()
		for i in range(len(connectivity)):
			line = Line2D([node[connectivity[i,0],0],node[connectivity[i,1],0]],[node[connectivity[i,0],1],node[connectivity[i,1],1]],linewidth=section[i]*scale,color=edge_color[i])
			ax.add_line(line)
		for i in range(np.shape(node)[0]):
			ax.plot([node[i,0]],[node[i,1]], node_shape[i], color=node_color[i], ms=node_size[i]+5)
			if node_shape[i] == "o":
				ax.plot([node[i,0]],[node[i,1]], node_shape[i], color='white', ms=node_size[i])
		if edge_annotation is not None:
			for i in range(len(connectivity)):
				ax.annotate("{:3.1f}".format(edge_annotation[i]),(np.mean(node[connectivity[i,:],0]),np.mean(node[connectivity[i,:],1])))
		pyplot.xlim([np.min(node[:,0]),np.max(node[:,0])])
		pyplot.ylim([np.min(node[:,1]),np.max(node[:,1])])
		pyplot.tick_params(labelbottom="off",bottom="off",labelleft="off",left="off")
		pyplot.axis('scaled')
		pyplot.axis('off')
		pyplot.savefig(r'result\t{0:0=3}.png'.format(name))
		if show:
			pyplot.show()
		pyplot.close(fig)

	else: # 3D

		# make space
		ax = Axes3D(fig)
		ax.set_box_aspect((1,1,1))

		# axis label
		ax.set_xlabel("x")
		ax.set_ylabel("y")
		ax.set_zlabel("z")

		if node_shape is None:
			node_shape = ["o" for i in range(node.shape[0])]
		if node_size is None:
			node_size = [3 for i in range(node.shape[0])]
		if node_color is None:
			node_color = [(0.0,0.0,0.0) for i in range(node.shape[0])]
		if edge_color is None:
			edge_color = ["gray" for i in range(connectivity.shape[0])]

		# plot node
		for i in range(np.shape(node)[0]):
			ax.plot(node[i,0],node[i,1],node[i,2], node_shape[i], color=node_color[i], ms=node_size[i])

		# connect member
		section = section*scale
		for i in range(len(connectivity)):
			x = [node[connectivity[i,0],0],node[connectivity[i,1],0]]
			y = [node[connectivity[i,0],1],node[connectivity[i,1],1]]
			z = [node[connectivity[i,0],2],node[connectivity[i,1],2]]
			ax.plot(x,y,z, linewidth=section[i], color=edge_color[i])

		# plot front node
		if front_node_index is not None:
			for fni in front_node_index:
				ax.plot(node[fni,0],node[fni,1],node[fni,2], node_shape[fni], color=node_color[fni], ms=node_size[fni])

		ax.tick_params(labelbottom="off",bottom="off") # x軸の削除
		ax.tick_params(labelleft="off",left="off") # y軸の削除
		ax.set_xticklabels([]) 
		ax.axis("off") #枠線の削除

		# view angle
		ax.set_proj_type('ortho')
		ax.set_box_aspect(np.ptp(node,axis=0))
		# # view angle
		ax.view_init(elev=15, azim=60) # For grid 3D truss
		# ax.view_init(elev=45, azim=90) # For Hangai truss

		# # save figure
		pyplot.savefig("result\isom{0:0=3}.png".format(name))

		# # view angle
		# ax.view_init(elev=0, azim=0)

		# # save figure
		# pyplot.savefig("result\elev1("+str(name)+").eps")

		# # view angle
		# ax.view_init(elev=0, azim=270-1.0e-5)

		# # save figure
		# pyplot.savefig("result\elev2("+str(name)+").eps")	

		# # view angle
		# ax.view_init(elev=90, azim=270)

		# # save figure
		# pyplot.savefig("result\plan("+str(name)+").eps")	

		# # save 360 figure
		# for rot in range(360):
		# 	ax.view_init(elev=30, azim=rot)
		# 	pyplot.savefig(r"result\movie{0:0=3}.png".format(rot))

		# close
		# pyplot.show()
		pyplot.close(fig)

	return fig

def graph(y):
	x = np.linspace(0,y.shape[1],y.shape[1])
	for i in range(y.shape[0]):
		pyplot.figure(figsize=(10,4))
		pyplot.plot(x,y[i],linewidth=1)
		pyplot.savefig(r"result\graph({0:d}).png".format(i))
		pyplot.clf
	pyplot.close()

