from PIL import Image,ImageDraw
import os
import re

"""
hyperparameters
"""
png = re.compile("png") # search images that match this suffix
skip = 1 # interval to skip the images (if no skip, 1)
duration = 50 # [ms]
init_duration = 500 # [ms]
last_duration = 500 # [ms]

'''
operation
'''
files = os.listdir(os.getcwd())
images = []
count = 0
for file in files:
	if png.search(file):
		im = Image.open(file)
		w = im.width
		h = im.height
		# im_crop = im.crop((100,150,w-100,h-250)) # left, upper, right, lower
		# draw = ImageDraw.Draw(im).text((0.10*w, 0.25*h),"step {:}".format(count),(150,150,150)) # Add caption
		images.append(im)
		count += 1

mov = [images[i] for i in range(0,len(images)-1,skip)]
mov.reverse()
mov.append(images[-1])
durs = [duration for i in range(len(mov))]
durs[0] = init_duration
durs[-1] = last_duration

'''
save gif animation
'''
mov[0].save("animation.gif",save_all=True, append_images=mov[1:], optimize=True, duration=durs, transparency=255, loop=0)