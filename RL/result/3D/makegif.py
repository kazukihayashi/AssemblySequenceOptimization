from audioop import reverse
from PIL import Image,ImageDraw
import os
import re

"""
hyperparameters
"""
png = re.compile("png") # search images that match this suffix
skip = 1 # interval to skip the images (if no skip, 1)
duration = 100 # [ms]
init_duration = 500 # [ms]
last_duration = 500 # [ms]

'''
operation
'''
def ImageList(dir,flip_left_right=False,flip_top_bottom=False):
	files = os.listdir(dir)
	images = []
	count = 0
	for file in files:
		if png.search(file):
			im = Image.open(os.path.join(dir,file))
			if flip_left_right:
				im = im.transpose(Image.FLIP_LEFT_RIGHT)
			if flip_top_bottom:
				im = im.transpose(Image.FLIP_TOP_BOTTOM)
			w = im.width
			h = im.height
			# im_crop = im.crop((100,150,w-100,h-250)) # left, upper, right, lower
			# draw = ImageDraw.Draw(im).text((0.10*w, 0.25*h),"step {:}".format(count),(150,150,150)) # Add caption
			images.append(im)
			count += 1
	return images

def Makegif(images,skip,duration,init_duration=None,last_duration=None,reverse=False,name="animation.gif"):
	mov = [images[i] for i in range(0,len(images)-1,skip)]
	mov.append(images[-1])
	durs = [duration for i in range(len(mov))]
	if init_duration is not None:
		durs[0] = init_duration
	if last_duration is not None:
		durs[-1] = last_duration
	if reverse:
		mov.reverse()
	mov[0].save(name,save_all=True, append_images=mov[1:], optimize=True, duration=durs, loop=0)

def Concat_Images(images_1,images_2):
	images = []
	canvas = Image.new('RGB', (images_1[0].width + images_2[0].width, images_1[0].height), color=(255,255,255))
	for i in range(len(images_1)):
		dst = canvas.copy()
		dst.paste(images_1[i], (0, 0))
		dst.paste(images_2[i], (images_1[i].width, 0))
		images.append(dst)
	return images

aaa = os.getcwd()
aaa2 = os.path.join(os.getcwd(),"isom")
images = ImageList(os.getcwd())
# Makegif(images,skip,duration,init_duration,last_duration,True,"animation.gif")
images2 = ImageList(os.path.join(os.getcwd(),"isom"),True,False)
# Makegif(images2,skip,duration,init_duration,last_duration,True,"animation2.gif")

images3 = Concat_Images(images,images2)
Makegif(images3,skip,duration,init_duration,last_duration,True,"animation2.gif")


