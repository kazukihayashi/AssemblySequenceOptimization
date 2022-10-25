from PIL import Image
import os

input_extension = ".png"
output_extension = ".jpg"
nx = 10
skip = 1 # interval to skip the images (if no skip, 1)

crop_left = 30
crop_right = 30
crop_top = 30
crop_bottom = 30
# crop_left = 50
# crop_right = 50
# crop_top = 140
# crop_bottom = 140

def listing(filelist,extension,skip):
	target_filelist = []
	for fichier in filelist[:]: # filelist[:] makes a copy of filelist.
		if fichier.endswith(extension):
			target_filelist.append(fichier)

	skipped_filelist = []
	for i in range(0,len(target_filelist)-1,skip):
		skipped_filelist.append(target_filelist[i])
	skipped_filelist.append(target_filelist[-1])
			
	return skipped_filelist

def get_concat_h(im1, im2):
	if im1.height == 0:
		h = im2.height
	else:
		h = im1.height
	dst = Image.new('RGB', (im1.width + im2.width, h), color=(255,255,255))
	dst.paste(im1, (0, 0))
	dst.paste(im2, (im1.width, 0))
	return dst

def get_concat_v(im1, im2):
	if im1.width == 0:
		w = im2.width
	else:
		w = im1.width
	dst = Image.new('RGB', (w, im1.height + im2.height), color=(255,255,255))
	dst.paste(im1, (0, 0))
	dst.paste(im2, (0, im1.height))
	return dst

def get_wh(file):
	im = Image.open(file)
	w = im.width
	h = im.height
	return w,h

def all_concat(files,nx,w,h,cl=0,cr=0,ct=0,cb=0):
	im_rows = []
	for i in range((len(files)//nx)+1):
		im_rows.append(Image.new('RGB',(0,0)))
		if i < (len(files)//nx):
			for j in range(nx):
				new_im = Image.open(files[nx*i+j])
				new_im = new_im.crop((cl,ct,w-cr,h-cb))
				im_rows[i] = get_concat_h(im_rows[i],new_im)
		else: # last row
			for j in range(len(files)%nx):
				new_im = Image.open(files[nx*i+j])
				new_im = new_im.crop((cl,ct,w-cr,h-cb))
				im_rows[i] = get_concat_h(im_rows[i],new_im)
	im = Image.new('RGB',(0,0))
	for im_row in im_rows:
		im = get_concat_v(im,im_row)
	return im

if os.path.isfile("concat{0}".format(output_extension)):
	os.remove("concat{0}".format(output_extension))
files = os.listdir(os.getcwd())
files = listing(files,input_extension,skip)
files.reverse()
w,h = get_wh(files[0])
image = all_concat(files[1:],nx,w,h,cl=crop_left,cr=crop_right,ct=crop_top,cb=crop_bottom)
image.save("concat{0}".format(output_extension))