import image.selectnucleus as sn
import glob



file_number = 0
for file in glob.glob("/home/maciek/DOKTORAT/all_data/all_images/*"):
	a = sn.SelectNucleus(file,False, 'output/'+str(file_number))
	# for i in range (0, a.all_frames):
	# 	a.set_frame(i)
	# 	a.convert_to_grey_scale()
	# 	a.split_nucleus()
	# 	# # a.center_nucles()
	# 	a.show_image()

	a.set_frame(1)
	a.convert_to_grey_scale()
	a.split_nucleus()
	# # a.center_nucles()
	# a.show_image()	


	file_number += 1
