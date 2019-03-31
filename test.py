import image.selectnucleus as sn
import glob


window_size = {}
file_number = 0
# for file in glob.glob("/home/maciek/DOKTORAT/all_data/all_images/*"):

for file in glob.glob("/home/maciek/DOKTORAT/Harry/0Gy/*"):
	# a = sn.SelectNucleus(file,False, 'output_mean_c/'+str(file_number), window_size)
	a = sn.SelectNucleus(file,False, 'test/'+str(file_number), window_size)
	# for i in range (0, a.all_frames):
	# 	a.set_frame(i)
	# 	a.convert_to_grey_scale()
	# 	a.split_nucleus()
	# 	# # a.center_nucles()
	# 	a.show_image()

	# a.set_frame(1)
	a.convert_to_grey_scale()
	a.show_image()	
	a.apply_clahe()
	a.apply_threshold()
	# a.convert_to_grey_scale()
	a.split_nucleus()
	# # a.center_nucles()
	
	file_number += 1
	window_size = a.give_window_size()
	# a.save_window_size('output_mean_c/sizes.json')
	a.save_window_size('test/sizes.json')



# test
# a = sn.SelectNucleus('/home/maciek/DOKTORAT/all_data/all_images/1-4.tif',False, 'output/test')
# a.set_frame(1)
# # a.apply_clahe()
# a.apply_threshold()
# a.show_threshold()
# # a.show_image()
# # a.save_img('dupa')

#tests
# for i in range(20):
# 	for j in (4,6,8,10,15,20,25,30,40,50,60,70,80,90,100,150,200,300):
# 		a = sn.SelectNucleus('/home/maciek/DOKTORAT/all_data/all_images/1-4.tif',False, 'output/test')
# 		a.set_frame(1)
# 		# a.show_image()
# 		a.convert_to_grey_scale()
# 		# a.show_image()
# 		a.apply_clahe(i, (j,j))
# 		a.save_img(str(i)+str(j))