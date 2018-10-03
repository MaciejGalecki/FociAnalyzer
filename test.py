import image.selectnucleus as sn

a = sn.SelectNucleus('/home/maciek/github/FociAnalyzer/image/1-4.tif')
a.convert_to_grey_scale()
a.split_nucleus()

