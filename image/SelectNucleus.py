class SelectNucleus( ImageTools ):
	"""Find each nucleus on the image, split them, enumarate and create new images
	   with nucleus at the center"""

	def __init__( self ):
		"""Load image etc."""
		return None

	def LoadImage( path ):
		"""Loads a single image to opencv format"""
		return None

	def ConvertTogreyScale():
		"""creates a temporary greyscale image, later on we will need RGB one too"""
		return None

	def SplitNucleus():
		"""split image and into few parts, one neclues in each"""
		return None

	def CenterNucles():
		"""Find the mass center of the nucleus and place it at the center of the image"""
		return None

	def Transformation():
		"""Performs mathematical transformation of coordinates based of the shape changes between 
		   referenced nucleus and currect nucles"""
		return None

