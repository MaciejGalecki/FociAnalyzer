from . import imagetools as it

class FindFoci( it.ImageTools ):
	"""Class used to find and analyze foci in nucleus images"""

	def __init__( self, path ):
		"""Load image etc."""
		
		self.path = path
		print ( "Loaded image: ", path )
		return None

	def PrepareGreyScale( self ):
		"""Prepares greyscale 256 bit image. Chooses the RGB channel with the highest contrast,
		   subtracks background etc."""
		return None

	def PrepareBinary( self ):
		"""Prepares binary image. PrepareGreyScale must by used first"""
		return None

	def FindGreyScale( self ):
		"""Find the coordinations of foci centers and measures their sizes.
		   Uses greyscale image."""
		return None

	def FindBinary( self ):
		"""Find the coordinations of foci centers and measures their sizes.
		   Uses binary image."""
		return None

	def MeasureArea( self ):
		"""Measures are of each focus"""
		return None

	def MeasureIntensity( self ):
		"""Measures intensity of each focus"""
		return None