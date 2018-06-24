class FindFoci( ImageTools ):
	"""Class used to find and analyze foci in nucleus images"""

	def __init__( self ):
		"""Load image etc."""
		return None

	def PrepareGrayScale():
		"""Prepares grayscale 256 bit image. Chooses the RGB channel with the highest contrast,
		   subtracks background etc."""
		return None

	def PrepareBinary():
		"""Prepares binary image. PrepareGrayScale must by used first"""
		return None

	def FindGrayScale():
		"""Find the coordinations of foci centers and measures their sizes.
		   Uses grayscale image."""
		return None

	def FindBinary():
		"""Find the coordinations of foci centers and measures their sizes.
		   Uses binary image."""
		return None

	def MeasureArea():
		"""Measures are of each focus"""
		return None

	def MeasureIntensity():
		"""Measures intensity of each focus"""
		return None
