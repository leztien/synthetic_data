# printed_digits
datasets of printed digits for machine learning

the module 'generate_printed_digits.py' makes dataset(s) of printed digit matreces

DESCRIPTION:
a 3-piece collection of datasets of printed digits
standard dataset: non-modified printed digits
augmented dataset: digits shifted one pixel up/down/left/right
rotated dataset: digits rotated ca. 10 degrees clockwise and counter-clockwise
Notes:
the datasets are not shuffled and not split into training/test sets
the standard dataset contains a duplicate of itself to compensate for the low number of samples
the augmented dataset contains the standard dataset in itself
the rotated dataset also contains the standard dataset in itself, as well as italicized versions of digits
