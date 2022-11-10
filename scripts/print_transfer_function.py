import numpy as np
from fnl_pipe.pipe import TransferFunction
import sys

if __name__ == "__main__":
	args = sys.argv
	assert len(args) == 2
	xfer_path = args[1]
	
	xfer_function = TransferFunction.from_file(xfer_path)
	print(xfer_function)