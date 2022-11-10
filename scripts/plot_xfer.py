from fnl_pipe.pipe import TransferFunction
from fnl_pipe.util import OutputManager
import numpy as np
import matplotlib.pyplot as plt

import sys

# xfer_path = 'transfer_function.h5'

if __name__ == "__main__":
	args = sys.argv
	assert len(args) == 2
	xfer_path = args[1]

	om = OutputManager(title='plot_xfer')

	tf = TransferFunction.from_file(xfer_path)
	ntrial_nl = tf.ntrial_nl
	nave_fl = tf.nave_fl
	ntrial_fl = tf.ntrial_fl
	ells = np.arange(tf.lmax + 1)
	norm = ells * (ells + 1) / 2 / np.pi

	plt.figure(dpi=300)
	plt.title('Transfer Function')
	plt.plot(ells, tf.fl / tf.bl2)
	plt.ylabel('fl (normalized)')
	plt.xlabel('l')
	om.savefig(f'fl_nave_{nave_fl}_ntrial_{ntrial_fl}.png')
	plt.close()

	plt.figure(dpi=300)
	plt.title('nl')
	plt.plot(ells, norm * tf.nl)
	plt.ylabel('nl (normalized) uK^2')
	plt.xlabel('l')
	om.savefig(f'nl_ntrial_{ntrial_nl}.png')
	plt.close()

	plt.figure(dpi=300)
	plt.title('nl_tilde')
	plt.plot(ells, norm * tf.nl_tilde)
	plt.ylabel('nl_tilde (normalized) uK^2')
	plt.xlabel('l')
	om.savefig(f'nl_tilde_ntrial_{ntrial_nl}.png')
	plt.close()