from fnl_pipe.pipe import TransferFunction
import numpy as np
import matplotlib.pyplot as plt

xfer_path = 'transfer_function.h5'
outdir = 'plots/'

if __name__ == "__main__":
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
	plt.savefig(outdir + f'fl_nave_{nave_fl}_ntrial_{ntrial_fl}.png')

	plt.figure(dpi=300)
	plt.title('nl')
	plt.plot(ells, norm * tf.nl)
	plt.ylabel('nl (normalized) uK^2')
	plt.xlabel('l')
	plt.savefig(outdir + f'nl_ntrial_{ntrial_nl}.png')

	plt.figure(dpi=300)
	plt.title('nl_tilde')
	plt.plot(ells, norm * tf.nl_tilde)
	plt.ylabel('nl_tilde (normalized) uK^2')
	plt.xlabel('l')
	plt.savefig(outdir + f'nl_tilde_ntrial_{ntrial_nl}.png')