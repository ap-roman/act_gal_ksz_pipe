from fnl_pipe.util import OutputManager

import numpy as np

logfile = '/home/aroman/ksz_repos/act_gal_ksz_pipe/output/phack-bootstrap_5/logs/log.log'

headerline = 'zerr_max, do_lrg_cut, vr_width, alpha_bs_2'

if __name__ == "__main__":
    # om = OutputManager(base_path='output', title='plot-phack-bootstrap', logs=['log',])
    # printlog = om.printlog

	with open(logfile, 'r') as f:
		lines = [l.rstrip() for l in f.readlines()]

		assert headerline in lines

		iheader = 0
		for line in lines:
			if headerline == line:
				break
			iheader += 1

		nres = len(lines) - iheader - 1
		print(f'there are {nres} results')
		results = np.empty((nres, 4))
		for line, ind in zip(lines[iheader + 1:], range(nres)):
			z, lrg_cut, vr_width, alpha = line.split(',')
			results[ind] = (float(z), float(lrg_cut), float(vr_width), float(alpha))

		print(results)

		zs = []
		docuts = []
		vr_widths = []