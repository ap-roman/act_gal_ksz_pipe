from fnl_pipe.pipe import TransferFunction, Metadata2D

import numpy as np

test_path = 'xfer_test.h5'

if __name__ == "__main__":
    print('testing xfer data structure to/from file')
    lmax = 12000
    fl = np.arange(lmax + 1).astype(float)
    nave = 2
    ntrial = 1
    ntrial_nl = 100

    # includes a test of ave_fl:
    fl_ave = fl.copy()
    fl_ave[1:] = np.repeat(fl_ave[1:].reshape(lmax//2, 2).sum(axis=1), 2)/2
    fl_ave[0] = fl_ave[1]

    nl = np.ones(lmax + 1)
    nl_tilde = nl.copy() + 1.

    r_fkp = 0.62
    r_lwidth = 1.5
    cmb_fname = 'cmb_test.fits'
    gal_fname = 'gal_test.h5'

    md = Metadata2D(r_fkp, r_lwidth, cmb_fname, gal_fname)
    tf = TransferFunction(nl, nl_tilde, ntrial_nl, fl, nave, ntrial, md, do_ave=True)

    TransferFunction.to_file(test_path, tf, fmode='w')
    tf2 = TransferFunction.from_file(test_path)

    print('testing data integrity')
    assert nave == tf2.nave_fl
    assert lmax == tf2.lmax
    assert ntrial == tf2.ntrial_fl
    assert ntrial_nl == tf2.ntrial_nl
    assert r_fkp == tf2.metadata.r_fkp
    assert r_lwidth == tf2.metadata.r_lwidth
    assert cmb_fname == tf2.metadata.cmb_fname
    assert gal_fname == tf2.metadata.gal_fname
    assert np.all(nl == tf2.nl)
    assert np.all(nl_tilde == tf2.nl_tilde)
    print('PASS')

    print('testing ave_fl correctness')
    assert np.all(fl_ave == tf2.fl)
    print('PASS')