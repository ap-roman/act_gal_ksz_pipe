import numpy as np
from fnl_pipe.util import average_fl, downgrade_fl, upgrade_fl, fequal

if __name__ == "__main__":
    lmax = 1024
    n_ave = 4
    lmax_ni = lmax + (n_ave - 1)

    assert lmax % n_ave == 0

    ar = np.arange(lmax + 1) + 1.
    ar2 = np.arange(lmax_ni + 1) + 1.
    ave = np.empty(lmax//n_ave + 1)
    ave[0] = ar[0]

    ave[1:] = ar[1:].reshape(lmax//n_ave, n_ave).sum(axis=-1) / n_ave

    ave_fl_ret = np.empty(lmax + 1)
    ave_fl_ret[0] = ar[0]
    ave_fl_ret[1:] = np.repeat(ave[1:], n_ave)

    print('testing average_fl')
    assert fequal(ave_fl_ret, average_fl(ar, n_ave)), 'FAIL'
    print('PASS')
    
    print('testing downgrade_fl')
    assert fequal(ave, downgrade_fl(ar, n_ave)), 'FAIL'
    print('PASS')

    print('consistency check on average_fl and downgrade_fl -> upgrade_fl')
    assert fequal(average_fl(ar, n_ave), upgrade_fl(downgrade_fl(ar, n_ave), n_ave)), 'FAIL'
    print('PASS')

    ave_ni = np.empty(lmax // n_ave + 2)
    ave_ni[0] = ar2[0]
    ave_ni[1:-1] = ar2[1:-(n_ave - 1)].reshape(lmax // n_ave, n_ave).sum(axis=-1) / n_ave
    ave_ni[-1] = ar2[-(n_ave -1):].sum() / (n_ave -1)

    ni_repeat = np.empty(lmax_ni + 1)
    ni_repeat[0] = ave_ni[0]
    ni_repeat[1:-(n_ave - 1)] = np.repeat(ave_ni[1:-1], n_ave)
    ni_repeat[-(n_ave - 1):] = np.repeat(ave_ni[-1], n_ave - 1)

    print('average_fl non-integer case')
    assert fequal(average_fl(ar2, n_ave), ni_repeat), 'FAIL'
    print('PASS')
