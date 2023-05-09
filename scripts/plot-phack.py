import matplotlib.pyplot as plt
import numpy as np
from fnl_pipe.util import OutputManager

res_file = '/home/aroman/ksz_repos/act_gal_ksz_pipe/output/phack-mc_220/logs/res.csv'


def get_data(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        assert len(lines) > 1
        headers = [a.strip() for a in lines[0].split(',')]
        data = []
        for line in lines[1:]:
            data.append([float(a.strip()) for a in line.split(',')])
        assert len(headers) == len(data[0])
        data = np.array(data)

        return np.array(headers), data

        # ret = {}
        # for header, col in zip(headers, data.T):
        #     ret[header] = col
        # return ret


def get_unique(col):
    ret = []
    for val in col:
        if val not in ret: ret.append(val)

    return np.array(ret)

if __name__ == "__main__":
    om = OutputManager(base_path='output', title='plot-phack', logs=['log',])
    headers, data = get_data(res_file)

    izerr = np.where(headers == 'zerr_max')[0][0]
    ialpha = np.where(headers == 'alpha_mc')[0][0]
    ido_lrg = np.where(headers == 'do_lrg_cut')[0][0]
    ivr_width = np.where(headers == 'vr_width')[0][0]

    row_mask = data[:, ido_lrg] == 1.
    assert row_mask.sum() > 0

    vr_widths = get_unique(data[:, ivr_width])

    delta_vr = max(vr_widths) - min(vr_widths)
    zerrs = get_unique(data[:, izerr])
    delta_zerr = max(zerrs) - min(zerrs)
    
    nz = len(zerrs)
    nvr = len(vr_widths)

    a_ksz = np.empty((nz, nvr), dtype=np.float64)

    for iz in range(nz):
        this_zerr = zerrs[iz]
        a_ksz[iz,:] = -data[row_mask * (data[:, izerr] == this_zerr), ialpha] 

    # zerrs = res_dic['zerr_max']
    # delta_zerr = zerrs[1] - zerrs[0]

    plt.figure(dpi=300)
    plt.title('-alpha_ksz')
    plt.imshow(a_ksz[::-1,:], extent=[vr_widths[0], vr_widths[-1], zerrs[0], zerrs[-1]], aspect=delta_vr/delta_zerr)
    plt.colorbar()
    plt.xlabel('vr_width')
    plt.ylabel('zerr_max')
    om.savefig('zerr_vr_alpha.png')
    plt.close()