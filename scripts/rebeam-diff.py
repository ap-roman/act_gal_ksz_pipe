from pixell import enmap, enplot
from pixell.curvedsky import map2alm, almxfl, alm2map
from pixell.enmap import downgrade, upgrade
from fnl_pipe.util import OutputManager, get_yaml_dict, average_fl, downgrade_fl, get_fname, parse_act_beam
from os import sys

import numpy as np

def make_fkp_comb(ivars, cl0):
    domega = ivars[0].pixsizemap()

    ivar_sum = enmap.zeros(ivars[0].shape, wcs=ivars[0].wcs)
    for ivar in ivars:
        ivar_sum += ivar

    mask = enmap.ones(ivars[0].shape, wcs=ivars[0].wcs)
    if len(ivars) == 2:
        and_mask = (ivar[0] == 0) * (ivar[1] == 0)
        or_mask = np.logical_or(ivar[0] == 0, ivar[1] == 0).astype(float)
        mask = or_mask.copy()
        mask[and_mask] = 0

    nl_inv = ivar_sum / domega

    return mask * nl_inv / (cl0 * nl_inv + 1)

if __name__ == "__main__":
    om = OutputManager(base_path='output', title='rebeam-diff', logs=['log'], replace=True)
    printlog = om.printlog

    config_file = sys.argv[1]
    printlog('got config file ' + config_file)
    config_dict = get_yaml_dict(config_file)
    printlog('dumping config')
    for key, value in config_dict.items():
        printlog(f'{key}: {value}')
    globals().update(config_dict) # I need to stop doing this
    printlog('################## DONE ##################')

    beam_090 = parse_act_beam(beam_090_path)[:lmax + 1][1]
    map_090 = enmap.read_map(map_090_path)[0]
    ivar_090 = enmap.read_map(ivar_090_path)[0]

    beam_150 = parse_act_beam(beam_150_path)[:lmax + 1][1]
    map_150 = enmap.read_map(map_150_path)[0]
    ivar_150 = enmap.read_map(ivar_150_path)[0]

    fkp_090_kendrick = enmap.read_map(weight_090)

    fig = enplot.plot(downgrade(fkp_090_kendrick, 20), ticks=15)
    om.savefig('fkp_kendrick_090.png', fig=fig, mode='pixell')

    fkp_090 = make_fkp_comb([ivar_090,], cl0)

    fig = enplot.plot(downgrade(fkp_090, 20), ticks=15)
    om.savefig('fkp_090.png', fig=fig, mode='pixell')

    fkp_combined = make_fkp_comb([ivar_090, ivar_150], cl0)

    fig = enplot.plot(downgrade(fkp_combined, 20), ticks=15)
    om.savefig('fkp_090_150.png', fig=fig, mode='pixell')

    enmap.write_map(weight_out_path, fkp_combined)

    map_150_rebeam = enmap.zeros(map_150.shape, wcs=map_150.wcs)

    printlog('rebeaming the 150 map to the 90 GHz beam')
    map_150_rebeam = alm2map(almxfl(map2alm(map_150, lmax=lmax), lfilter=beam_090/beam_150), map=map_150_rebeam)
    printlog('done')

    map_diff = map_150_rebeam - map_090

    enmap.write_map(map_out_path, map_diff)