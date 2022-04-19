from fnl_pipe.catalog import get_files, save_gal_summaries

cat_base = '/home/aroman/data/vr_source/v01/desils/'
sdss_base = '/home/aroman/data/vr_source/v01/sdss/'

# v0_base = '/home/aroman/data/vr_source/v00/'
# v0_set = get_files(v0_base)
# north_cmass = cat_base + 'v01_desils_north_cmass.h5'
# south_cmass = cat_base + 'v01_desils_south_cmass.h5'
# north_lowz = cat_base + 'v01_desils_north_lowz.h5'
# south_lowz = cat_base + 'v01_desils_south_lowz.h5'

# v0_north_cmass = v0_base + 'v00_sdss_cmass_north.h5'
# v0_south_cmass = v0_base + 'v00_sdss_cmass_south.h5'
# v0_north_lowz = v0_base + 'v00_sdss_lowz_north.h5'
# v0_south_lowz = v0_base + 'v00_sdss_lowz_south.h5'

sdss_cmass_north = sdss_base + 'v01_sdss_cmass_north.h5'
sdss_cmass_south = sdss_base + 'v01_sdss_cmass_south.h5'
sdss_lowz_north = sdss_base + 'v01_sdss_lowz_north.h5'
sdss_lowz_south = sdss_base + 'v01_sdss_lowz_south.h5'

# catalog_sets = [[north_cmass, north_lowz, south_cmass, south_lowz],
#                 [north_cmass, north_lowz],
#                 [south_cmass, south_lowz],
#                 [north_cmass, south_cmass],
#                 [north_lowz, south_lowz],
#                 [south_cmass,],
#                 [north_cmass,],
#                 v0_set,
#                 [v0_north_cmass,],
#                 [v0_south_cmass,],
#                 [v0_north_lowz,],
#                 [v0_south_lowz,]]

catalog_sets = [[sdss_cmass_north, sdss_cmass_south, sdss_lowz_north, sdss_lowz_south],
                [sdss_cmass_north, sdss_cmass_south],
                [sdss_lowz_north, sdss_lowz_south],
                [sdss_cmass_north, sdss_lowz_north],
                [sdss_cmass_south, sdss_lowz_south],
                [sdss_cmass_north,]]

set_names = ['sdss_all', 'sdss_cmass', 'sdss_lowz', 'sdss_north', 'sdss_south', 'sdss_cmass_north']

# set_names = ['all', 'north', 'south', 'cmass', 'lowz', 'south_cmass', 'north_cmass',
#              'v0_all', 'v0_cmass_north', 'v0_cmass_south', 'v0_lowz_north', 'v0_lowz_south']

def do_gal_summaries():
    # catalog_files = [cat_base + 'v01_desils_north_cmass.h5',
    #                  cat_base + 'v01_desils_south_cmass.h5',
    #                  cat_base + 'v01_desils_north_lowz.h5',
    #                  cat_base + 'v01_desils_south_lowz.h5']
    gal_out_base = '/home/aroman/data/vr_summaries/v01_'
    ref_map_path = '/home/aroman/data/act/act_planck_s08_s19_cmb_f150_daynight_srcfree_map.fits'
    # catalog_files = [cat_base + 'v01_desils_north_cmass.h5',]
    for cat_set, set_name in zip(catalog_sets, set_names):
        gal_out_path = gal_out_base + set_name + '.h5'
        save_gal_summaries(ref_map_path, cat_set, gal_out_path)

if __name__ == "__main__":
    do_gal_summaries()