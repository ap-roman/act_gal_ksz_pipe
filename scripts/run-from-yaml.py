# run many scripts from a yaml config file
# at the core, a valid config file has the structure:
# 
# base_path: <the base path from which paths are referenced, usually the repository folder>
# tmp_dir: <a directory in which to place temporary yaml files>
# defaults: <a yaml file contain default fields/values, optional>
# scripts:
#   <script1>:
#       defaults: <optional default override>
#       base_path: <path>
#       <params>: <vals>
#   ...
#   <script_n>:
#       ...

import sys
import yaml
import uuid
import subprocess

from fnl_pipe.util import get_yaml_dict, validate_script_config, validate_config


supported_scripts = ['phack-bootstrap.py', 'explore-multifreq-bootstrap.py', 'make-paper.py',
                     'make-mf-bootstrap.py']


def run_single_config(yaml_config):
    yaml_in = get_yaml_dict(yaml_config, vad_fun=lambda x: validate_config(x, supported_scripts))
    base_path = yaml_in['base_path']
    tmp_dir = yaml_in['tmp_dir']
    has_defaults = 'defaults' in yaml_in.keys()
    
    defaults = {}
    if has_defaults:
        defaults = get_yaml_dict(yaml_in['defaults'])

    scripts_dict = yaml_in['scripts']
    for scriptname in scripts_dict.keys():
        script_dict = scripts_dict[scriptname]
        validate_script_config(script_dict)

        has_defaults = 'defaults' in script_dict.keys()
        if has_defaults:
            defaults = get_yaml_dict(script_dict)

        tmp_dict = defaults.copy()

        excluded_keys = ['defaults', 'base_path']
        for key, value in script_dict.items():
            if key != excluded_keys: tmp_dict[key] = value

        tmp_name = f'{uuid.uuid4()}.yaml'

        script_base = base_path + script_dict['base_path']
        script_path = script_base + scriptname

        tmp_path = tmp_dir + tmp_name

        with open(tmp_path, 'w') as file:
            file.write(yaml.dump(tmp_dict))

        subprocess.call(["python3", script_path, tmp_path])


if __name__ == "__main__":
    config_files = sys.argv[1:]

    for cfile in config_files:
        run_single_config(cfile)