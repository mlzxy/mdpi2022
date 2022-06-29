#!/usr/bin/python3
import re
import os.path as osp
import os 
import jstyleson as json
import sys
import importlib.util
import copy


_global_ = {
    'active_exp_id': ''
}

def require(path, default=False):
    """
    support loads json and python
    with suffix as run id
    """
    # transform path to absolute path
    if not path.startswith('/'):
        parent_scope = sys._getframe(1).f_globals
        if '__file__' in parent_scope:
            parent_path = parent_scope['__file__']
            caller_folder = osp.dirname(parent_path)
        else:
            caller_folder = osp.abspath('')
        path = osp.abspath(osp.join(caller_folder, path))

    # get run_id 
    folder = osp.dirname(path)
    bname = osp.basename(path)
    run_id = '0'
    if '#' in bname:
        bname, run_id = bname.split('#')
        path = osp.join(folder, bname)
    
    # guess ext
    if osp.isdir(path):
        path = osp.join(path, "base")
    _, ext = osp.splitext(path)
    if len(ext) == 0:
        if osp.exists(path + ".json"):
            path += '.json'
        elif osp.exists(path + ".py"):
            path += '.py'
        else:
            raise FileNotFoundError(path + "{.json,.py}")
        
    if path.endswith(".json"):
        with open(path) as f:
            conf = json.load(f)
        if 'name' not in conf:
            print(f'WARNING: name is not set in {path}')
        else:
            _global_['active_exp_id'] = conf['name']
            if run_id is not None:
                conf['name'] += ('/' + str(run_id))
        return conf 
    else:
        module_name = re.sub('/+', '/', path.strip('.py')).replace('/', '.')
        spec = importlib.util.spec_from_file_location(module_name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        if default:
            exports = copy.deepcopy(mod.exports)
            if 'name' not in exports:
                print(f'WARNING: name is not set in `exports` of {path}')
            else:
                _global_['active_exp_id'] = exports['name']
                exports['name'] += ('/' + str(run_id))
            return exports
        else:
            setattr(mod, 'run_id', str(run_id))
            return mod

def ensure_dir(path):
    path = osp.abspath(path)
    if not osp.exists(path):
        os.makedirs(path)

def dumps_json(data):
    return json.dumps(data, indent=4)

def dump_json(path: str, data: object):
    ensure_dir(osp.dirname(path))
    with open(path, "w") as f:
        print(f"writing to {path}")
        f.write(dumps_json(data))