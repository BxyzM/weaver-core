import math
import os
import re
import numpy as np
import awkward as ak
import tqdm
import traceback
from .tools import _concat
from ..logger import _logger, warn_n_times


def _read_hdf5(filepath, branches, load_range=None):
    import tables
    tables.set_blosc_max_threads(4)
    with tables.open_file(filepath) as f:
        outputs = {k: getattr(f.root, k)[:] for k in branches}
    if load_range is None:
        load_range = (0, 1)
    start = math.trunc(load_range[0] * len(outputs[branches[0]]))
    stop = max(start + 1, math.trunc(load_range[1] * len(outputs[branches[0]])))
    for k, v in outputs.items():
        outputs[k] = v[start:stop]
    return ak.Array(outputs)


def _read_root(filepath, branches, load_range=None, treename=None, branch_magic=None):
    import uproot
    with uproot.open(filepath) as f:
        if treename is None:
            treenames = set([k.split(';')[0] for k, v in f.items() if getattr(v, 'classname', '') == 'TTree'])
            if len(treenames) == 1:
                treename = treenames.pop()
            else:
                raise RuntimeError(
                    'Need to specify `treename` as more than one trees are found in file %s: %s' %
                    (filepath, str(treenames)))
        
        # Handle cycle numbers: find the actual key in the file
        # ROOT files may have keys like 'tree;1' or 'tree;1;1' (double cycle from mktree)
        tree_key = None
        for key in f.keys():
            if key.split(';')[0] == treename:
                tree_key = key
                break
        
        if tree_key is None:
            raise RuntimeError(f'Tree {treename} not found in file {filepath}. Available keys: {list(f.keys())}')
        
        tree = f[tree_key]
        if load_range is not None:
            start = math.trunc(load_range[0] * tree.num_entries)
            stop = max(start + 1, math.trunc(load_range[1] * tree.num_entries))
        else:
            start, stop = None, None
        if branch_magic is not None:
            branch_dict = {}
            for name in branches:
                decoded_name = name
                for src, tgt in branch_magic.items():
                    if src in decoded_name:
                        decoded_name = decoded_name.replace(src, tgt)
                branch_dict[name] = decoded_name
            outputs = tree.arrays(filter_name=list(branch_dict.values()), entry_start=start, entry_stop=stop)
            for name, decoded_name in branch_dict.items():
                if name != decoded_name:
                    outputs[name] = outputs[decoded_name]
        else:
            outputs = tree.arrays(filter_name=branches, entry_start=start, entry_stop=stop)
    return outputs


def _resolve_root_tree(file_handle, treename):
    file_path = getattr(file_handle, 'file_path', '<root>')
    if treename is None:
        treenames = set([k.split(';')[0] for k, v in file_handle.items() if getattr(v, 'classname', '') == 'TTree'])
        if len(treenames) == 1:
            treename = treenames.pop()
        else:
            raise RuntimeError(
                'Need to specify `treename` as more than one trees are found in file %s: %s' %
                (file_path, str(treenames)))

    tree_key = None
    for key in file_handle.keys():
        if key.split(';')[0] == treename:
            tree_key = key
            break

    if tree_key is None:
        raise RuntimeError(f'Tree {treename} not found in file {file_path}. '
                           f'Available keys: {list(file_handle.keys())}')
    return file_handle[tree_key], treename


def _apply_branch_magic(name, branch_magic):
    if branch_magic is None:
        return name
    decoded_name = name
    for src, tgt in branch_magic.items():
        if src in decoded_name:
            decoded_name = decoded_name.replace(src, tgt)
    return decoded_name


def _derive_qfim_path(root_path, qfim_cfg):
    h5_ext = qfim_cfg.get('h5_ext', '.h5')
    if qfim_cfg.get('h5_path'):
        base = qfim_cfg['h5_path']
        if os.path.isdir(base):
            stem, _ = os.path.splitext(os.path.basename(root_path))
            return os.path.join(base, stem + h5_ext)
        return base
    if qfim_cfg.get('h5_dir'):
        stem, _ = os.path.splitext(os.path.basename(root_path))
        return os.path.join(qfim_cfg['h5_dir'], stem + h5_ext)
    if qfim_cfg.get('h5_replace'):
        repl = qfim_cfg['h5_replace']
        if isinstance(repl, dict):
            pattern = repl.get('pattern', '.root')
            replacement = repl.get('replace', h5_ext)
        else:
            pattern, replacement = repl
        return re.sub(pattern, replacement, root_path)
    stem, _ = os.path.splitext(root_path)
    return stem + h5_ext


def _qfim_feature_indices(names, prefix):
    indices = {}
    for name in names:
        m = re.match(r'^%s(\\d+)$' % re.escape(prefix), name)
        if not m:
            raise RuntimeError(f'QFIM branch name `{name}` does not match prefix `{prefix}`.')
        indices[name] = int(m.group(1))
    return indices


def _read_qfim_sidecar(root_path, names, load_range, qfim_cfg, n_entries):
    import tables
    qfim_path = _derive_qfim_path(root_path, qfim_cfg)
    qfim_key = qfim_cfg.get('qfim_key', 'qfim_matrices')
    particle_axis = int(qfim_cfg.get('particle_axis', 1))
    feature_axis = int(qfim_cfg.get('feature_axis', 2))
    with tables.open_file(qfim_path) as f:
        if not hasattr(f.root, qfim_key):
            raise RuntimeError(f'HDF5 key `{qfim_key}` not found in {qfim_path}')
        node = getattr(f.root, qfim_key)
        if len(node) != n_entries:
            raise RuntimeError(
                f'Entry count mismatch for {root_path}: root={n_entries}, qfim={len(node)}')
        if load_range is None:
            load_range = (0, 1)
        start = math.trunc(load_range[0] * n_entries)
        stop = max(start + 1, math.trunc(load_range[1] * n_entries))
        qfim = node[start:stop]
    if qfim.ndim == 2:
        qfim = qfim[:, None, :]
    elif qfim.ndim != 3:
        raise RuntimeError(f'qfim_matrices must be 2D or 3D, got shape {qfim.shape}')
    if (particle_axis, feature_axis) != (1, 2):
        qfim = np.moveaxis(qfim, (particle_axis, feature_axis), (1, 2))
    n_features = qfim.shape[2]
    idx_map = _qfim_feature_indices(names, qfim_cfg.get('feature_prefix', 'qfim_f'))
    out = {}
    for name, idx in idx_map.items():
        if idx >= n_features:
            raise RuntimeError(
                f'QFIM feature index {idx} out of range (n_features={n_features}) for {name}')
        out[name] = ak.Array(qfim[:, :, idx])
    return out


def _read_awkd(filepath, branches, load_range=None):
    import awkward0
    with awkward0.load(filepath) as f:
        outputs = {k: f[k] for k in branches}
    if load_range is None:
        load_range = (0, 1)
    start = math.trunc(load_range[0] * len(outputs[branches[0]]))
    stop = max(start + 1, math.trunc(load_range[1] * len(outputs[branches[0]])))
    for k, v in outputs.items():
        outputs[k] = ak.from_awkward0(v[start:stop])
    return ak.Array(outputs)


def _read_parquet(filepath, branches, load_range=None):
    outputs = ak.from_parquet(filepath, columns=branches)
    if load_range is not None:
        start = math.trunc(load_range[0] * len(outputs))
        stop = max(start + 1, math.trunc(load_range[1] * len(outputs)))
        outputs = outputs[start:stop]
    return outputs


def _read_files(filelist, branches, load_range=None, show_progressbar=False, file_magic=None, **kwargs):
    import os
    branches = list(branches)
    table = []
    if show_progressbar:
        filelist = tqdm.tqdm(filelist)
    for filepath in filelist:
        ext = os.path.splitext(filepath)[1]
        if ext not in ('.h5', '.root', '.awkd', '.parquet'):
            raise RuntimeError('File %s of type `%s` is not supported!' % (filepath, ext))
        try:
            if ext == '.h5':
                a = _read_hdf5(filepath, branches, load_range=load_range)
            elif ext == '.root':
                qfim_cfg = kwargs.get('qfim', None)
                if qfim_cfg:
                    import uproot
                    with uproot.open(filepath) as f:
                        tree, treename = _resolve_root_tree(f, kwargs.get('treename', None))
                        tree_branches = set(tree.keys())
                        root_branches = []
                        missing = []
                        for name in branches:
                            decoded = _apply_branch_magic(name, kwargs.get('branch_magic', None))
                            if decoded in tree_branches:
                                root_branches.append(name)
                            else:
                                missing.append(name)
                        if not root_branches:
                            raise RuntimeError(
                                f'No ROOT branches found in {filepath} for requested inputs.')
                        if missing:
                            prefix = qfim_cfg.get('feature_prefix', 'qfim_f')
                            non_qfim = [m for m in missing if not m.startswith(prefix)]
                            if non_qfim:
                                raise RuntimeError(
                                    f'Missing branches in ROOT file (not QFIM): {sorted(non_qfim)}')
                    a = _read_root(filepath, root_branches, load_range=load_range,
                                   treename=treename,
                                   branch_magic=kwargs.get('branch_magic', None))
                    if missing:
                        qfim_fields = _read_qfim_sidecar(
                            filepath, missing, load_range, qfim_cfg, n_entries=tree.num_entries)
                        for k, v in qfim_fields.items():
                            a[k] = v
                else:
                    a = _read_root(filepath, branches, load_range=load_range,
                                   treename=kwargs.get('treename', None),
                                   branch_magic=kwargs.get('branch_magic', None))
            elif ext == '.awkd':
                a = _read_awkd(filepath, branches, load_range=load_range)
            elif ext == '.parquet':
                a = _read_parquet(filepath, branches, load_range=load_range)
        except Exception as e:
            a = None
            _logger.error('When reading file %s:', filepath)
            _logger.error(traceback.format_exc())
        if a is not None:
            if file_magic is not None:
                import re
                for var, value_dict in file_magic.items():
                    if var in a.fields:
                        warn_n_times(f'Var `{var}` already defined in the arrays '
                                     f'but will be OVERWRITTEN by file_magic {value_dict}.')
                    a[var] = 0
                    for fn_pattern, value in value_dict.items():
                        if re.search(fn_pattern, filepath):
                            a[var] = value
                            break
            table.append(a)
    table = _concat(table)  # ak.Array
    if len(table) == 0:
        raise RuntimeError(f'Zero entries loaded when reading files {filelist} with `load_range`={load_range}.')
    return table


def _write_root(file, table, treename='Events', compression=-1, step=1048576):
    import uproot
    if compression == -1:
        compression = uproot.LZ4(4)
    with uproot.recreate(file, compression=compression) as fout:
        tree = fout.mktree(treename, {k: table[k].type for k in table.fields})
        start = 0
        while start < len(table[table.fields[0]]) - 1:
            tree.extend({k: table[k][start:start + step] for k in table.fields})
            start += step
