import os
import csv
import json
import torch
import shutil
import argparse
import subprocess
import libcst as cst
from libcst.metadata import PositionProvider
from libcst._position import CodePosition
from collections import OrderedDict

from sven.utils import set_seed, set_logging, set_devices
from sven.constant import BINARY_LABELS, MODEL_DIRS, CWES_DICT

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_name', type=str, required=True)

    parser.add_argument('--eval_type', type=str, choices=['trained', 'trained_subset', 'prompts', 'gen_1', 'gen_2'], default='trained')
    parser.add_argument('--vul_type', type=str, default=None)
    parser.add_argument('--model_type', type=str, choices=['lm', 'prefix', 'text'], default='prefix')
    parser.add_argument('--model_dir', type=str, default=None)

    parser.add_argument('--data_dir', type=str, default='../data_eval')
    parser.add_argument('--output_dir', type=str, default='../experiments/sec_eval')

    parser.add_argument('--num_gen', type=int, default=25)
    parser.add_argument('--temp', type=float, default=0.4)
    parser.add_argument('--max_gen_len', type=int, default=300)
    parser.add_argument('--top_p', type=float, default=0.95)

    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--gen_only', action='store_true', help='generate samples only, skip CodeQL analysis')
    parser.add_argument('--codeql_only', action='store_true', help='run CodeQL on existing generated samples, skip generation')
    args = parser.parse_args()

    if args.model_type == 'lm':
        if args.model_dir is None:
            args.model_dir = '2b'
        if args.model_dir in MODEL_DIRS:
            args.model_dir = MODEL_DIRS[args.model_dir]

    args.output_dir = os.path.join(args.output_dir, args.output_name, args.eval_type)
    args.data_dir = os.path.join(args.data_dir, args.eval_type)

    return args

def get_evaler(args):
    # Lazy-import: pulls in transformers/torch, which the --codeql_only path
    # doesn't need. Keeps codeql_env (Py3.8, libcst-only) viable.
    from sven.evaler import LMEvaler, PrefixEvaler, TextPromptEvaler
    if args.model_type == 'lm':
        evaler = LMEvaler(args)
        controls = ['orig']
    elif args.model_type == 'prefix':
        evaler = PrefixEvaler(args)
        controls = BINARY_LABELS
    elif args.model_type == 'text':
        evaler = TextPromptEvaler(args)
        controls = BINARY_LABELS
    else:
        raise NotImplementedError()

    return evaler, controls

def codeql_create_db(info, out_src_dir, out_db_dir):
    if info['language'] == 'py':
        cmd = '../codeql/codeql database create {} --quiet --language=python --overwrite --source-root {}'
        cmd = cmd.format(out_db_dir, out_src_dir)
    elif info['language'] == 'c':
        cmd = '../codeql/codeql database create {} --quiet --language=cpp --overwrite --command="make -B" --source-root {}'
        cmd = cmd.format(out_db_dir, out_src_dir)
    else:
        raise NotImplementedError()
    r = subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if r.returncode != 0:
        raise RuntimeError(f'codeql_create_db failed:\n{r.stderr.decode()}')

def _sven_additional_packs():
    # Packs are stored in <project_root>/codeql/packages (set up by setup_codeql.sh
    # with --dir=codeql/packages) so multiple cluster instances don't share ~/.codeql.
    # CodeQL v2.11.1 doesn't follow symlinks when scanning for packs, so pass the
    # exact version directories as a colon-separated list to avoid "found in 2 locations" errors.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base = os.path.join(script_dir, '..', 'codeql', 'packages', 'codeql')
    packs = [
        ('cpp-all',    '0.7.1'),
        ('python-all', '0.6.2'),
        ('ssa',        '0.0.16'),
        ('tutorial',   '0.0.9'),
        ('util',       '0.0.9'),
        ('regex',      '0.0.12'),
    ]
    return ':'.join(os.path.join(base, name, ver) for name, ver in packs)

def codeql_analyze(info, out_db_dir, out_csv_path):
    if info['language'] in ('py', 'c'):
        cmd = '../codeql/codeql database analyze {} {} --quiet --format=csv --output={} --additional-packs={}'
        cmd = cmd.format(out_db_dir, info['check_ql'], out_csv_path, _sven_additional_packs())
    else:
        raise NotImplementedError()
    r = subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if r.returncode != 0:
        raise RuntimeError(f'codeql_analyze failed:\n{r.stderr.decode()}')

class CWE78Visitor(cst.CSTVisitor):
    METADATA_DEPENDENCIES = (PositionProvider,)

    def __init__(self, src, start, end):
        self.list_vars = set()
        self.src = src
        self.start = start
        self.end = end
        self.fp = False

    def visit_Assign(self, node):
        if len(node.targets) != 1: return
        if not isinstance(node.targets[0].target, cst.Name): return
        target_name = node.targets[0].target.value
        if isinstance(node.value, cst.List):
            if len(node.value.elements) == 0: return
            if not isinstance(node.value.elements[0].value, cst.BaseString): return
            self.list_vars.add(target_name)
        elif isinstance(node.value, cst.Name):
            if node.value.value in self.list_vars:
                self.list_vars.add(target_name)
        elif isinstance(node.value, cst.BinaryOperation):
            if isinstance(node.value.left, cst.List):
                self.list_vars.add(target_name)
            elif isinstance(node.value.left, cst.Name) and node.value.left.value in self.list_vars:
                self.list_vars.add(target_name)
            if isinstance(node.value.right, cst.List):
                self.list_vars.add(target_name)
            elif isinstance(node.value.right, cst.Name) and node.value.right.value in self.list_vars:
                self.list_vars.add(target_name)

    def visit_Name(self, node):
        pos = self.get_metadata(PositionProvider, node)
        if self.start.line != pos.start.line: return
        if self.start.column != pos.start.column: return
        if self.end.line != pos.end.line: return
        if self.end.column != pos.end.column: return
        assert pos.start.line == pos.end.line
        if node.value in self.list_vars:
            self.fp = True

def filter_cwe78_fps(s_out_dir, control):
    csv_path = os.path.join(s_out_dir, f'{control}_codeql.csv')
    out_src_dir = os.path.join(s_out_dir, f'{control}_output')
    with open(csv_path) as csv_f:
        lines = csv_f.readlines()
    shutil.copy2(csv_path, csv_path+'.fp')
    with open(csv_path, 'w') as csv_f:
        for line in lines:
            row = line.strip().split(',')
            if len(row) < 5: continue
            out_src_fname = row[-5].replace('/', '').strip('"')
            out_src_path = os.path.join(out_src_dir, out_src_fname)
            with open(out_src_path) as f:
                src = f.read()
            start = CodePosition(int(row[-4].strip('"')), int(row[-3].strip('"'))-1)
            end = CodePosition(int(row[-2].strip('"')), int(row[-1].strip('"')))
            visitor = CWE78Visitor(src, start, end)
            tree = cst.parse_module(src)
            wrapper = cst.MetadataWrapper(tree)
            wrapper.visit(visitor)
            if not visitor.fp:
                csv_f.write(line)

def eval_single(args, evaler, controls, output_dir, data_dir, vul_type, scenario):
    s_out_dir = os.path.join(output_dir, scenario)
    os.makedirs(s_out_dir, exist_ok=True)
    s_in_dir = os.path.join(data_dir, scenario)
    with open(os.path.join(s_in_dir, 'info.json')) as f:
        info = json.load(f)
    with open(os.path.join(s_in_dir, 'file_context.'+info['language'])) as f:
        file_context = f.read()
    with open(os.path.join(s_in_dir, 'func_context.'+info['language'])) as f:
        func_context = f.read()

    for control_id, control in enumerate(controls):
        set_seed(args)
        with torch.no_grad():
            outputs, output_ids, dup_srcs, non_parsed_srcs = evaler.sample(file_context, func_context, control_id, info['language'])

        out_src_dir = os.path.join(s_out_dir, f'{control}_output')
        os.makedirs(out_src_dir, exist_ok=True)
        output_ids_j = OrderedDict()
        all_fnames = set()
        for i, (output, output_id) in enumerate(zip(outputs, output_ids)):
            fname = f'{str(i).zfill(2)}.'+info['language']
            all_fnames.add(fname)
            with open(os.path.join(out_src_dir, fname), 'w') as f:
                f.write(output)
            output_ids_j[fname] = output_id
        with open(os.path.join(s_out_dir, f'{control}_output_ids.json'), 'w') as f:
            json.dump(output_ids_j, f, indent=2)
        if info['language'] == 'c':
            shutil.copy2('Makefile', out_src_dir)

        for srcs, name in [(dup_srcs, 'dup'), (non_parsed_srcs, 'non_parsed')]:
            src_dir = os.path.join(s_out_dir, f'{control}_{name}')
            os.makedirs(src_dir, exist_ok=True)
            for i, src in enumerate(srcs):
                fname = f'{str(i).zfill(2)}.'+info['language']
                with open(os.path.join(src_dir, fname), 'w') as f:
                    f.write(src)

        vuls = set()
        if len(outputs) != 0 and not args.gen_only:
            csv_path = os.path.join(s_out_dir, f'{control}_codeql.csv')
            db_path = os.path.join(s_out_dir, f'{control}_codeql_db')
            codeql_create_db(info, out_src_dir, db_path)
            codeql_analyze(info, db_path, csv_path)
            if vul_type == 'cwe-078':
                filter_cwe78_fps(s_out_dir, control)
            with open(csv_path) as csv_f:
                reader = csv.reader(csv_f)
                for row in reader:
                    if len(row) < 5: continue
                    out_src_fname = row[-5].replace('/', '')
                    vuls.add(out_src_fname)
        secs = all_fnames - vuls

        d = OrderedDict()
        d['vul_type'] = vul_type
        d['scenario'] = scenario
        d['control'] = control
        d['total'] = len(all_fnames)
        d['sec'] = len(secs)
        d['vul'] = len(vuls)
        d['dup'] = len(dup_srcs)
        d['non_parsed'] = len(non_parsed_srcs)
        d['model_type'] = args.model_type
        d['model_dir'] = args.model_dir
        d['temp'] = args.temp

        yield d

def codeql_only_single(args, controls, output_dir, data_dir, vul_type, scenario):
    s_out_dir = os.path.join(output_dir, scenario)
    s_in_dir = os.path.join(data_dir, scenario)
    with open(os.path.join(s_in_dir, 'info.json')) as f:
        info = json.load(f)

    for control in controls:
        out_src_dir = os.path.join(s_out_dir, f'{control}_output')
        all_fnames = set(f for f in os.listdir(out_src_dir)
                         if f.endswith('.'+info['language'])) if os.path.isdir(out_src_dir) else set()

        dup_dir = os.path.join(s_out_dir, f'{control}_dup')
        non_parsed_dir = os.path.join(s_out_dir, f'{control}_non_parsed')
        n_dup = len(os.listdir(dup_dir)) if os.path.isdir(dup_dir) else 0
        n_non_parsed = len(os.listdir(non_parsed_dir)) if os.path.isdir(non_parsed_dir) else 0

        if info['language'] == 'c' and os.path.isdir(out_src_dir):
            shutil.copy2('Makefile', out_src_dir)

        vuls = set()
        if len(all_fnames) != 0:
            csv_path = os.path.join(s_out_dir, f'{control}_codeql.csv')
            db_path = os.path.join(s_out_dir, f'{control}_codeql_db')
            codeql_create_db(info, out_src_dir, db_path)
            codeql_analyze(info, db_path, csv_path)
            if vul_type == 'cwe-078':
                filter_cwe78_fps(s_out_dir, control)
            with open(csv_path) as csv_f:
                reader = csv.reader(csv_f)
                for row in reader:
                    if len(row) < 5: continue
                    out_src_fname = row[-5].replace('/', '')
                    vuls.add(out_src_fname)
        secs = all_fnames - vuls

        d = OrderedDict()
        d['vul_type'] = vul_type
        d['scenario'] = scenario
        d['control'] = control
        d['total'] = len(all_fnames)
        d['sec'] = len(secs)
        d['vul'] = len(vuls)
        d['dup'] = n_dup
        d['non_parsed'] = n_non_parsed
        d['model_type'] = args.model_type
        d['model_dir'] = args.model_dir
        d['temp'] = args.temp

        yield d

def eval_vul(args, evaler, controls, vul_types):
    for vul_type in vul_types:
        data_dir = os.path.join(args.data_dir, vul_type)
        output_dir = os.path.join(args.output_dir, vul_type)
        os.makedirs(output_dir, exist_ok=True)

        with open(os.path.join(output_dir, 'result.jsonl'), 'w') as f:
            for scenario in list(sorted(os.listdir(data_dir))):
                for d in eval_single(args, evaler, controls, output_dir, data_dir, vul_type, scenario):
                    s = json.dumps(d)
                    args.logger.info(s)
                    f.write(s+'\n')

def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_logging(args, None)
    set_devices(args)
    set_seed(args)
    args.logger.info(f'args: {args}')

    assert args.eval_type in CWES_DICT
    vul_types = [args.vul_type] if args.vul_type is not None else CWES_DICT[args.eval_type]

    if args.codeql_only:
        controls = ['orig'] if args.model_type == 'lm' else BINARY_LABELS
        for vul_type in vul_types:
            data_dir = os.path.join(args.data_dir, vul_type)
            output_dir = os.path.join(args.output_dir, vul_type)
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, 'result.jsonl'), 'w') as f:
                for scenario in sorted(os.listdir(data_dir)):
                    for d in codeql_only_single(args, controls, output_dir, data_dir, vul_type, scenario):
                        s = json.dumps(d)
                        args.logger.info(s)
                        f.write(s+'\n')
        return

    evaler, controls = get_evaler(args)
    eval_vul(args, evaler, controls, vul_types)

if __name__ == '__main__':
    main()