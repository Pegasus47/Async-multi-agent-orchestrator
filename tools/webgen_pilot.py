#!/usr/bin/env python3
"""Prepare a pinned WebGen-Bench task and map a generated run to evaluator input.

This adapter does not run a model judge or manufacture benchmark scores.
"""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent.parent
TASK_FILE = ROOT / 'benchmarks' / 'webgen' / 'task_000080.json'


def load_task(path=TASK_FILE):
    record = json.loads(Path(path).read_text())
    digest = hashlib.sha256(json.dumps(record['task'],sort_keys=True,ensure_ascii=False).encode()).hexdigest()
    if digest != record['provenance']['task_sha256']:
        raise ValueError('Pinned task checksum mismatch')
    return record


def prepare(output: Path, record: dict):
    output.mkdir(parents=True,exist_ok=True)
    (output/'task.json').write_text(json.dumps(record,indent=2,ensure_ascii=False))
    (output/'prompt.txt').write_text(record['task']['instruction'])
    return {'task_id':record['task']['id'], 'prompt':str(output/'prompt.txt'),
            'functional_cases':sum(case['task_category']['primary_category'] != 'Design Validation Testing' for case in record['task']['ui_instruct'])}


def map_run(manifest: Path, base_url: str, output: Path, record: dict, include_design=False, case_indices=None):
    parsed=urlparse(base_url)
    if parsed.scheme!='http' or parsed.hostname not in {'localhost','127.0.0.1','::1'} or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError('Use an HTTP localhost application URL without credentials/query/fragment')
    report=json.loads(manifest.read_text())
    task=record['task']
    if report['task'] != task['instruction']:
        raise ValueError('Run task must exactly match the pinned original generation instruction')
    if report.get('packaging',{}).get('status') != 'ready':
        raise ValueError('Run packaging is not ready; fix startup/packaging before evaluation')
    output.mkdir(parents=True,exist_ok=True)
    if case_indices is not None and any(index < 0 or index >= len(task['ui_instruct']) for index in case_indices):
        raise ValueError('Case index is outside the pinned task')
    tasks=[]; excluded=[]
    for index, case in enumerate(task['ui_instruct']):
        if case_indices is not None and index not in case_indices:
            excluded.append({'case_index':index,'reason':'not selected for this pilot'})
            continue
        if not include_design and case['task_category']['primary_category']=='Design Validation Testing':
            excluded.append({'case_index':index,'reason':'appearance scoring deferred'})
            continue
        tasks.append({'web_name':task['id'],'id':f"{task['id']}_{index}",
            'ques':f"Task: {case['task']}\n\nExpected Result: {case['expected_result']}\n\nAttempt using the UI in at most 15 interactions. Answer YES, NO, or PARTIAL based on observed evidence.",
            'web':base_url,'task':case['task'],'expected_result':case['expected_result']})
    if not tasks:
        raise ValueError('No cases selected; design cases need --include-design')
    (output/'webvoyager_tasks.jsonl').write_text(''.join(json.dumps(item,ensure_ascii=False)+'\n' for item in tasks))
    mapping={'task_id':task['id'],'run_id':report['run_id'],'application_url':base_url,
        'benchmark':record['provenance'],'included_case_ids':[item['id'] for item in tasks],
        'excluded_cases':excluded,'scope':'single-task functional-only adaptation' if not include_design else 'single-task subset',
        'full_benchmark_result':False,'startup_status':'not_verified_by_adapter','judge_status':'not_run',
        'generation_kind':'not_inferred_by_adapter','functional_score':None}
    (output/'mapping.json').write_text(json.dumps(mapping,indent=2))
    return mapping


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    subs=parser.add_subparsers(dest='command',required=True)
    prep=subs.add_parser('prepare');prep.add_argument('--output',type=Path,required=True)
    mapper=subs.add_parser('map');mapper.add_argument('--manifest',type=Path,required=True)
    mapper.add_argument('--url',required=True);mapper.add_argument('--output',type=Path,required=True)
    mapper.add_argument('--include-design',action='store_true')
    mapper.add_argument('--case-index',type=int,action='append',help='Original zero-based case index; repeat to select multiple cases')
    args=parser.parse_args();record=load_task()
    if args.command=='prepare': result=prepare(args.output,record)
    else: result=map_run(args.manifest,args.url,args.output,record,args.include_design,args.case_index)
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
