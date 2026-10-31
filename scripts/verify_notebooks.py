"""Execute instructor and a temporarily completed student notebook in fresh kernels.

The distributed student file is never filled with answers. Completion is in memory only.
Student widget display is skipped in this headless check; widget callbacks are tested
with pytest, and the instructor run checks the three rendered widget views.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
import nbformat
from nbclient import NotebookClient

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def validate_display_outputs(notebook,expect_widgets=True):
    formats={}
    warnings=[]
    for cell in notebook.cells:
        for output in cell.get('outputs',[]):
            for mime in output.get('data',{}):formats[mime]=formats.get(mime,0)+1
            if output.get('output_type')=='stream' and 'non-interactive' in output.get('text',''):
                warnings.append(output['text'])
    assert formats.get('image/png',0)>=7, 'Expected inline charts for all four tasks.'
    assert formats.get('image/gif',0)>=1, 'Expected a displayed highway animation.'
    if expect_widgets:
        assert formats.get('application/vnd.jupyter.widget-view+json',0)>=3, 'Expected all three interactive controls.'
    assert not warnings, 'A plotting backend suppressed notebook figures.'
    return formats


def verify(mode='smoke',train_highway=True,save_instructor=False,role='both'):
    os.environ['CH3_RUN_MODE']=mode
    os.environ['CH3_TRAIN_HIGHWAY']='1' if train_highway else '0'
    os.environ['MPLBACKEND']='module://matplotlib_inline.backend_inline'
    folder=ROOT/'results/verification';folder.mkdir(parents=True,exist_ok=True)
    os.environ.setdefault('JUPYTER_RUNTIME_DIR',str(folder/'runtime'))
    os.environ.setdefault('IPYTHONDIR',str(folder/'ipython'))
    instructor=nbformat.read(ROOT/'notebooks/chapter3_instructor.ipynb',as_version=4)
    solutions={tuple(c.metadata.get('tags',[])):c.source for c in instructor.cells
               if c.cell_type=='code' and 'student_source' in c.metadata}
    report={}
    roles=['instructor','student'] if role=='both' else [role]
    for role in roles:
        notebook=nbformat.read(ROOT/f'notebooks/chapter3_{role}.ipynb',as_version=4)
        if role=='student':
            for cell in notebook.cells:
                key=tuple(cell.metadata.get('tags',[]))
                if key in solutions:cell.source=solutions[key]
                if cell.cell_type=='code' and 'SHOW_WIDGETS = True' in cell.source:
                    cell.source=cell.source.replace('SHOW_WIDGETS = True','SHOW_WIDGETS = False')
        started=time.perf_counter()
        client=NotebookClient(notebook,timeout=600 if train_highway else 180,kernel_name='rl-chapter3',
                              store_widget_state=True,
                              resources={'metadata':{'path':str(ROOT)}})
        client.execute()
        seconds=time.perf_counter()-started
        formats=validate_display_outputs(notebook,expect_widgets=role=='instructor')
        nbformat.write(notebook,folder/f'{role}_{mode}_completed_check.ipynb')
        if role=='instructor' and save_instructor:
            nbformat.write(notebook,ROOT/'notebooks/chapter3_instructor.ipynb')
        report[role]={'mode':mode,'fresh_kernel':True,'highway_training':train_highway,
                      'code_cells':sum(c.cell_type=='code' for c in notebook.cells),
                      'seconds':round(seconds,2),'display_formats':formats,
                      'widget_display_skipped_in_headless_check':role=='student',
                      'student_answers_injected_for_check_only':role=='student'}
        print(role,report[role],flush=True)
    (folder/'report.json').write_text(json.dumps(report,indent=2))
    (folder/f'report_{mode}.json').write_text(json.dumps(report,indent=2))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',choices=['smoke','teaching'],default='smoke')
    parser.add_argument('--cached-highway',action='store_true')
    parser.add_argument('--save-instructor',action='store_true')
    parser.add_argument('--role',choices=['instructor','student','both'],default='both')
    args=parser.parse_args()
    verify(args.mode,not args.cached_highway,args.save_instructor,args.role)
