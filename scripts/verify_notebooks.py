"""Execute instructor and a temporarily completed student notebook in fresh kernels.

The distributed student file is never filled with answers. Completion is in memory only.
Student widget display is skipped in this headless check; widget callbacks are tested
with pytest, and the instructor run checks rendered widget views. --core-only checks
the recording route and result export without executing the self-study appendix.
"""
import argparse
import base64
import io
import json
import os
import sys
import time
from pathlib import Path
import nbformat
import numpy as np
from PIL import Image
from nbclient import NotebookClient
from nbclient.output_widget import OutputWidget
from notebook_preview import static_preview

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


class SnapshotOutputWidget(OutputWidget):
    """Capture Output widget contents without echoing them back to the kernel.

    nbclient already writes captured output into its serialized widget state.
    Sending every snapshot back over the shell channel can create a feedback
    loop with ipywidgets; a headless verifier needs only the serialized snapshot.
    Live JupyterLab widgets and their callbacks are unaffected.
    """
    def sync_state(self):
        pass


def validate_display_outputs(notebook,expect_widgets=True,core_only=False):
    formats={}
    warnings=[]
    for cell in notebook.cells:
        for output in cell.get('outputs',[]):
            for mime in output.get('data',{}):formats[mime]=formats.get(mime,0)+1
            if 'image/gif' in output.get('data',{}):
                # A GIF MIME type can still be a single all-black frame.
                with Image.open(io.BytesIO(base64.b64decode(output['data']['image/gif']))) as animation:
                    assert animation.n_frames > 1, 'Expected actual moving highway frames.'
                    for index in [0, animation.n_frames - 1]:
                        animation.seek(index)
                        assert np.asarray(animation.convert('RGB')).std() > 10, 'Highway animation is blank.'
            if output.get('output_type')=='stream' and 'non-interactive' in output.get('text',''):
                warnings.append(output['text'])
    expected_pngs=5 if core_only else 6
    assert formats.get('image/png',0)>=expected_pngs, 'Expected cafe, highway scene/learning/outcome and spread charts; full run also checks dilemma.'
    assert formats.get('image/gif',0)>=1, 'Expected the study DQN animation; older playback is optional.'
    if expect_widgets:
        expected_widgets=2 if core_only else 3
        assert formats.get('application/vnd.jupyter.widget-view+json',0)>=expected_widgets, 'Expected cafe/Spread controls; full run also checks dilemma.'
    assert not warnings, 'A plotting backend suppressed notebook figures.'
    return formats


def verify(mode='smoke',train_highway=True,save_instructor=False,role='both',core_only=False):
    if core_only and save_instructor:
        raise ValueError('Use the full run to save instructor previews; a core-only copy omits appendix cells.')
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
        if core_only:
            export=next(c for c in notebook.cells if c.id=='2fe4c806')
            finish=next(i for i,c in enumerate(notebook.cells) if c.id=='recording-finish')
            notebook.cells=notebook.cells[:finish+1]
            # Check that exporting after recording needs no appendix variables.
            export.source=("assert 'pd_metadata' not in globals() and 'tournament' not in globals()\n"
                           +export.source+"\nassert manifest['prisoners_dilemma'] is None\n"
                           "assert not (OUTPUT_DIR/'prisoners_dilemma_tournament.csv').exists()\n")
            notebook.cells.append(export)
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
        client.widget_registry={'@jupyter-widgets/output':{'OutputModel':SnapshotOutputWidget}}
        client.execute()
        seconds=time.perf_counter()-started
        formats=validate_display_outputs(notebook,expect_widgets=role=='instructor',core_only=core_only)
        suffix='_core' if core_only else ''
        nbformat.write(notebook,folder/f'{role}_{mode}{suffix}_completed_check.ipynb')
        if role=='instructor' and save_instructor:
            preview=static_preview(notebook)
            nbformat.validate(preview)
            assert 'widgets' not in preview.metadata
            assert not any('application/vnd.jupyter.widget-view+json' in output.get('data',{})
                           for cell in preview.cells for output in cell.get('outputs',[]))
            nbformat.write(preview,ROOT/'notebooks/chapter3_instructor.ipynb')
        report[role]={'mode':mode,'fresh_kernel':True,'highway_training':train_highway,
                      'scope':'recording route plus export' if core_only else 'full notebook including appendix',
                      'code_cells':sum(c.cell_type=='code' for c in notebook.cells),
                      'seconds':round(seconds,2),'display_formats':formats,
                      'widget_display_skipped_in_headless_check':role=='student',
                      'student_answers_injected_for_check_only':role=='student'}
        print(role,report[role],flush=True)
    (folder/f'report{suffix}.json').write_text(json.dumps(report,indent=2))
    (folder/f'report_{mode}{suffix}.json').write_text(json.dumps(report,indent=2))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',choices=['smoke','teaching'],default='smoke')
    parser.add_argument('--cached-highway',action='store_true')
    parser.add_argument('--save-instructor',action='store_true')
    parser.add_argument('--core-only',action='store_true',help='Check recording route and export while skipping the self-study appendix.')
    parser.add_argument('--role',choices=['instructor','student','both'],default='both')
    args=parser.parse_args()
    verify(args.mode,not args.cached_highway,args.save_instructor,args.role,args.core_only)
