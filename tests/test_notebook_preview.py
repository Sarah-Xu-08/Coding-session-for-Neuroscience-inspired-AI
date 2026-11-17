import copy

import nbformat

from notebook_preview import WIDGET_STATE, WIDGET_VIEW, static_preview


def test_static_preview_preserves_captured_results_and_live_source():
    text=nbformat.v4.new_output('stream',name='stdout',text='Shared reward: -0.2\n')
    # A real one-pixel PNG is enough to check that rich image data survive unchanged.
    png=nbformat.v4.new_output('display_data',data={'image/png':
        'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jC1sAAAAASUVORK5CYII='})
    chart=nbformat.v4.new_output('display_data',data={'text/html':'<p>Comparison table</p>'})
    view=nbformat.v4.new_output('display_data',data={WIDGET_VIEW:{
        'model_id':'box','version_major':2,'version_minor':0},'text/plain':'VBox(...)'})
    notebook=nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(
        'spread_ui = spread_controls(team_reward)',outputs=[chart,view])])
    notebook.metadata.widgets={WIDGET_STATE:{'version_major':2,'version_minor':0,'state':{
        'box':{'model_name':'VBoxModel','state':{'children':['IPY_MODEL_buttons','IPY_MODEL_output']}},
        'buttons':{'model_name':'HBoxModel','state':{'children':['IPY_MODEL_box']}},  # tolerate cycles
        'output':{'model_name':'OutputModel','state':{'outputs':[text,png]}},
    }}}
    original=copy.deepcopy(notebook)
    preview=static_preview(notebook)
    assert notebook==original  # Verification still has the actual live widget output.
    assert preview.cells[0].source==notebook.cells[0].source
    assert preview.cells[0].outputs[:3]==[chart,text,png]
    assert 'JupyterLab' in preview.cells[0].outputs[-1].data['text/plain']
    assert 'widgets' not in preview.metadata and WIDGET_VIEW not in nbformat.writes(preview)
    assert static_preview(preview)==preview
    nbformat.validate(preview)


def test_missing_widget_state_becomes_a_readable_rerun_hint():
    notebook=nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell('show_controls()',outputs=[
        nbformat.v4.new_output('display_data',data={WIDGET_VIEW:{
            'model_id':'missing','version_major':2,'version_minor':0}})])])
    preview=static_preview(notebook)
    assert len(preview.cells[0].outputs)==1
    assert set(preview.cells[0].outputs[0].data)=={'text/html','text/plain'}
    assert 'run this cell' in preview.cells[0].outputs[0].data['text/plain']
