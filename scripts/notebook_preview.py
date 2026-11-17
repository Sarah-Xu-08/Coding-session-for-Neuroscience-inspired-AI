"""Make saved notebook outputs readable without a widget-capable frontend."""
import copy

import nbformat

WIDGET_VIEW = 'application/vnd.jupyter.widget-view+json'
WIDGET_STATE = 'application/vnd.jupyter.widget-state+json'


def static_preview(notebook):
    """Keep captured widget results and live cell code, but save no widget views.

    Saved controls cannot call Python until their cells run in a live kernel.
    Plain notebook viewers can show the captured text/images and the rerun hint.
    """
    result = copy.deepcopy(notebook)
    models = result.metadata.get('widgets', {}).get(WIDGET_STATE, {}).get('state', {})

    def snapshots(model_id, seen):
        if model_id in seen:
            return []
        seen.add(model_id)
        state = models.get(model_id, {}).get('state', {})
        outputs = []
        for output in state.get('outputs', []):
            view = output.get('data', {}).get(WIDGET_VIEW)
            if view:
                outputs.extend(snapshots(view['model_id'], seen))
            else:
                outputs.append(nbformat.from_dict(copy.deepcopy(output)))
        for child in state.get('children', []):
            if isinstance(child, str) and child.startswith('IPY_MODEL_'):
                outputs.extend(snapshots(child[len('IPY_MODEL_'):], seen))
        return outputs

    hint = 'Interactive controls: run this cell in JupyterLab to use the buttons, selectors or slider.'
    for cell in result.cells:
        if cell.cell_type != 'code':
            continue
        outputs = []
        for output in cell.outputs:
            view = output.get('data', {}).get(WIDGET_VIEW)
            if view:
                outputs.extend(snapshots(view['model_id'], set()))
                outputs.append(nbformat.v4.new_output('display_data', data={
                    'text/plain': hint,
                    'text/html': f'<p><em>{hint}</em></p>',
                }))
            else:
                outputs.append(output)
        cell.outputs = outputs
    result.metadata.pop('widgets', None)
    return result
