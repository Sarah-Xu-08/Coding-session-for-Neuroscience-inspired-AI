"""Generate a clean student workbook from the maintained instructor notebook."""
import copy
from pathlib import Path
import nbformat

ROOT = Path(__file__).resolve().parents[1]


def student_notebook(source):
    result = copy.deepcopy(source)
    result.cells = []
    for original in source.cells:
        tags = original.metadata.get('tags', [])
        if 'instructor-only' in tags:
            continue
        cell = copy.deepcopy(original)
        alternate = cell.metadata.pop('student_source', None)
        if alternate is not None:
            cell.source = alternate
            # Student TODOs must be visible regardless of instructor display settings.
            if cell.cell_type == 'code':
                cell.metadata.get('jupyter', {}).pop('source_hidden', None)
        if cell.cell_type == 'code':
            cell.outputs = []
            cell.execution_count = None
        cell.metadata.pop('execution', None)
        result.cells.append(cell)
    result.metadata.pop('widgets', None)
    result.metadata['lab_role'] = 'student'
    return result


def build():
    path = ROOT / 'notebooks/chapter3_instructor.ipynb'
    notebook = student_notebook(nbformat.read(path, as_version=4))
    destination = path.with_name('chapter3_student.ipynb')
    nbformat.validate(notebook)
    nbformat.write(notebook, destination)
    print('Generated', destination.relative_to(ROOT))


if __name__ == '__main__':
    build()
