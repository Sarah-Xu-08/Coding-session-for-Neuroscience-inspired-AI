import ast
import json
from pathlib import Path
import nbformat
from build_student import student_notebook

ROOT=Path(__file__).resolve().parents[1]


def test_generated_student_has_no_solution_cells_or_outputs():
    instructor=nbformat.read(ROOT/'notebooks/chapter3_instructor.ipynb',as_version=4)
    student=nbformat.read(ROOT/'notebooks/chapter3_student.ipynb',as_version=4)
    expected=student_notebook(instructor)
    assert [(c.cell_type,c.source) for c in student.cells]==[(c.cell_type,c.source) for c in expected.cells]
    nbformat.validate(student)
    for cell in student.cells:
        assert 'student_source' not in cell.metadata
        assert 'instructor-only' not in cell.metadata.get('tags',[])
        if cell.cell_type=='code':
            assert cell.outputs==[] and cell.execution_count is None
            ast.parse(cell.source)
    text=json.dumps(student)
    assert 'USE_REFERENCE_SOLUTIONS' not in text
    assert 'reference_epsilon_greedy' not in text
    exercise_cells=[c for c in student.cells if any(t.startswith('exercise-') for t in c.metadata.get('tags',[]))]
    assert len(exercise_cells)==4  # Exercise 2 is a decision/interpretation activity.
    assert all('raise NotImplementedError' in c.source for c in exercise_cells)


def test_notebook_images_exist():
    assert (ROOT/'assets/figures/four_tasks_overview.png').stat().st_size>10000
    assert (ROOT/'assets/figures/four_tasks_overview.svg').is_file()
