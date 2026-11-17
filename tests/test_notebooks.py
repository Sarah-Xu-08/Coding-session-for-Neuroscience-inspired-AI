import ast
import json
from pathlib import Path
import nbformat
import pytest
from build_student import student_notebook

ROOT=Path(__file__).resolve().parents[1]
INSTRUCTOR=ROOT/'notebooks/chapter3_instructor.ipynb'
requires_instructor=pytest.mark.skipif(
    not INSTRUCTOR.exists(),reason='Requires the local instructor notebook, excluded from Git.')
ROLES=['student',pytest.param('instructor',marks=requires_instructor)]


def test_generated_student_has_no_solution_cells_or_outputs():
    student=nbformat.read(ROOT/'notebooks/chapter3_student.ipynb',as_version=4)
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
    assert len(exercise_cells)==5  # Three required tasks, two optional exercises.
    assert all('raise NotImplementedError' in c.source for c in exercise_cells)
    required=[c for c in exercise_cells if 'optional' not in c.metadata.get('tags',[])]
    assert len(required)==3
    assert all('...' in c.source for c in required)
    assert all(not c.metadata.get('jupyter',{}).get('source_hidden',False) for c in exercise_cells)


@requires_instructor
def test_student_matches_instructor_source():
    instructor=nbformat.read(INSTRUCTOR,as_version=4)
    student=nbformat.read(ROOT/'notebooks/chapter3_student.ipynb',as_version=4)
    expected=student_notebook(instructor)
    assert [(c.cell_type,c.source) for c in student.cells]==[(c.cell_type,c.source) for c in expected.cells]


@requires_instructor
def test_student_and_recording_pause_scaffolds_match():
    instructor=nbformat.read(ROOT/'notebooks/chapter3_instructor.ipynb',as_version=4)
    student=nbformat.read(ROOT/'notebooks/chapter3_student.ipynb',as_version=4)
    previews=[c for c in instructor.cells if 'pause-scaffold' in c.metadata.get('tags',[])]
    required=[c for c in student.cells if any(t.startswith('exercise-') for t in c.metadata.get('tags',[]))
              and 'optional' not in c.metadata.get('tags',[])]
    assert len(previews)==len(required)==3
    for preview,exercise in zip(previews,required):
        # display(Code(<literal scaffold>, ...)) must show the student's exact blanks.
        argument=ast.parse(preview.source).body[0].value.args[0].args[0]
        assert ast.literal_eval(argument)==exercise.source.strip()
    for notebook in [instructor,student]:
        assert notebook.cells[1].id=='session-overview'
        assert notebook.cells[1].source.startswith('## Session overview')


def test_optional_student_exercises_do_not_block_the_core_flow():
    student=nbformat.read(ROOT/'notebooks/chapter3_student.ipynb',as_version=4)
    finish=next(i for i,c in enumerate(student.cells) if c.id=='recording-finish')
    namespace={'RUN_OPTIONAL_PRACTICE':False}
    for i,cell in enumerate(student.cells):
        if cell.cell_type=='code' and 'optional' in cell.metadata.get('tags',[]):
            assert i>finish
            exec(cell.source,namespace)
    assert 'practice_tit_for_tat' not in namespace
    assert 'practice_team_reward' not in namespace


@pytest.mark.parametrize('role',ROLES)
def test_notebook_images_exist(role):
    import re
    notebook=nbformat.read(ROOT/f'notebooks/chapter3_{role}.ipynb',as_version=4)
    text='\n'.join(c.source for c in notebook.cells)
    assert 'four_tasks_overview' not in text
    assert 'stag_hunt' not in text and 'Stag Hunt' not in text
    images=re.findall(r'!\[.*?\]\((\.\./assets/figures/[^)]+)\)',text)
    assert len(images)==4
    for image in images:
        assert (ROOT/'notebooks'/image).stat().st_size>10000


@pytest.mark.parametrize('role',ROLES)
def test_dqn_lesson_switches_to_sb3_before_the_experiment(role):
    notebook=nbformat.read(ROOT/f'notebooks/chapter3_{role}.ipynb',as_version=4)
    cells={c.id:c for c in notebook.cells}
    order=[c.id for c in notebook.cells]
    update=order.index('1f4813a7')
    assert order[update+1:update+3]==['sb3-introduction','sb3-example']
    assert order.index('sb3-example')<order.index('0a6d4310')
    experiment=cells['0a6d4310'].source
    assert 'run_sb3_highway_study(' in experiment and 'load_sb3_study()' in experiment
    assert 'DQNAgent' not in experiment
    playback=cells['ebd14b6c'].source
    assert 'load_sb3_study_dqn(' in playback and 'PlaybackDQN' not in playback
    assert 'shown_dqn.save(' in cells['2fe4c806'].source
