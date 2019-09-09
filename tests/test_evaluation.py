from core.evaluation import accuracy


def test_accuracy_exact():
    y_true = ['equ : x + var1 = var2']
    y_pred = list(y_true)

    res = accuracy(y_true, y_pred)
    compare(res, 1.0, 1.0, 1.0)

    y_true = ['equ : x + 1 = 3 ; equ : y + 1 = 3']
    y_pred = list(y_true)

    res = accuracy(y_true, y_pred)
    compare(res, 1.0, 1.0, 1.0)

    y_true = ['equ : x + 1 = 3 ; equ : y + 1 = 3', 'equ : x + 3 = 2']
    y_pred = list(y_true)

    res = accuracy(y_true, y_pred)
    compare(res, 1.0, 1.0, 1.0)


def test_accuracy_equation_level():
    y_true = ['equ : x + 1 = 3 ; equ : y + 1 = 3']
    y_pred = ['equ : y = 5 + 1 ; equ : y + 1 = 3']

    res = accuracy(y_true, y_pred)
    compare(res, 0.0, 0.5, 0.5)

    y_true = ['equ : x + 1 = 3 ; equ : y + 1 = 3']
    y_pred = ['equ : x + 1 = 3 ; equ : y + 1 = 5']

    res = accuracy(y_true, y_pred)
    compare(res, 0.0, 0.5, 0.5)


def compare(res, exp_question_level, exp_equation_level, exp_equation_structure_level):
    assert res['question_level'] == exp_question_level
    assert res['equation_level'] == exp_equation_level
    assert res['equation_structure_level'] == exp_equation_structure_level
