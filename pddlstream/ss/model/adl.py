from ss.model.functions import process_parameters
from ss.to_pddl import pddl_parameter


class Not(object):
    _pddl = 'not'

    def __init__(self, formula):
        self.formula = formula

    def heads(self):
        return self.formula.heads()

    def substitute(self, mapping):
        return self.__class__(self.formula.substitute(mapping))

    def pddl(self):
        return '(%s %s)' % (self._pddl, self.formula.pddl())


class Connective(object):

    def __init__(self, *formulas):
        self.formulas = tuple(formulas)

    def heads(self):
        return {h for f in self.formulas for h in f.heads()}

    def substitute(self, mapping):
        formulas = []
        for f in self.formulas:
            formulas.append(f.substitute(mapping))
        return self.__class__(*formulas)

    def pddl(self):
        return '(%s %s)' % (self._pddl, ' '.join(f.pddl() for f in self.formulas))


class And(Connective):
    _pddl = 'and'


class Or(Connective):
    _pddl = 'or'


class When(Connective):
    _pddl = 'when'


class Imply(Connective):
    _pddl = 'imply'


class Quantifier(object):

    def __init__(self, parameters, formula):
        self.parameters = process_parameters(parameters)
        self.formula = formula

    def substitute(self, mapping):

        return self.__class__(self.parameters, self.formula.substitute(mapping))

    def heads(self):
        return self.formula.heads()

    def pddl(self):
        param_s = ' '.join(map(pddl_parameter, self.parameters))
        return '(%s (%s) %s)' % (self._pddl, param_s, self.formula.pddl())


class Exists(Quantifier):
    _pddl = 'exists'
    _Connective = Or


class ForAll(Quantifier):
    _pddl = 'forall'
    _Connective = And


class Temporal(object):

    def __init__(self, formula):

        self.formula = formula

    def heads(self):
        return self.formula.heads()

    def substitute(self, mapping):
        return self.__class__(self.formula.substitute(mapping))

    def pddl(self):
        return '(%s %s)' % (self._pddl, self.formula.pddl())


class AtStart(Temporal):
    _pddl = 'at start'


class OverAll(Temporal):
    _pddl = 'over all'


class AtEnd(Temporal):
    _pddl = 'at end'
