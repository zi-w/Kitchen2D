from ss.to_pddl import pddl_head, pddl_parameter
from ss.utils import Hashable, INT_INF

OBJECT_NAME = 'Object'
COST_NAME = 'total-cost'
TIME_NAME = 'total-time'


def is_parameter(name):
    return (type(name) == str) and (name[0] == '?') and (len(name) != 1)


def check_parameters(names):
    for name in names:
        if not is_parameter(name):
            raise ValueError('{} is not a parameter'.format(name))


def process_parameters(parameters):
    if type(parameters) == str:
        parameters = parameters.split()
    check_parameters(parameters)
    return tuple(parameters)


def implied_atoms(evaluations):
    implied = set()
    for e in evaluations:
        implied.update(e.head.implied())
    return implied


def infer_evaluations(evals):
    return set(evals) | implied_atoms(evals)


def process_domain(domain):
    return tuple(set(domain) - implied_atoms(domain))


class Head(Hashable):

    def __init__(self, func, args):
        self.function = func
        self.args = tuple(args)
        if len(self.function.inputs) != len(self.args):
            raise ValueError('Function {} has {} parameters but {} arguments passed'.format(
                self.function, len(self.function.inputs), len(self.args)))
        super(Head, self).__init__(func, self.args)

    def has_constants(self):
        return not all(map(is_parameter, self.args))

    def has_parameters(self):
        return any(map(is_parameter, self.args))

    def domain(self):
        for atom in self.function.domain:
            yield atom.substitute(self.mapping())

    def computed(self):
        return self.function.computed(self.args)

    def get_eval(self):
        return self.function.get_eval(self.args)

    def get_bound(self):
        return self.function.get_bound(self.args)

    def implied(self):

        implied = set()
        for atom in self.domain():
            if atom not in implied:
                implied |= {atom} | atom.head.implied()
        return implied

    def mapping(self):
        return dict(zip(self.function.inputs, self.args))

    def substitute(self, mapping):
        return self.__class__(self.function, (mapping.get(a, a) for a in self.args))

    def pddl(self):
        return pddl_head(self.function.name, self.args)

    def __repr__(self):
        return '{}({})'.format(self.function.name, ','.join(map(repr, self.args)))


class Evaluation(Hashable):

    def __init__(self, head, value):
        super(Evaluation, self).__init__(head, value)
        self.head = head
        self.value = value

    def holds(self, state):
        return state[self.head] == self.value

    def assign(self, state):
        state[self.head] = self.value

    def heads(self):
        return {self.head}

    def substitute(self, mapping):
        return self.__class__(self.head.substitute(mapping), self.value)

    def pddl(self):
        return '(= {} {})'.format(self.head.pddl(), repr(self.value))

    def __repr__(self):
        return '{}={}'.format(self.head, repr(self.value))


class Literal(Evaluation):

    def __init__(self, head):
        assert isinstance(head.function, Predicate)
        super(Literal, self).__init__(head, not self.negated)

    def substitute(self, mapping):
        return self.__class__(self.head.substitute(mapping))


class Atom(Literal):
    negated = False

    def __invert__(self):
        return NegatedAtom(self.head)

    def pddl(self):
        return self.head.pddl()

    def __repr__(self):
        return repr(self.head)


class NegatedAtom(Literal):
    negated = True

    def __invert__(self):
        return Atom(self.head)

    def pddl(self):
        return '(not {})'.format(self.head.pddl())

    def __repr__(self):
        return '~{}'.format(self.head)


class Function(object):
    _prefix = 'F'
    num = 0

    def __init__(self, inp, domain=tuple(), fn=None, bound=None, eager=True, name=None):

        self.inputs = process_parameters(inp)
        if name is not OBJECT_NAME:
            domain = list(domain) + [Object(p) for p in self.inputs]
        self.domain = process_domain(domain)
        assert {a for e in self.domain for h in e.heads()
                for a in h.args if is_parameter(a)} <= set(self.inputs)

        self.fn = fn
        self.bound = bound
        self.eager = eager
        self.n = self.__class__.num
        self.__class__.num += 1
        if name is None:
            name = '{}{}'.format(self._prefix, self.n)
        self.name = name
        self.evaluations = {}

    def bound_fn(self, *args):
        if callable(self.bound):
            return self.bound(*args)
        return self.bound

    def computed(self, args):
        return tuple(args) in self.evaluations

    def get_eval(self, args):
        assert (self.fn is not None) and not self.computed(args)
        self.evaluations[args] = self.fn(*args)
        return initialize(self.get_head(args), self.evaluations[args])

    def get_bound(self, args):
        assert (self.bound_fn is not None) and not self.computed(args)
        return initialize(self.get_head(args), self.bound_fn(*args))

    def is_defined(self):
        return self.fn is not None

    def is_bound_defined(self):
        return self.bound is not None

    def get_head(self, args):
        return Head(self, args)

    def __call__(self, *args):

        return self.get_head(args)

    def pddl(self):
        return pddl_head(self.name, map(pddl_parameter, self.inputs))

    def __repr__(self):
        return self.name


class RealFunction(Function):

    def __init__(self, *args, **kwargs):
        super(RealFunction, self).__init__(*args, bound=-INT_INF, **kwargs)


class NonNegFunction(Function):

    def __init__(self, inp, bound=0, **kwargs):
        super(NonNegFunction, self).__init__(inp, bound=bound, **kwargs)


class Predicate(Function):
    _prefix = 'P'
    num = 0

    def __init__(self, inp, bound=True, **kwargs):
        super(Predicate, self).__init__(inp, bound=bound, **kwargs)

    def __call__(self, *args):

        return Atom(super(Predicate, self).__call__(*args))

Object = Predicate('?x', name=OBJECT_NAME)
TotalCost = NonNegFunction('', name=COST_NAME)
TotalTime = NonNegFunction('', name=TIME_NAME)


def initialize(head, value):
    if isinstance(head.function, Predicate):
        assert value in (False, True)
        if value is True:
            return Atom(head)
        elif value is False:
            return NegatedAtom(head)
        else:
            raise ValueError(value)
    else:
        return Evaluation(head, value)


class Operation(object):

    def __init__(self, head, value):
        self.head = head
        self.value = value

    def evaluate(self, state):
        return state[self.value] if isinstance(self.value, Head) else self.value

    def heads(self):

        return {self.head, self.value} if isinstance(self.value, Head) else {self.head}

    def substitute(self, mapping):
        if isinstance(self.value, Head):
            return self.__class__(self.head.substitute(mapping), self.value.substitute(mapping))
        return self.__class__(self.head.substitute(mapping), self.value)

    def pddl(self):
        return '({} {} {})'.format(self._pddl_name, self.head.pddl(),
                                   self.value.pddl() if isinstance(self.value, Head) else repr(self.value))

    def __repr__(self):
        return '{}{}{}'.format(self.head, self._operation, self.value)


class Increase(Operation):
    _pddl_name = 'increase'
    _operation = '+='

    def assign(self, state):
        state[self.head] += self.evaluate(state)


class Decrease(Operation):
    _pddl_name = 'decrease'
    _operation = '-='

    def assign(self, state):
        state[self.head] -= self.evaluate(state)


class Multiply(Operation):
    _pddl_name = 'multiply'
    _operation = '*='

    def assign(self, state):
        state[self.head] *= self.evaluate(state)


class Divide(Operation):
    _pddl_name = 'divide'
    _operation = '/='

    def assign(self, state):
        state[self.head] /= self.evaluate(state)


def rename_functions(assignments):
    for name, value in assignments.iteritems():
        if isinstance(value, Function) and (value not in [Object, TotalCost, TotalTime]):
            value.name = name
