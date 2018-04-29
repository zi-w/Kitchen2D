from ss.model.functions import process_parameters, is_parameter, Atom, Head
from ss.to_pddl import pddl_parameter, pddl_conjunction, pddl_at_start, pddl_at_end, pddl_over_all


def applicable(preconditions, state):
    for p in preconditions:
        if not p.holds(state):
            return False
    return True


def apply(effects, state):
    new_state = state.copy()
    for e in effects:
        e.assign(new_state)
    return new_state


class Operator(object):

    def __init__(self, param, pre, eff):
        self.parameters = process_parameters(param)
        self.preconditions = tuple(pre)
        self.effects = tuple(eff)
        assert len(self.parameters) == len(set(self.parameters))
        assert set(self.parameters) <= self.arguments()

    def arguments(self):
        return {a for e in (self.preconditions + self.effects) for h in e.heads() for a in h.args}

    def constants(self):

        return {a for a in self.arguments() if not is_parameter(a)}

    def functions(self):
        return {h.function for e in (self.preconditions + self.effects) for h in e.heads()}

    def applicable(self, state):
        return applicable(self.preconditions, state)

    def apply(self, state):
        return apply(self.effects, state)

    def instantiate(self, values):
        param_mapping = dict(zip(self.parameters, values))
        return self.__class__(tuple(),
                              [p.substitute(param_mapping)
                               for p in self.preconditions],
                              [e.substitute(param_mapping) for e in self.effects])

    def __repr__(self):
        return 'Op({},{},{})'.format(list(self.parameters), list(self.preconditions), list(self.effects))


class Initial(Operator):

    def __init__(self, eff):
        super(Initial, self).__init__([], [], eff)

    def __repr__(self):
        return self.__class__.__name__ + repr(self.effects)


class Goal(Operator):

    def __init__(self, pre):
        super(Goal, self).__init__([], pre, [])

    def __repr__(self):
        return self.__class__.__name__ + repr(self.preconditions)


class Action(Operator):

    def __init__(self, name, param, pre, eff):
        super(Action, self).__init__(param, pre, eff)
        self.name = name.lower()

    def instantiate(self, values):
        param_mapping = dict(zip(self.parameters, values))
        return self.__class__(self.name, tuple(),
                              [p.substitute(param_mapping)
                               for p in self.preconditions],
                              [e.substitute(param_mapping) for e in self.effects])

    def substitute_constants(self, mapping):
        constant_mapping = {c: mapping[c]
                            for c in self.constants() if c in mapping}

        return self.__class__(self.name, self.parameters,
                              [p.substitute(constant_mapping)
                               for p in self.preconditions],
                              [e.substitute(constant_mapping) for e in self.effects])

    def pddl(self):
        return '\t(:action {}\n'                '\t\t:parameters ({})\n'                '\t\t:precondition {}\n'                '\t\t:effect {})'.format(self.name,
                                                                                                                                                         ' '.join(
                                                                                                                                                             map(pddl_parameter, self.parameters)),
                                                                                                                                                         pddl_conjunction(
                                                                                                                                                             self.preconditions),
                                                                                                                                                         pddl_conjunction(self.effects))

    def __repr__(self):
        return self.name


class DurativeAction(Action):

    def __init__(self, name, param, duration, start_pre=tuple(), over_pre=tuple(), end_pre=tuple(), start_eff=tuple(), end_eff=tuple()):
        self.duration = duration
        self.start_pre = tuple(start_pre)
        self.over_pre = tuple(over_pre)
        self.end_pre = tuple(end_pre)
        self.start_eff = tuple(start_eff)
        self.end_eff = tuple(end_eff)
        super(DurativeAction, self).__init__(name, param, self.start_pre + self.over_pre + self.end_pre,
                                             self.start_eff + self.end_eff)

    def functions(self):
        functions = super(DurativeAction, self).functions()
        if isinstance(self.duration, Head):
            functions.add(self.duration.function)
        return functions

    def instantiate(self, values):
        param_mapping = dict(zip(self.parameters, values))
        return self.__class__(self.name, tuple(), self.duration,
                              [p.substitute(param_mapping)
                               for p in self.start_pre],
                              [p.substitute(param_mapping)
                               for p in self.over_pre],
                              [p.substitute(param_mapping)
                               for p in self.end_pre],
                              [e.substitute(param_mapping)
                               for e in self.start_eff],
                              [e.substitute(param_mapping) for e in self.end_eff])

    def substitute_constants(self, mapping):

        constant_mapping = {c: mapping[c]
                            for c in self.constants() if c in mapping}

        return self.__class__(self.name, self.parameters, self.duration,
                              [p.substitute(constant_mapping)
                               for p in self.start_pre],
                              [p.substitute(constant_mapping)
                               for p in self.over_pre],
                              [p.substitute(constant_mapping)
                               for p in self.end_pre],
                              [e.substitute(constant_mapping)
                               for e in self.start_eff],
                              [e.substitute(constant_mapping) for e in self.end_eff])

    def pddl(self):
        s_param = ' '.join(map(pddl_parameter, self.parameters))
        s_pre = '\n\t\t\t\t\t\t'.join(map(pddl_at_start, self.start_pre) +
                                      map(pddl_over_all, self.over_pre) + map(pddl_at_end, self.end_pre))
        s_eff = '\n\t\t\t\t\t\t'.join(
            map(pddl_at_start, self.start_eff) + map(pddl_at_end, self.end_eff))
        s_dur = self.duration.pddl() if isinstance(
            self.duration, Head) else str(self.duration)
        return '\t(:durative-action {}\n'               '\t\t:parameters ({})\n'               '\t\t:duration (= ?duration {})\n'               '\t\t:condition (and {})\n'               '\t\t:effect (and {}))'.format(self.name, s_param, s_dur, s_pre, s_eff)


class DurativeAction2(Action):

    def __init__(self, name, param, duration, con, eff):
        self.duration = duration
        self.condition = con
        self.effect = eff
        super(DurativeAction2, self).__init__(name, param, [con], [eff])

    def instantiate(self, values):
        param_mapping = dict(zip(self.parameters, values))
        return self.__class__(self.name, tuple(), self.duration,
                              self.condition.substitute(param_mapping), self.effect.substitute(param_mapping))

    def substitute_constants(self, mapping):
        constant_mapping = {c: mapping[c]
                            for c in self.constants() if c in mapping}
        return self.__class__(self.name, self.parameters, self.duration,
                              self.condition.substitute(constant_mapping), self.effect.substitute(constant_mapping))

    def pddl(self):
        s_param = ' '.join(map(pddl_parameter, self.parameters))
        return '\t(:durative-action {}\n'               '\t\t:parameters ({})\n'               '\t\t:duration (= ?duration {})\n'               '\t\t:condition {}\n'               '\t\t:effect {})'.format(self.name, s_param, str(self.duration), self.condition.pddl(), self.effect.pddl())


class Axiom(Operator):

    def __init__(self, param, pre, eff):
        if not isinstance(eff, Atom):
            raise ValueError('{} eff must be an single Atom, not {}'.format(
                self.__class__.__name__, eff))
        super(Axiom, self).__init__(param, pre, [eff])
        self.effect = eff

    def instantiate(self, values):
        param_mapping = dict(zip(self.parameters, values))
        return self.__class__(tuple(),
                              [p.substitute(param_mapping)
                               for p in self.preconditions],
                              self.effect.substitute(param_mapping))

    def substitute_constants(self, mapping):
        constant_mapping = {c: mapping[c]
                            for c in self.constants() if c in mapping}
        return self.__class__(self.parameters,
                              [p.substitute(constant_mapping)
                               for p in self.preconditions],
                              self.effect.substitute(constant_mapping))

    def __repr__(self):
        return repr(self.effect)
