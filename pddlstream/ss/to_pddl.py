DEFAULT_TYPE = 'object'


def pddl_parameter(param):
    return '{} - {}'.format(param, DEFAULT_TYPE)


def pddl_functions(predicates):
    return '\n\t\t'.join(sorted(p.pddl() for p in predicates))


def pddl_connective(literals, connective):
    if not literals:
        return '()'
    if len(literals) == 1:
        return literals[0].pddl()
    return '({} {})'.format(connective, ' '.join(l.pddl() for l in literals))


def pddl_conjunction(literals):
    return pddl_connective(literals, 'and')


def pddl_disjunction(literals):
    return pddl_connective(literals, 'or')


def pddl_at_start(literal):
    return '(at start {})'.format(literal.pddl())


def pddl_over_all(literal):
    return '(over all {})'.format(literal.pddl())


def pddl_at_end(literal):
    return '(at end {})'.format(literal.pddl())


def pddl_head(name, args):
    if not args:
        return '({})'.format(name)
    return '({} {})'.format(name, ' '.join(args))


def pddl_actions(actions):
    for action in actions:
        yield action.pddl()


def pddl_axioms(axioms):
    axioms_from_derived = {}
    for axiom in axioms:

        derived = axiom.effect.head
        if derived not in axioms_from_derived:
            axioms_from_derived[derived] = []
        axioms_from_derived[derived].append(axiom)

    for derived in axioms_from_derived:
        clauses = []
        for axiom in axioms_from_derived[derived]:
            quantified = set(axiom.parameters) - set(derived.args)
            condition = pddl_conjunction(axiom.preconditions)
            if quantified:
                param_s = ' '.join(map(pddl_parameter, quantified))
                clauses.append('(exists ({}) {})'.format(param_s, condition))
            else:
                clauses.append(condition)

        yield '\t(:derived {}\n'               '\t\t(or {}))'.format(pddl_head(derived.function.name,
                                                                               map(pddl_parameter, derived.args)),
                                                                     '\n\t\t\t'.join(clauses))


def pddl_domain(domain, constants, predicates, functions, actions, axioms):

    return '(define (domain {})\n'           '\t(:requirements :typing)\n'           '\t(:types {})\n'           '\t(:constants {})\n'           '\t(:predicates {})\n'           '\t(:functions {})\n'           '{})\n'.format(domain, DEFAULT_TYPE,
                                                                                                                                                                                                                                 pddl_parameter(
                                                                                                                                                                                                                                     ' '.join(sorted(constants))),
                                                                                                                                                                                                                                 pddl_functions(
                                                                                                                                                                                                                                     predicates),
                                                                                                                                                                                                                                 pddl_functions(
                                                                                                                                                                                                                                     functions),
                                                                                                                                                                                                                                 '\n'.join(list(pddl_actions(actions)) +
                                                                                                                                                                                                                                           list(pddl_axioms(axioms))))


def pddl_problem(problem, domain, objects, initial_atoms, goal_literals, objective=None):

    s = '(define (problem {})\n'           '\t(:domain {})\n'           '\t(:init {})\n'           '\t(:goal {})'.format(problem, domain,

                                                                                                                         pddl_functions(
                                                                                                                             initial_atoms),
                                                                                                                         pddl_conjunction(goal_literals))
    if objective is not None:
        s += '\n\t(:metric minimize {})'.format(objective.pddl())
    return s + ')\n'
