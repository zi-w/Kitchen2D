from bounds import SharedOutputSet, INF
from functions import process_parameters, process_domain, Object
from ss.model.bounds import unique_bound_fn, shared_bound_fn, no_bound_fn, cyclic_bound_fn, depth_bound_fn, PartialBoundFn
from ss.utils import Hashable


class Context(object):

    def __init__(self, output_sets, conditions=tuple()):
        self.output_sets = output_sets
        self.conditions = conditions

    def __repr__(self):
        return '{}({},conditions={})'.format(self.__class__.__name__, self.output_sets, self.conditions)


class CondGen(object):

    def __init__(self, *inputs):
        self.inputs = tuple(inputs)
        self.calls = 0
        self.enumerated = False

    def generate(self, context=None):

        raise NotImplementedError()


class StreamInstance(Hashable):

    def __init__(self, stream, inputs):
        assert len(stream.inputs) == len(inputs)
        super(StreamInstance, self).__init__(stream, inputs)
        self.stream = stream
        self.inputs = inputs
        self.enumerated = False
        self.generator = None
        self.disabled = False
        self.calls = 0
        self.use_unique = (stream.bound_type == 'unique')

    def domain_mapping(self):
        return dict(zip(self.stream.inputs, self.inputs))

    def graph_mapping(self, outputs):
        assert len(self.stream.outputs) == len(outputs)
        return dict(zip(self.stream.inputs, self.inputs) + zip(self.stream.outputs, outputs))

    def domain(self):
        mapping = self.domain_mapping()
        return [atom.substitute(mapping) for atom in self.stream.domain]

    def substitute_graph(self, outputs):
        mapping = self.graph_mapping(outputs)
        return [atom.substitute(mapping) for atom in self.stream.graph]

    def next_outputs(self, context=None):
        assert not self.enumerated
        if self.generator is None:
            self.generator = self.stream.fn(*self.inputs)
        self.calls += 1
        if self.stream.max_calls <= self.calls:
            self.enumerated = True
        if isinstance(self.generator, CondGen):
            outputs = self.generator.generate(context=context)
            self.enumerated = self.generator.enumerated
            return outputs
        else:
            try:
                return next(self.generator)
            except StopIteration:
                self.enumerated = True
                return []

    def next_atoms(self, context=None):
        if isinstance(self.stream, WildStream):
            return self.next_outputs(context=context)
        return [a for atoms in map(self.substitute_graph, self.next_outputs(context=context)) for a in atoms]

    def get_effort(self):
        if self.enumerated or self.disabled:
            return INF
        return self.stream.effort_fn(*self.inputs)

    def bound_outputs(self):
        if self.enumerated or self.disabled:
            return []
        return self.stream.bound_fn(*self.inputs)

    def bound_atoms(self):
        return [a for atoms in map(self.substitute_graph, self.bound_outputs()) for a in atoms]

    def bound_repr(self):
        return '{}{}->{}'.format(self.stream.name, self.inputs, self.bound_outputs())

    def __repr__(self):
        return '{}{}->{}'.format(self.stream.name, self.inputs, self.stream.outputs)


class Stream(object):
    """ Function to generator """

    def __init__(self, inp, domain, fn, out, graph, bound='cyclic', effort=1, max_calls=INF, eager=False, name=""):
        self.inputs = process_parameters(inp)
        self.domain = process_domain(
            list(domain) + [Object(p) for p in self.inputs])
        self.fn = fn
        self.outputs = process_parameters(out)
        self.graph = tuple(graph) + tuple(Object(p) for p in self.outputs)
        self.max_calls = max_calls
        self.eager = eager
        self.name = name
        if any(atom.head.has_constants() for atom in self.domain):
            raise NotImplementedError(
                'Unable to have constants in domain currently')

        self.bound_type = bound
        if bound == 'unique':
            self.bound = unique_bound_fn(self)
        elif bound == 'shared':
            self.bound = shared_bound_fn(self)
        elif bound == 'cyclic':
            self.bound = cyclic_bound_fn(self)
        elif bound == 'depth':
            self.bound = depth_bound_fn(self, max_depth=1)
        elif bound is None:
            self.bound = no_bound_fn(self)
        elif isinstance(bound, PartialBoundFn):
            self.bound = bound(self)
        else:

            self.bound = bound
        self.effort = effort

        self.instances = {}

    def bound_fn(self, *args):
        assert callable(self.bound)
        return self.bound(*args)

    def effort_fn(self, *args):

        if callable(self.effort):
            return self.effort(*args)
        return self.effort

    def get_instance(self, inputs):
        inputs = tuple(inputs)
        if inputs not in self.instances:
            self.instances[inputs] = StreamInstance(self, inputs)
        return self.instances[inputs]

    def __repr__(self):
        return '{}{}->{}'.format(self.name, self.inputs, self.outputs)


class WildStream(Stream):
    """ Is just handled differently"""

    pass


class GenStream(Stream):
    """ Generator of values """

    def __init__(self, inp, domain, fn, out, graph, **kwargs):
        def gen(*inputs):
            for outputs in fn(*inputs):
                if outputs is None:
                    yield []
                else:
                    yield [outputs]
        super(GenStream, self).__init__(inp, domain, gen,
                                        out, graph, **kwargs)


class ListStream(Stream):
    """ Function to list """

    def __init__(self, inp, domain, fn, out, graph, **kwargs):
        super(ListStream, self).__init__(inp, domain,
                                         lambda *args: iter([fn(*args)]),
                                         out, graph, max_calls=1, **kwargs)


class FnStream(ListStream):
    """ Function """

    def __init__(self, inp, domain, fn, out, graph, **kwargs):
        def list_fn(*args):
            outputs = fn(*args)
            if outputs is None:
                return []
            return [outputs]
        super(FnStream, self).__init__(
            inp, domain, list_fn, out, graph, **kwargs)


class TestStream(FnStream):
    """ Function """

    def __init__(self, inp, domain, test, graph, **kwargs):
        super(TestStream, self).__init__(inp, domain,
                                         lambda *args: tuple() if test(*args) else None,
                                         [], graph, **kwargs)
