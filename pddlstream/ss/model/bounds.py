from ss.utils import Hashable, INF


class OutputSet(Hashable):
    pass


class InputOutputSet(OutputSet):

    def __init__(self, stream, inputs, index):
        self.stream = stream
        self.inputs = tuple(inputs)
        self.index = index
        assert len(inputs) == len(stream.inputs)
        super(InputOutputSet, self).__init__(stream, inputs, index)

    def get_instance(self):
        return self.stream.get_instance(self.inputs)

    def get_dependents(self):
        dependents = []
        for inp in self.inputs:
            if isinstance(inp, OutputSet):
                dependents += inp.get_dependents()
            dependents.append(inp)
        return dependents

    def get_depth(self):
        depth = 1
        for inp in self.inputs:
            if isinstance(inp, OutputSet):
                depth = max(depth, inp.get_depth() + 1)
        return depth

    def __getitem__(self, param):
        mapping = dict(zip(self.stream.inputs, self.inputs))
        return mapping[param]

    def __repr__(self):
        param = self.stream.outputs[self.index]

        return '%s-%s' % (param, id(self) % 100)


class SharedOutputSet(OutputSet):

    def __init__(self, stream, index):
        self.stream = stream
        self.index = index
        super(SharedOutputSet, self).__init__(stream, index)

    def get_dependents(self):
        return []

    def get_depth(self):
        return 1

    def __repr__(self):
        param = self.stream.outputs[self.index]
        return '%s-%s' % (param, self.stream.name)


class PartialOutputSet(OutputSet):

    def __init__(self, stream, parameters, inputs, index):
        self.stream = stream
        self.parameters = tuple(parameters)
        self.inputs = tuple(inputs)
        self.index = index
        super(PartialOutputSet, self).__init__(
            stream, self.parameters, self.inputs, index)

    def get_dependents(self):
        dependents = []
        for inp in self.inputs:
            if isinstance(inp, OutputSet):
                dependents += inp.get_dependents()
            dependents.append(inp)
        return dependents

    def get_depth(self):
        depth = 1
        for inp in self.inputs:
            if isinstance(inp, OutputSet):
                depth = max(depth, inp.get_depth() + 1)
        return depth

    def __getitem__(self, param):
        mapping = dict(zip(self.parameters, self.inputs))
        return mapping[param]

    def __repr__(self):
        param = self.stream.outputs[self.index]
        return '%s[%s]' % (param[1:], id(self) % 100)


class UniqueOutputSet(object):

    def __repr__(self):
        return 'o[{}]'.format(id(self) % 100)

identical_output = UniqueOutputSet()


def unique_bound_fn(stream):
    return lambda *args: [tuple(InputOutputSet(stream, args, i)
                                for i in xrange(len(stream.outputs)))]


def shared_bound_fn(stream):
    return lambda *args: [tuple(SharedOutputSet(stream, i)
                                for i in xrange(len(stream.outputs)))]


def depth_bound_fn(stream, max_depth=1):
    shared_fn = shared_bound_fn(stream)
    unique_fn = unique_bound_fn(stream)

    def fn(*args):
        depth = 1
        for inp in args:
            if isinstance(inp, OutputSet):
                depth = max(depth, inp.get_depth() + 1)
        if max_depth < depth:
            return shared_fn(*args)
        return unique_fn(*args)
    return fn


def cyclic_bound_fn(stream):
    unique_fn = unique_bound_fn(stream)
    shared_fn = shared_bound_fn(stream)

    def fn(*args):
        for inp in args:
            if not isinstance(inp, OutputSet):
                continue
            if any(stream == x.stream for x in inp.get_dependents() if isinstance(x, OutputSet)):
                return shared_fn(*args)
        return unique_fn(*args)
    return fn


def no_bound_fn(stream):
    return lambda *args: []


class PartialBoundFn(object):

    def __init__(self, parameters):
        self.parameters = tuple(parameters)

    def __call__(self, stream):
        def fn(*args):
            assert len(stream.inputs) == len(args)
            mapping = dict(zip(stream.inputs, args))
            params = [p for p in stream.inputs if p in self.parameters]
            inputs = [mapping[p] for p in params]
            return [tuple(PartialOutputSet(stream, params, inputs, i)
                          for i in xrange(len(stream.outputs)))]
        return fn

    def __repr__(self):
        return '{}{}'.format(self.__class__.__name__, self.parameters)


class Bound(object):
    pass


class GenericBound(Bound):

    def __init__(self, relation, inputs, parameter):
        self.relation = relation
        self.inputs = inputs
        self.parameter = parameter


class Interval(Bound):

    def __init__(self, minimum, maximum):
        assert minimum <= maximum
        self.minimum = minimum
        self.maximum = maximum


class Finite(Bound):

    def __init__(self, values):
        self.values = values


class Singleton(Bound):

    def __init__(self, value):
        self.value = value

zero_to_inf = Interval(0, INF)
neg_inf_to_inf = Interval(-INF, INF)
