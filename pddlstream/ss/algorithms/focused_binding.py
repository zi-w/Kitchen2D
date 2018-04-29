from collections import defaultdict
from itertools import product

from ss.algorithms.focused_utils import disable_stream
from ss.model.streams import Context


def get_context(bound_stream, external_conditions):

    conditions = [a for a in external_conditions if (
        set(bound_stream.bound_outputs) & set(a.head.args))]
    return Context(bound_stream.bound_outputs, conditions)


def call_streams(evaluations, disabled, bound_streams, external, eager_fail=False, verbose=True):
    success = True
    for i, bs in enumerate(bound_streams):
        instance = bs.stream
        if set(instance.domain()) <= evaluations:
            new_atoms = instance.next_atoms(context=get_context(bs, external))
            if not new_atoms:
                success = False
                if eager_fail:
                    break
            evaluations.update(new_atoms)
            disable_stream(disabled, instance)
            if verbose:
                print i + 1, instance, new_atoms
        else:
            success = False

    return success


def bind_call_streams(evaluations, disabled, bound_streams, external, verbose=True):

    bindings = {}
    for i, bs in enumerate(bound_streams):
        old_instance = bs.stream
        new_inputs = [bindings.get(inp, inp) for inp in old_instance.inputs]
        instance = old_instance.stream.get_instance(new_inputs)

        if not instance.enumerated and (set(instance.domain()) <= evaluations):

            for outputs in instance.next_outputs(context=get_context(bs, external)):
                evaluations.update(instance.substitute_graph(outputs))
                for b, o in zip(bs.bound_outputs, outputs):
                    bindings[b] = o
            disable_stream(disabled, instance)
            if verbose:
                print i + 1, instance

    return False


def multi_bind_call_streams(evaluations, disabled, bound_streams, external, single=False, verbose=True):
    bindings = defaultdict(set)
    bound_outputs = {bo for bs in bound_streams for bo in bs.bound_outputs}
    success = True
    for i, bs in enumerate(bound_streams):
        bound_domains = [bindings[bi] if bi in bound_outputs else {
            bi} for bi in bs.stream.inputs]
        stream_success = False

        for combo in product(*bound_domains):
            instance = bs.stream.stream.get_instance(combo)
            if not instance.enumerated and (set(instance.domain()) <= evaluations):
                new_outputs = instance.next_outputs(
                    context=get_context(bs, external))
                for outputs in new_outputs:
                    for b, o in zip(bs.bound_outputs, outputs):
                        bindings[b].add(o)
                    evaluations.update(instance.substitute_graph(outputs))
                if verbose:
                    print i, instance, new_outputs
                stream_success = True
                disable_stream(disabled, instance)
                if single:
                    break
        success &= stream_success
    return success
