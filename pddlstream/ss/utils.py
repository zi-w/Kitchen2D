class Hashable(object):

    def __init__(self, *args):
        self._tuple = tuple(args)
        self._hash = hash((self.__class__,) + self._tuple)

    def __eq__(self, other):
        return (self.__class__ is other.__class__) and (self._hash == other._hash) and (self._tuple == other._tuple)

    def __ne__(self, other):
        return not (self == other)

    def __hash__(self):
        return self._hash


INF = float('inf')
INT_INF = 1e6
