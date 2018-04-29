# pddlstream

Lightweight implementation of stripstream with support for functions

## Examples

1. ```python -m examples.1d_table ```
2. ```python -m examples.1d_table_belief ```
3. ```python -m examples.hierarchy ```
4. ```python -m examples.tutorial ```
5. ```python -m examples.tutorial2 ```

## Algorithms

1. Exhaustive - evaluates streams exhaustively and searches once at the end
2. Incremental - alterates between search and stream evaluation
3. Focused - lazily evaluates streams, searches, and greedily identifies useful streams
4. Plan focused - treats streams as explicit actions in search
5. Dual focused - lazily evaluates streams, searches, and optimally identifies useful streams  
6. Sequence focused - dual focused + maintains queue of past stream sequences
