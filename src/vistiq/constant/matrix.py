"""Bitmask constants for square matrix region selection.

Row index = ``i``, column index = ``j``. Atomic flags combine with bitwise OR;
named presets are sums of atoms.

- ``DIAGONAL`` (``1``): ``i == j``
- ``LOWER_ND`` (``2``): ``i > j``
- ``UPPER_ND`` (``4``): ``i < j``
- ``LOWER`` (``3``): ``DIAGONAL | LOWER_ND`` → ``i >= j``
- ``UPPER`` (``5``): ``DIAGONAL | UPPER_ND`` → ``i <= j``
- ``OFF_DIAGONAL`` (``6``): ``LOWER_ND | UPPER_ND`` → ``i != j``
- ``FULL`` (``7``): ``DIAGONAL | LOWER_ND | UPPER_ND`` → entire matrix
"""

DIAGONAL = 1
LOWER_ND = 2
UPPER_ND = 4
LOWER = DIAGONAL | LOWER_ND
UPPER = DIAGONAL | UPPER_ND
OFF_DIAGONAL = LOWER_ND | UPPER_ND
FULL = DIAGONAL | LOWER_ND | UPPER_ND
