"""Integer comparison observations at the pinned commit. The map stores no mutant answers."""
import marshal
import operator
import re
import zlib

OPS = {"<": operator.lt, "<=": operator.le, ">": operator.gt, ">=": operator.ge,
       "==": operator.eq, "!=": operator.ne, "is": operator.is_, "is not": operator.is_not}


def load(path):
    try:
        with open(path, "rb") as stream:
            return marshal.loads(zlib.decompress(stream.read()))
    except (OSError, EOFError, ValueError, zlib.error):
        return {}


def unchanged(sites, old, new):
    """Return only tests whose every recorded integer comparison has the same outcome."""
    for op_start, op_end, lit_start, lit_end, literal_first, groups in sites:
        op = old[op_start:op_end]
        if lit_start is None:
            for replacement in OPS:
                if replacement.startswith("is") and not (old[op_start-1].isspace() and old[op_end].isspace()):
                    continue
                if old[:op_start] + replacement + old[op_end:] == new:
                    # Relational observations are bitmasks of outcomes. An unsupported
                    # operation (e.g. ordering None) has no entry and stays selected.
                    mask = 0
                    for compatible, tests in groups.items():
                        if all(facts & (1 << tuple(OPS).index(replacement)) for facts in compatible):
                            mask |= tests
                    return mask
            continue
        literal = old[lit_start:lit_end]
        if op not in OPS or not literal.isascii() or not literal.isdecimal():
            continue
        for start, end in ((op_start, op_end), (lit_start, lit_end)):
            if not new.startswith(old[:start]) or not new.endswith(old[end:]):
                continue
            token = new[start:len(new) - len(old[end:])]
            new_op, new_literal = op, int(literal)
            if start == op_start and token in OPS and token not in ("is", "is not"):
                new_op = token
            elif start == lit_start and re.fullmatch(r"-?[0-9]{1,18}", token):
                new_literal = int(token)
            else:
                continue
            mask = 0
            for values, tests in groups.items():
                if literal_first:
                    same = all(OPS[op](int(literal), v) == OPS[new_op](new_literal, v) for v in values)
                else:
                    same = all(OPS[op](v, int(literal)) == OPS[new_op](v, new_literal) for v in values)
                if same:
                    mask |= tests
            return mask
    return 0
