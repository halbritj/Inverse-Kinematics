


def g_parser(line):
    command, *data = line.split(' ')
    pairs = []
    for pair in data:
        pairs.append( (pair[0], pair[1:]) )
    return command, pairs

#code = 'G0 A5 B6 C50.2 D21 E41 F1'


r = g_parser(code)
print(r)
