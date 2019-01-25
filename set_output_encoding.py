# -*- coding: utf-8 -*-

# see https://stackoverflow.com/a/19700891/2050986

def set_output_encoding(encoding='utf-8', force=False):
    import sys
    import codecs
    '''When piping to the terminal, python knows the encoding needed, and
       sets it automatically. But when piping to another program (for example,
       | less), python can not check the output encoding. In that case, it 
       is None. What I am doing here is to catch this situation for both 
       stdout and stderr and force the encoding'''
    current = sys.stdout.encoding
    if current is None or force:
        sys.stdout = codecs.getwriter(encoding)(sys.stdout)
    current = sys.stderr.encoding
    if current is None or force:
        sys.stderr = codecs.getwriter(encoding)(sys.stderr)
