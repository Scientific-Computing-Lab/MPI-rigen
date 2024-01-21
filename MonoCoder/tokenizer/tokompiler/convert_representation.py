import re
import random
from .parse_tools import parse


RE_NUMBERS = re.compile(r"(?<![_a-zA-Z])\b[0-9]+(?:\.[0-9]+)?(?:f)?\b(?![0-9.-])")
RE_HEXA = re.compile(r"0x[0-9a-fA-F]+")
RE_CHARS = re.compile(r"\'.\'")
RE_STR = re.compile(r"\"(?:\\.|[^\\\"])*\"")
RE_STR_MULTI_LINE = re.compile(r"\"(?:\\.|[^\"\\])*?\"")


VAR, ARR, FUNC, STRUCT, FIELD, TYPE, NUM, CHAR, STR, ARG = 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
replaced_prefixes = { VAR: 'var_',
                      ARR: 'arr_',
                      FUNC: 'func_',
                      STRUCT: 'struct_',
                      FIELD: 'field_',
                      TYPE: 'type_',
                      NUM: 'num_',
                      CHAR: 'char_',
                      STR: 'str_',
                      ARG: 'arg_'             
                    }


def prettify_ast(ast):
    tokens = ast.split()
    updated_ast = ''
    ident = 0

    for token in tokens:
        if token.endswith('('):
            updated_ast += '\t'*ident + token + '\n'
            ident += 1
        elif token.startswith(')'):
            ident -= 1
            updated_ast += '\t'*ident + token + '\n'
        else:
            updated_ast += '\t'*ident + token + '\n'
            
    return updated_ast



def count_newlines(code):
    counter = 0

    for letter in code:
        if letter == '\n':
            counter += 1
            continue

        return counter
    
    return counter


def replace_vars(code, var_mapping):
    '''
        Create replaced representation 
    '''
    updated_code = ''
    prev_idx = 0
    offset = count_newlines(code)
    updated_mappings = []
    var_offset = 0

    for old_var, new_var, start, end in var_mapping:
        updated_mappings.append((new_var, old_var, start-offset+var_offset, start-offset+var_offset+len(new_var)))
        var_offset += len(new_var)-len(old_var)
        updated_code += code[prev_idx:start-offset] + new_var
        prev_idx = end - offset

    updated_code += code[prev_idx:]

    return updated_code, updated_mappings


def get_identifiers(node, kind=''):
    '''
        Find identifiers names in code

        Parameters:
            node - declaration node in the AST
            kind - the type of  the sub node
        Return:
            list for each replaced variable kind (variable, array, function)
    '''
    # print(node.type, node.text)
    if node.type == 'identifier':
        # print('--', kind, node.text)
        return  ([(node.text, node.start_byte, node.end_byte)],[],[],[],[],[],[],[],[]) if kind=='args' else ([],[],[],[(node.text, node.start_byte, node.end_byte)],[],[],[],[],[]) if kind=='func' else ([],[],[(node.text, node.start_byte, node.end_byte)],[],[],[],[],[],[]) if kind=='arr' else ([],[(node.text, node.start_byte, node.end_byte)],[],[],[],[],[],[],[])
    elif node.type == 'name' and kind == 'func':
        return ([],[],[],[(node.text, node.start_byte, node.end_byte)],[],[],[],[],[])
    elif node.type == 'field_identifier':
        return ([],[],[],[],[(node.text, node.start_byte, node.end_byte)],[],[],[],[])
    elif node.type == 'type_identifier':
        return ([],[],[],[],[],[(node.text, node.start_byte, node.end_byte)],[],[],[])
    elif node.type == 'number_literal':
        return ([],[],[],[],[],[],[(node.text, node.start_byte, node.end_byte)],[],[])
    elif node.type == 'char_literal':
        return ([],[],[],[],[],[],[],[(node.text, node.start_byte, node.end_byte)],[])
    elif node.type == 'string_literal':
        return ([],[],[],[],[],[],[],[],[(node.text, node.start_byte, node.end_byte)])

    args, vars, arrays, funcs, fields, types, numbers, chars, strings = [], [], [], [], [], [], [], [], []
    for child in node.children:
        arg, va, ar, fu, fi, ty, nu, ch, st = get_identifiers(child, kind=('arr' if child.type == 'array_declarator' else
                                                  'args' if child.type in ['parameters', 'parameter_list', 'parameter_declaration'] else
                                                  'func' if child.type in ['call_expression', 'function_declarator', 'function_statement', 'subroutine_statement'] else
                                                  '' if child.type in ['argument_list', 'field_expression', 'compound_statement'] else
                                                  'field' if child.type == 'field_identifier' else
                                                   kind if len(kind)>0 else  ''))

        args, vars, arrays, funcs, fields, types, numbers, chars, strings = args+arg, vars+va, arrays+ar, funcs+fu, fields+fi, types+ty, numbers+nu, chars+ch, strings+st

    return args, vars, arrays, funcs, fields, types, numbers, chars, strings


def generate_serial_numbers(N):
    numbers = list(range(N))
    random.shuffle(numbers)

    return numbers


def generate_random_numbers(N):
    max_num = 1000

    if N > max_num:
        raise ValueError(f'N cannot be larger than {max_num}.')

    numbers = random.sample(range(max_num+1), N)

    return numbers


def replace_constants(code, replace_token, regex):
    '''
        Replace constatns in code with a given token

        Parameters:
            code - the original code to be updated
            replace_token - the token that will replace the constants
            regex - the regular expression that captures the constants
    '''
    matches = regex.finditer(code)

    offset = 0
    for match in matches:
        start = match.start() + offset
        end = match.end() + offset
        code = code[:start] + replace_token + code[end:]
        offset += len(replace_token) - len(match.group())

    return code


def update_var_names(ast, num_generator):
    name_map = {}
    args, vars, arrays, functions, fields, types, numbers, chars, strings = get_identifiers(ast)

    arg_names = [arg[0] for arg in args]

    new_args = [var for var in vars if var[0] in arg_names]
    vars = [var for var in vars if var[0] not in arg_names]
    args += new_args

    new_args = [function for function in functions if function[0] in arg_names]
    functions = [function for function in functions if function[0] not in arg_names]
    args += new_args

    for type, identifiers in zip([ARG, VAR, ARR, FUNC, FIELD, TYPE, NUM, CHAR, STR], [args, vars, arrays, functions, fields, types, numbers, chars, strings]):
        unique_vars= list(set([var[0] for var in identifiers]))
        random_numbers_vars = num_generator(len(unique_vars))

        for var, num in zip(unique_vars, random_numbers_vars):
            name_map[var] = f'{replaced_prefixes[type]}{num}'

    # replace and sort the vars according to their location
    vs = args+fields+vars+arrays+functions+types+numbers+chars+strings
    vs.sort(key=lambda tup: tup[1])
    var_mapping = [(var.decode(), name_map[var], start, end) for var, start, end in vs]
    
    updated_code, updated_mappings = replace_vars(ast.text.decode(), var_mapping)

    return updated_code, updated_mappings


def generate_replaced(tree, num_generator=generate_random_numbers):
    '''
        Main funtion to create the replaced represrntation
    '''
    updated_code = ''
    mappings = []

    try:
        updated_code, mappings = update_var_names(tree.root_node, num_generator)
    except ValueError as e: # N cannot be larger than 1000.
        print(e)
    except RecursionError as e:
        print(e)

    return updated_code, {v:k for (k,v,_,_) in mappings}
