from tree_sitter import Language, Parser
import os


def get_parser(lang):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    LANGUAGE = Language(os.path.join(dir_path, "parsers/tokompiler-languages.so"), lang.lower())
    parser = Parser()
    parser.set_language(LANGUAGE)
    return parser


parsers = {
    'c': get_parser('c'),
    'cpp': get_parser('cpp'),
    'fortran': get_parser('fortran')
}


def parse(code, lang):
    '''
    Convert @code into an AST according to its programming @lang
    '''
    parser = parsers[lang]
    tree = parser.parse(bytes(code, "utf8"))
    return tree

