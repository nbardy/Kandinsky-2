from lark import Lark, Transformer, v_args

# Load the grammar
with open("super_prompt.lark", "r") as f:
    grammar = f.read()

# Create a Lark parser
dsl_parser = Lark(grammar, start="start", parser="lalr")

# Define a transformer class to convert the AST into a more convenient format


class SuperPromptTransformer(Transformer):
    @v_args(inline=True)
    def sexpr(self, items):
        return [items]

    def SNAME(self, token):
        return str(token)

    def STRING(self, token):
        return eval(token)

# Parse the DSL code


def parse_super_prompt(dsl_code):
    tree = dsl_parser.parse(dsl_code)
    transformer = SuperPromptTransformer()
    return transformer.transform(tree)


with open("film.super_prompt", "r") as f:
    dsl_code = f.read()


def eval_str(ast, env):
    result = []
    for item in ast:
        if isinstance(item, str):
            result.append(item)
        else:
            result.append(eval_ast(item, env))
    return "".join(result)


def eval_let(ast, env):
    new_env = env.copy()
    bindings = ast[:-1]

    for i in range(0, len(bindings), 2):
        key = bindings[i]
        value = eval_ast(bindings[i + 1], new_env)
        new_env[key] = value

    return eval_ast(ast[-1], new_env)


def eval_sname(name, env):
    return env.get(name, name)


def eval_ast(ast, env):
    if isinstance(ast, list):
        head, *tail = ast
        if head == "cond":
            return eval_cond(tail, env)
        elif head == "case":
            return eval_case(tail, env)
        elif head == "str":
            return eval_str(tail, env)
        elif head == "let":
            return eval_let(tail, env)
        else:
            raise ValueError(f"Unknown function: {head}")
    elif isinstance(ast, str):
        return eval_sname(ast, env)
    else:
        return ast


ast = parse_super_prompt(dsl_code)
print(ast)
