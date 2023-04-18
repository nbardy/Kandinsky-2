from lark import Tree, Token
import lark
import re
from lark import Lark
from lark import Lark, Transformer, v_args

import openai

# Add your OpenAI API key here

openai.api_key = "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"
openai.organization = "org-9FuYsKOwtUzcDFxwPQBCgHe1"


def send_to_llm(content):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": content},
        ],
    )

    assistant_reply = response.choices[0].message['content']
    return assistant_reply


def eval_str(item, env):
    print(item.children)
    # text = item.children[0].value
    text = "a"
    llm_response = send_to_llm(text)
    print("Querying the LLM, response: ", llm_response)
    return llm_response


def eval_ast(expr, env):
    print("eval ast")
    if expr.data == 'start' or expr.data == 'expr':
        return eval_ast(expr.children[0], env)
    elif expr.data == 'name':
        return eval_name(expr, env)
    elif expr.data == 'val':
        return eval_val(expr, env)
    elif expr.data == 'string':
        return eval_str(expr, env)
    elif expr.data == 'sexp':
        wrapped_expr = expr.children[0]
        return eval_sexpr(wrapped_expr, env)
    else:
        raise ValueError(f'Unknown expression: {expr}')


def eval_let_form(tree: Tree, context: dict):
    new_context = context.copy()
    for binding in tree.children:
        print(binding)
        print(f"Binding: {binding}")  # Add this line for debugging purposes
        if len(binding.children) == 2:
            name = binding.children[0]
            value = eval_ast(binding.children[1], new_context)
            new_context[name] = value
        else:
            raise ValueError("Invalid binding format")
    return eval_ast(tree.children[1], new_context)


SNAME_RE = r"[a-zA-Z\-_0-9:]+"


with open("super_prompt.lark", "r") as file:
    grammar = file.read()


with open("super_prompt.lark", "r") as file:
    grammar = file.read()


dsl_parser = Lark(grammar, start="start", parser="lalr")


def parse_super_prompt(dsl_code):
    tree = dsl_parser.parse(dsl_code)
    transformer = SuperPromptTransformer()
    return transformer.transform(tree)


# Define a transformer class to convert the AST into a more convenient format


class SuperPromptTransformer(Transformer):
    @v_args(inline=True)
    def sexpr(self, items):
        return [items]

    def SNAME(self, token):
        try:
            return str(token)
        except KeyError as e:
            raise ValueError(f"Undefined token: {e}")

    def STRING(self, token):
        return eval(token)

    def binding(self, items):
        return Tree("binding", [items[0], items[1]])

    # unwrap start

    def start(self, items):
        return items[0]


# Parse the DSL code


def parse_super_prompt(dsl_code):
    tree = dsl_parser.parse(dsl_code)
    transformer = SuperPromptTransformer()
    return transformer.transform(tree)


with open("film.super_prompt", "r") as f:
    dsl_code = f.read()


def eval_case(ast, env):
    test_value = eval_ast(ast[0], env)
    cases = ast[1:]
    for i in range(0, len(cases), 2):
        case_value = eval_ast(cases[i], env)
        if test_value == case_value:
            return eval_ast(cases[i + 1], env)
    if len(cases) % 2 == 1:
        return eval_ast(cases[-1], env)
    return None


def eval_loose_match_form(expr, env):
    operand1 = eval_ast(expr.children[0], env)
    operand2 = eval_ast(expr.children[1], env)
    return operand1 == operand2


def eval_question_form(expr, env):
    return input(expr.children[0].value)


def eval_cond_form(expr, env):
    for clause in expr.children:
        condition = eval_ast(clause.children[0], env)
        if condition:
            return eval_ast(clause.children[1], env)
    return None


def eval_case_form(expr, env):
    key = eval_ast(expr.children[0], env)
    for clause in expr.children[1:]:
        if key == clause.children[0].value:
            return eval_ast(clause.children[1], env)
    return None


def eval_let(tree: Tree, context: dict):
    new_context = context.copy()
    bindings = tree.children[0].children
    for binding in bindings:
        name = binding.children[0].children[0]
        value = eval_ast(binding.children[1], new_context)
        new_context[name] = value
    return eval_ast(tree.children[1], new_context)


def eval_if_form(expr, env):
    condition = eval_ast(expr.children[0], env)
    if condition:
        return eval_ast(expr.children[1], env)
    else:
        return eval_ast(expr.children[2], env)


def eval_math_form(expr, env):
    operand1 = eval_ast(expr.children[0], env)
    operand2 = eval_ast(expr.children[1], env)
    op = expr.data
    if op == 'add':
        return operand1 + operand2
    elif op == 'sub':
        return operand1 - operand2
    elif op == 'mul':
        return operand1 * operand2
    elif op == 'div':
        return operand1 / operand2
    else:
        raise ValueError(f'Unknown math operation: {op}')


def eval_comp_form(expr, env):
    operand1 = eval_ast(expr.children[0], env)
    operand2 = eval_ast(expr.children[1], env)
    op = expr.data
    if op == 'eq':
        return operand1 == operand2
    elif op == 'ne':
        return operand1 != operand2
    elif op == 'lt':
        return operand1 < operand2
    elif op == 'le':
        return operand1 <= operand2
    elif op == 'gt':
        return operand1 > operand2
    elif op == 'ge':
        return operand1 >= operand2
    else:
        raise ValueError(f'Unknown comparison operation: {op}')


def eval_str_form(expr, env):
    return ''.join(map(lambda child: str(eval_ast(child, env)), expr.children))


def eval_name(expr, env):
    return env[expr.children[0]]


def eval_val(expr, env):
    return int(expr.children[0].value)


def eval_sexpr(expr, env):
    func_name = expr.children[0].data
    return eval_functions[func_name](expr.children[0], env)


eval_functions = {
    'if_form': eval_if_form,
    'math_form': eval_math_form,
    'comp_form': eval_comp_form,
    'loose_match_form': eval_loose_match_form,
    'question_form': eval_question_form,
    'cond_form': eval_cond_form,
    'case_form': eval_case_form,
    'let_form': eval_let,
    'str_form': eval_str_form,
}


# Implement the eval_* functions for each form as needed.
ast = parse_super_prompt(dsl_code)
print("==== EVAL =====")
result = eval_ast(ast, {})
print("==== EVAL DONE =====")
print("==== RESULT =====")
# print(result)
print("==== END RESULT =====")


# ====================
# Version
