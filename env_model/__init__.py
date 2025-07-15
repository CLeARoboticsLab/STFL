import os
import importlib

def import_termination_fns():
    cwd = os.path.dirname(os.path.abspath(__file__))
    termination_fns = {}
    for file in os.listdir(cwd):
        if file.endswith('.py') and not file.startswith('__'):
            module_name = file[:-3]
            module = importlib.import_module(f".{module_name}", 'env_model')
            try:
                termination_fns[module_name] = getattr(module.StaticFns, 'termination_fn')
            except AttributeError:
                print(f"No termination_fn found in {module_name}, skipping.")
    return termination_fns

termination_functions = import_termination_fns()
