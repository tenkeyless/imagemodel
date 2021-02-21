import imp
import sys
from importlib.abc import Loader
from importlib.machinery import ModuleSpec
from importlib.util import module_from_spec, spec_from_file_location
from types import ModuleType
from typing import Optional


def load_module(module_name: str, file_path: str) -> Optional[ModuleType]:
    """
    Load a module by name and search path.

    Parameters
    ----------
    module_name : str
        Module name.
    file_path : str
        File path. It includes '.py'.

    Returns
    -------
    Optional[ModuleType]
        Returns `None` if Module could not be loaded.

    Examples
    --------
    >>> load_module('name', 'models/name.py')
    """
    if sys.version_info >= (3, 5):
        spec: Optional[ModuleSpec] = spec_from_file_location(module_name, file_path)
        if not spec:
            return None

        module: ModuleType = module_from_spec(spec)
        assert isinstance(spec.loader, Loader)
        spec.loader.exec_module(module)

        return module
    else:
        mod = imp.load_source(module_name, file_path)
        return mod
