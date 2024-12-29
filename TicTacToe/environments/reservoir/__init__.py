import pkgutil
import importlib
import inspect

from .Decoding import *
from .Mean import *
from .ReservoirGameEnvironment import *
from .Sampled import *
from .SampledRoll import *
from .SampledWithEmpty import *
from .Sequence import *
from .ConductanceTable import *

# Initialize an empty list to store all classes
__all__ = []

# Iterate over all modules in the current package
for loader, module_name, is_pkg in pkgutil.walk_packages(__path__):
    # Import the module
    module = importlib.import_module(f"{__name__}.{module_name}")

    # Iterate over all members of the module
    for name, obj in inspect.getmembers(module):
        # Check if the member is a class
        if inspect.isclass(obj):
            # Add the class to the current module's globals
            globals()[name] = obj
            # Add the class name to __all__
            __all__.append(name)
