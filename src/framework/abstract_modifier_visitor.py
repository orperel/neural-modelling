from abc import abstractmethod, ABC

# A couple helper functions first
def _qualname(obj):
    """Get the fully-qualified name of an object (including module)."""
    return obj.__module__ + '.' + obj.__qualname__


def _declaring_class(obj):
    """Get the name of the class that declared an object."""
    name = _qualname(obj)
    return name[:name.rfind('.')]


# Stores the actual visitor methods
_methods = {}


# Delegating visitor implementation
def _visitor_impl(self, arg):
    """Actual visitor method implementation."""
    full_name = _qualname(type(self))
    modifier_class = type(arg)
    method_key = (full_name, modifier_class)
    if method_key not in _methods:
        raise NotImplementedError(f'{full_name} not implemented for {modifier_class}')
    method = _methods[method_key]
    return method(self, arg)


# The actual @visitor decorator
def visitor(arg_type):
    """Decorator that creates a visitor method."""

    def decorator(fn):
        declaring_class = _declaring_class(fn)
        _methods[(declaring_class, arg_type)] = fn

        # Replace all decorated methods with _visitor_impl
        return _visitor_impl

    return decorator


class AbstractModifierVisitor(ABC):
    """
    Inheritors must implement the visitor interface for at least one modifier class
    """

    @abstractmethod
    def visit(self, modifier):
        pass
