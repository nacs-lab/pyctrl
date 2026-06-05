"""A tiny name -> factory registry for hardware backends.

Lets a run loop look a device up by name instead of importing a specific module.
Adoption is incremental: a driver opts in with ``@register("<name>")`` on its
class or a factory function; until then the registry is simply empty.
"""

_REGISTRY = {}


def register(name):
    """Decorator: register a device factory (class or callable) under ``name``."""
    def _deco(factory):
        _REGISTRY[name] = factory
        return factory
    return _deco


def create(name, *args, **kwargs):
    """Instantiate the device registered under ``name``."""
    return _REGISTRY[name](*args, **kwargs)


def available():
    """Sorted list of registered device names."""
    return sorted(_REGISTRY)
