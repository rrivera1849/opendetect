"""Auto-discover and register all detectors in this package."""

import importlib
import pkgutil

# Import every module in this package so that @register_detector decorators
# execute and populate the registry.
_package_path = __path__
_package_name = __name__

for _importer, _modname, _ispkg in pkgutil.iter_modules(_package_path):
    if _modname == "base":
        continue  # skip the abstract base
    importlib.import_module(f"{_package_name}.{_modname}")
