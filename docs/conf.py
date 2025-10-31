# Configuration file for the Sphinx documentation builder.

import os
import sys

os.environ["RANKERS_EAGER_IMPORTS"] = "1"

# --- Import path (ensure the repo root that contains 'rankers/' is present) ---
DOCS_DIR = os.path.dirname(__file__)
REPO_ROOT = os.path.abspath(os.path.join(DOCS_DIR, ".."))  # adjust if needed
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# ---- Docs-only: force eager imports and neutralize LazyModule proxies ----
import os, sys, importlib

# 1) Tell rankers to avoid its own lazy wiring during docs
os.environ["RANKERS_EAGER_IMPORTS"] = "1"

# 2) If transformers' LazyModule gets used to wrap *your* modules, replace it with an eager shim
try:
    import transformers.utils as _tf_utils
    class _EagerLazyModule:
        def __init__(self, name, module_file=None, import_structure=None, module_spec=None):
            # Import the real module immediately; keep a reference
            self.__name__ = name
            self.__dict__["_target"] = importlib.import_module(name)
        def __getattr__(self, item):
            return getattr(self.__dict__["_target"], item)
        def __dir__(self):
            return dir(self.__dict__["_target"])
    # Monkeypatch only if present
    if hasattr(_tf_utils, "LazyModule"):
        _tf_utils.LazyModule = _EagerLazyModule
except Exception:
    pass

# 3) Pre-import the concrete modules autodoc needs (so it never hits a proxy)
_FORCE_IMPORTS = [
    "rankers.modelling.dot.dot",
    "rankers.modelling.cat.cat",
    "rankers.modelling.sparse.sparse",
    "rankers.modelling.seq2seq.seq2seq",
    "rankers.modelling.bge.bge",
    "rankers.pyterrier.dot.dot",
    "rankers.pyterrier.cat.cat",
    "rankers.pyterrier.sparse.sparse",
    "rankers.pyterrier.seq2seq.seq2seq",
]
for _m in _FORCE_IMPORTS:
    try:
        importlib.import_module(_m)
    except Exception as _e:
        # Don't crash the build; you'll still see a clear warning if one fails
        print(f"[docs] WARN: could not import {_m}: {_e}", file=sys.stderr)
# -------------------------------------------------------------------------


# --- Project info ---
project = "rankers"
author = "Andrew"
copyright = "2024, Andrew"
release = "0.0.6"

# --- Extensions ---
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.coverage",
]

# --- Napoleon (Google-style) ---
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# --- Autodoc / Autosummary ---
autoclass_content = "both"  # include class docstring + __init__ docstring
autodoc_inherit_docstrings = True
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented"

autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": False,  # keep clean output by default
    "private-members": False,
    "inherited-members": False,
    "exclude-members": "__weakref__,__dict__,__module__,__annotations__,__doc__,__hash__,__repr__,__str__",
}

# If you still see empty pages, temporarily set:
# autodoc_warningiserror = True


# IMPORTANT: keep skip hook permissive; only skip what you explicitly list.
def skip_member(app, what, name, obj, skip, options):
    if name.startswith("_") and name not in ("__init__", "__call__"):
        return True
    torch_like = {
        "training",
        "eval",
        "parameters",
        "modules",
        "named_parameters",
        "named_modules",
        "children",
        "named_children",
        "apply",
        "cuda",
        "cpu",
        "to",
        "register_buffer",
        "register_parameter",
        "add_module",
        "state_dict",
        "load_state_dict",
        "zero_grad",
        "share_memory",
        "extra_repr",
        "train",
        "__dir__",
        "__sizeof__",
        "__reduce__",
        "__reduce_ex__",
        "__subclasshook__",
        "__init_subclass__",
        "__format__",
        "__new__",
        "__delattr__",
        "__setattr__",
        "__getattribute__",
    }
    if name in torch_like:
        return True
    return skip


def setup(app):
    app.connect("autodoc-skip-member", skip_member)


autodoc_mock_imports = [
    "torch",
    "transformers",
    "datasets",
    "ir_measures",
    "ir_datasets",
    "pandas",
    "numpy",
    "pyterrier",
    "flax",
    "optax",
    "orbax",
    "tira",
    "more_itertools",
]

autosummary_generate = True
autosummary_imported_members = False
autosummary_generate_overwrite = True

# --- Intersphinx ---
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "transformers": ("https://huggingface.co/docs/transformers/main/en/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

# --- Templates / Theme ---
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_theme_options = {
    "navigation_depth": 4,
    "collapse_navigation": False,
    "sticky_navigation": True,
    "includehidden": True,
    "titles_only": False,
}

# Root document
master_doc = "index"
add_module_names = False  # cleaner headings (no 'rankers.modelling...' prefixes)
