# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'rankers'
copyright = '2024, Andrew'
author = 'Andrew'
release = '0.0.6'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx.ext.coverage',
]

# Napoleon settings for Google-style docstrings
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

# Autodoc settings
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': False,  # Don't show undocumented members
    'private-members': False,  # Don't show private members
    'inherited-members': False,  # Don't show inherited members
    'exclude-members': '__weakref__,__dict__,__module__,__annotations__,__doc__,__hash__,__repr__,__str__'
}
autodoc_typehints = 'description'
autodoc_typehints_description_target = 'documented'

# Don't document these special methods
def skip_member(app, what, name, obj, skip, options):
    """Skip certain members during documentation generation."""
    # Skip private methods (starting with _)
    if name.startswith('_') and name not in ('__init__', '__call__'):
        return True
    # Skip certain inherited methods
    if name in ('training', 'eval', 'parameters', 'modules', 'named_parameters',
                'named_modules', 'children', 'named_children', 'apply', 'cuda',
                'cpu', 'to', 'register_buffer', 'register_parameter', 'add_module',
                'state_dict', 'load_state_dict', 'zero_grad', 'share_memory',
                'extra_repr', 'train', '__dir__', '__sizeof__', '__reduce__',
                '__reduce_ex__', '__subclasshook__', '__init_subclass__',
                '__format__', '__new__', '__delattr__', '__setattr__', '__getattribute__'):
        return True
    return skip

def setup(app):
    app.connect('autodoc-skip-member', skip_member)
autodoc_mock_imports = [
    'torch',
    'transformers',
    'datasets',
    'ir_measures',
    'ir_datasets',
    'pandas',
    'numpy',
    'pyterrier',
    'flax',
    'optax',
    'orbax',
    'tira',
    'more_itertools',
]

# Autosummary settings
autosummary_generate = True
autosummary_imported_members = False

# Intersphinx configuration
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'torch': ('https://pytorch.org/docs/stable/', None),
    'transformers': ('https://huggingface.co/docs/transformers/main/en/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
}

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# Theme options
html_theme_options = {
    'navigation_depth': 4,
    'collapse_navigation': False,
    'sticky_navigation': True,
    'includehidden': True,
    'titles_only': False
}

# The master toctree document.
master_doc = 'index'
