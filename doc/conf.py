"""Sphinx configuration for LADClassifier."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath('..'))

from lad import __version__


project = 'LADClassifier'
author = 'Gregory Morse'
copyright = '2026, Gregory Morse'
version = __version__
release = __version__

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.viewcode',
    'numpydoc',
    'sphinx_gallery.gen_gallery',
]

autosummary_generate = True
autodoc_default_options = {
    'members': True,
    'exclude-members': ','.join([
        'get_metadata_routing',
        'set_fit_request',
        'set_inverse_transform_request',
        'set_score_request',
        'set_transform_request',
    ]),
}
numpydoc_show_class_members = False

templates_path = ['_templates']
exclude_patterns = ['_build', '_templates']
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_css_files = ['css/project-template.css']
html_title = 'LADClassifier documentation'
htmlhelp_basename = 'LADClassifierdoc'

sphinx_gallery_conf = {
    'examples_dirs': '../examples',
    'gallery_dirs': 'auto_examples',
    'backreferences_dir': 'generated',
    'doc_module': 'lad',
    'reference_url': {'lad': None},
}


def setup(app):
    app.add_js_file('js/copybutton.js')
