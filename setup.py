#!/usr/bin/env python

import os
import setuptools

here = os.path.abspath(os.path.dirname(__file__))
exec(open(os.path.join(here, 'version.py')).read())

setuptools.setup(
    name        = 'metocean-ml',
    description = 'metocean-ml - Machine Learning Tool for metocean data',
    author      = 'Konstantinos Christakos MET Norway & NTNU',
    url         = 'https://github.com/MET-OM/metocean-ml',
    download_url = 'https://github.com/MET-OM/metocean-ml',
    version = __version__,
    license = 'GPLv3',
    install_requires = [
        'numpy>=1.17',
        'matplotlib>=3.1',
        'pandas',
        'pip',
        'scipy',
        'pip',
        'xarray',
        'scikit-learn',
        'keras',
        'tensorflow',
    ],
    packages = setuptools.find_packages(),
    include_package_data = True,
    setup_requires = ['setuptools_scm'],
    tests_require = ['pytest'],
    scripts = []
)
