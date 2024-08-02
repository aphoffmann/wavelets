from setuptools import setup

config = {
    'description': 'Continuous wavelet analysis in Python',
    'author': "Aaron O'Leary",
    'url': 'http://github.com/aaren/wavelets',
    'author_email': 'eeaol@leeds.ac.uk',
    'version': '0.11',
    'packages': ['wavelets'],
    'name': 'wavelets',
    'setup_requires': ['numpy'],
    'install_requires': ['numpy', 'scipy']
}

setup(**config)
