"""A setuptools based setup module.
See:
https://packaging.python.org/guides/distributing-packages-using-setuptools/
https://github.com/pypa/sampleproject
"""

# Always prefer setuptools over distutils
from setuptools import setup
import pathlib

here = pathlib.Path(__file__).parent.resolve()

setup(
    name='ExoCompanions',  # Required
    version='0.1.0',  # Required
    description='Package to plot data on binary companion stars from 
exoplanet system data.',  # Optional
    long_description=(here / 'README.md').read_text(encoding='utf-8'),  # 
Optional
    long_description_content_type='text/markdown',  # Optional (see note 
above)
    url='https://github.com/kristinagatto/Exocompanions',  # Optional

    author='Kristina Gatto',  # Optional

    author_email='km14gatt@siena.edu',  # Optional

    classifiers=[  # Optional
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 1 - Beginning stages',

        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        # Pick your license as you wish
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        "Programming Language :: Python :: 3.10",
        'Programming Language :: Python :: 3 :: Only'],

    keywords=['exoplanets', 'stars', 'binary companion'],  # Optional
    packages=['ExoCompanions'],
    platforms=['any'],
    license="MIT",
    python_requires='>=3.6, <4',
    setup_requires=['pytest-runner'],
    # install_requires=['requests'],  # Optional
    tests_require=['pytest'],
    project_urls={  # Optional
        'Source': 'https://github.com/kristinagatto/Exocompanions',
    }
)
