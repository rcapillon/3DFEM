from setuptools import setup

setup(
    name='3DFEM',
    version='0.1',
    packages=['meshing', 'solvers', 'elements', 'plotting', 'materials', 'structure', 'node_selection',
              'random_generators', 'initial_conditions', 'boundary_conditions'],
    package_dir={'': 'src'},
    url='https://github.com/rcapillon/3DFEM',
    license='GPL-3.0',
    author='Rémi Capillon',
    author_email='rcapillon@yahoo.fr',
    description=''
)
