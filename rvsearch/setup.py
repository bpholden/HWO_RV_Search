from setuptools import setup, find_packages
import os
import re

def get_property(prop, project):
    result = re.search(r'{}\s*=\s*[\'"]([^\'"]*)[\'"]'.format(prop),
                       open(project + '/__init__.py').read())
    return result.group(1)

def read_reqs(path):
    # Missing file is tolerated so a source tree without the docs requirements
    # still builds.
    if not os.path.exists(path):
        return []
    return [line.strip() for line in open(path, 'r').readlines() if line.strip()]

reqs = read_reqs('requirements.txt')

setup(
    name="rvsearch",
    version=get_property('__version__', 'rvsearch'),
    author="Lee Rosenthal, BJ Fulton",
    packages=find_packages(),
    entry_points={'console_scripts': ['rvsearch=rvsearch.cli:main']},
    python_requires=">=3.10",
    install_requires=reqs,
    extras_require={'docs': read_reqs('sphinx/requirements.txt')},
    data_files=[
        (
            'rvsearch_example_data',
            [
                'example_data/HD128311.csv',
                'example_data/recoveries.csv'
            ]
        )
    ],
)
