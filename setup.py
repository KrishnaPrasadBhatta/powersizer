import os

from setuptools import setup, find_packages

here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, 'README.txt')) as f:
    README = f.read()
with open(os.path.join(here, 'CHANGES.txt')) as f:
    CHANGES = f.read()

requires = [
    'plaster_pastedeploy',
    'pyramid',
    'pyramid_jinja2',
    'pyramid_debugtoolbar',
    'waitress',
    'alembic',
    'pyramid_retry',
    'pyramid_tm',
    'SQLAlchemy',
    'transaction',
    'zope.sqlalchemy',
    'pytest',
    'shapely',
    'numpy',
    'fiona>=1.7',
    'Rtree>=0.8.3',
    'matplotlib',
    'pandas',
    'seaborn',
    'jupyterlab',
    'WebTest',
    'pytest-cov',
    'pyproj',
    'sklearn'
]

tests_require = [
    'WebTest',
    'pytest',
    'pytest-cov',
]

setup(
    name='mpa_sim',
    version='0.2',
    description='mpa.sim',
    long_description=README + '\n\n' + CHANGES,
    classifiers=[
        'Programming Language :: Python',
        'Framework :: Pyramid',
        'Topic :: Internet :: WWW/HTTP',
        'Topic :: Internet :: WWW/HTTP :: WSGI :: Application',
    ],
    author='Peyman Behbahani',
    author_email='peyman_behbahani@dellteam.com',
    url='',
    keywords='web pyramid pylons',
    packages=find_packages(exclude=['tests']),
    include_package_data=True,
    zip_safe=False,
    extras_require={
        'testing': tests_require,
    },
    install_requires=requires,
    entry_points={
        'paste.app_factory': [
            'main = mpa_sim:main',
        ],
        'console_scripts': [
            'initialize_mpa_sim_db=mpa_sim.scripts.initialize_db:main',
        ],
    },
)
