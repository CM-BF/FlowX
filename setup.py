from setuptools import setup

setup(
    name='xgraph',
    version='0.0.1',
    url='',
    license='MIT',
    author='shurui.gui',
    author_email='shurui.gui@tamu.edu',
    description='',
    package_dir={"xgraph": "xgraph"},
    entry_points = {
        'console_scripts': [
            'xgraphtg = xgraph.kernel.pipeline:xgraph_main',
            # 'xgraphtl = xgraph.kernel.launch:launch'
        ]
    }
)
