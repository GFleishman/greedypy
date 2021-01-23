import setuptools

setuptools.setup(
    name="greedypy",
    version="0.0.6",
    author="Greg M. Fleishman",
    author_email="greg.nli10me@gmail.com",
    description="Fast deformable registration in python",
    url="https://github.com/GFleishman/greedypy",
    license="MIT",
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires=[
        'argparse',
        'numpy',
        'scipy',
        'nibabel',
        'pynrrd',
        'pyfftw',
        'zarr',
        'numcodecs',
    ]
)
