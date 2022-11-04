# from distutils.core import setup
from setuptools import setup


with open("README", 'r') as file:
    readme = file.read()


setup(
    name="deep_personality",
    version="0.1.2",
    packages=["dpcv"],
    url="https://testpypi.python.org/pypi/deep_personality",
    license="LICENSE",
    description="an open source bench mark for automatic personality recognition",
    long_description=readme,
    author="LiaoRongFan, SongSiYang",
    author_email="rongfan.liao@hotmail.com",
)
