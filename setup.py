from setuptools import setup, find_packages
from typing import List
def get_requirements(file_path: str) -> List[str]:
    requirements = []
    with open(file_path) as f:
        requirements = f.readlines()
        requirements = [req.replace("\n", "") for req in requirements]
        if "-e ." in requirements:
            requirements.remove("-e .")
    return requirements
        
setup(
    name="Faulty-Detection",
    version="0.0.1",
    author="Devesh",
    author_email="devesh@pm.me",
    install_requirements= get_requirements('requirements.txt'),
    packages=find_packages(),
)