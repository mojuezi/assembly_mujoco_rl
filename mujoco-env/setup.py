from setuptools import setup, find_packages

setup(
    name='mujoco_env',
    version='0.4.1',
    description="aral_rl: Robot Simulation Framework based Mujoco for Reinforcement Learning",
    url="http://git.aubo-robotics.cn:8001/aral/rl",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.10",
    # 依赖通过 pip install -r requirements.txt 统一管理，此处留空以保持精简
    install_requires=[],
    classifiers=[
        "License :: OSI Approved :: MIT License",
    ],
)
