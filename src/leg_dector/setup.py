from setuptools import setup

package_name = 'leg_dector'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='root@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts':[
            'leg_detection = leg_dector.leg_detection:main',
            'human_following = leg_dector.human_following:main',
            'go_back = leg_dector.go_back:main'
        ],
    },
)
