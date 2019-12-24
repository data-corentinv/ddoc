ddoc 
====

<img align='right' width="80" height="80" src="logo.png"/>

ddoc is a python module for generating data documentation in Excel and Word format. 

Installation
------------
This package is currently not available on <a href='https://pypi.org/'>PyPI</a>.

To install it you can clone before the git repository and install with pip : 

``` shell 
pip install ddoc/
```

Quick start
-----------

**Bash**
``` shell 
# for excel documentation 
make excel your_file.csv out_directory='.'
# for word documentation
make word your_file.csv addons='none' out_directory='.' metadata_directory='./metadata.json'
```

**Python**
``` python 
from ddoc import generate, generate_excel

generate('your_file.csv', out_directory='', metadata_directory='')
generate_excel('', out_directory='', metadata_directory='')
```

How to contribute
-----------------
**ddoc** has been developed during my last experience. Your help is very valuable to make it better for everyone. 

Check out call for contributions to see what can be improved, or open an issue if you want something. 
* **Contribute** to the **tests** to make it more reliable. 
* **Contribute** to the **doc** to make it clearer for everyone.
* **Contribute** to the **examples** and **notebooks** to share your experience with other users. 
* **Open issue** if you met problems during development. 

To make easier the contribution feel free to use virtualenv :

``` shell 
make install # create .venv and install requirement packages
. activate.sh # activate .venv
```

For more details, please refer to <a href="./docs/contributing.rst">CONTRIBUTING</a>.

Maintainers
-----------
For any quesiton you can contact the <a href="./AUTHORS.rst">team members</a> whose is responsible for the project.