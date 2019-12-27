ddoc 
====

.. image:: ../logo.png
   :width: 300
   :align: center

ddoc is a python module for generating data documentation in Excel and Word format. 

Installation
------------
This package is currently not available on <a href='https://pypi.org/'>PyPI</a>.

To install it you can clone before the git repository and install with pip : 

.. code-block:: html

   pip install ddoc/

Quick start
-----------

**Bash**

.. code-block:: html

   # for excel documentation 
   make excel data_location='your_file.csv'
   # for word documentation
   make word data_location='your_file.csv'

**Python**

.. code-block:: python
   
   from ddoc import generate, generate_excel
   generate('your_file.csv')
   generate_excel('')


How to contribute
-----------------
**ddoc** has been developed during my last experience. Your help is very valuable to make it better for everyone. 

Check out call for contributions to see what can be improved, or open an issue if you want something. 

* **Contribute** to the **tests** to make it more reliable.  
* **Contribute** to the **doc** to make it clearer for everyone.
* **Contribute** to the **examples** and **notebooks** to share your experience with other users. 
* **Open issue** if you met problems during development. 

To make easier the contribution feel free to use virtualenv :

.. code-block:: html

   # create .venv and install requirement packages
   make install 
   # activate .venv
   . activate.sh 


For more details, please refer to **contributing.rst** file.

Maintainers
-----------
For any quesiton you can contact the **authors.rst** file whose is responsible for the project.

TODO
----
- Sphinx documentation (create gallery examples)
- Explains differents addons
- Explains how to complete metadata.json file
- Correlation matrix of continuous features (only V cramer yet)
- Tests
