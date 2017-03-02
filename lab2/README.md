# Caffe and Python 
* Lab 2 Report
* Tuan Nguyen
* Deep Learning - Spring 2017
* Advisor: Dr. Martin Hagan

## Python Basics
1. What is an ”immutable” object? Give some examples.

  Objects whose value is unchangeable once they are created are called immutable. Example:

  ```python
  >>> str = "Deep Learning"
  >>> str
  'Deep Learning'
  >>> str[5]
  'L'
  >>> str[5] = 'l'
  Traceback (most recent call last):
    File "<stdin>", line 1, in <module>
  TypeError: 'str' object does not support item assignment
  ```

2. What is a ”module” in Python?

  A module is a file containing Python definitions and statements. The file name is the module name with the suffix .py appended

3. What is a method?

  In Python, a method is a function that “belongs to” an object (class instance objects and other objects such as list). In other languages such as Java, C++, a method is a member function of a class instance object.

4. What is a class?

  A class is a structure in Object-Oriented Programming that wrap both variables (attributes) and functions (behaviors). 

5. Explain how indentation is used in Python

  Python uses indentation to structure the code. In Java or C/C++, characters {} are used to group statements into code block. In Python, statements that have the same indentation belong to the same block. 

6. Identify differences between the interactive interpreter and Pycharm debugger.
 * Debugger uses pdb package while interpreter doesn't. 
 * Debugger is much more flexible and informative than intepreter (breakpoints, controls, watch,..)

7. How do you execute a Python program from the command line?

 * At bash, run `python 'file.py'`
 * At bash run `python` to go into the intepreter environment, then `execfile('file.py')`
 * At bash, run `ipython` then `run file.py`
  
## Square Diamon Pattern Recognition
### Procedures
* Append following lines to ~/.bashrc to set up environment variables
  
  ```bash
  #tuan added March 1 for caffe import python
  export PYTHONPATH=$PYTHONPATH:/home/martin/caffe/python
  export CAFFE_ROOT=/home/martin/caffe
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
  ```
* Go to iPython interactive intepreter
  ```bash
  martin@martin-XPS-8700:~/Desktop/tuandn/git_repo/deep_learning/lab2$ ipython
  Python 2.7.6 (default, Jun 22 2015, 17:58:13) 
  Type "copyright", "credits" or "license" for more information.

  IPython 2.3.0 -- An enhanced Interactive Python.
  ?         -> Introduction and overview of IPython's features.
  %quickref -> Quick reference.
  help      -> Python's own help system.
  object?   -> Details about 'object', use 'object??' for extra details.

  In [1]: 
  ```
  
* Run `run SquareDiamondLMDB.py`
  
  ```python
  In [2]: run SquareDiamondLMDB.py
  In [3]: who
  N	 X	 caffe	 d1	 d2	 d3	 d4	 d5	 datum	 
  diamond	 env	 file_content	 fo	 i	 lmdb	 map_size	 my_root np	 
  os	 s1	 s2	 s3	 s4	 s5	 square	 str_id	 txn	 
  y	
  ```
  
* Run `run SquareDiamon.py`
  ```python
  In [4]: run SquareDiamond.py
  ```
  ![]() ![]()
  
* Run `run SquareDiamon_Batch.py`
