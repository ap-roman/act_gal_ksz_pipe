import fnl_pipe
from fnl_pipe.util import OutputManager
import os

if __name__ == "__main__":
    print(os.path.abspath(fnl_pipe.util.__file__))
    om = OutputManager('output/')