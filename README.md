The python-script requires to be run in the virtual python environment. 
By using a virtual environment, you avoid conflicts between libraries required for different projects. 
This ensures that the dependencies of each project are isolated, preventing version clashes.
It also ensures that the specific versions of libraries you are using in your project can be reproduced by others (or on other machines). 

To run any python script, first create the virtual environment:  

`python3 -m venv myenv`

Then activate it by running:

`source myenv/bin/activate`

From here you can install all the packages in requirements.txt by running the following command.

`pip install -r requirements.txt `

This should download everything you need. If there is a need to download other packages. You can download them by running:

`pip install <package>` where <package> is the name of the package you want to download.