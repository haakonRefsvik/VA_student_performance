The python-script requires to be run in the virtual python environment. 
By using a virtual environment, you avoid conflicts between libraries required for different projects. 
This ensures that the dependencies of each project are isolated, preventing version clashes.
It also ensures that the specific versions of libraries you are using in your project can be reproduced by others (or on other machines). 

To run any python script, first activate the virtual environment:  
python3 -m venv myenv   
`source myenv/bin/activate`