## Installing Anaconda
First, we need to install Anaconda. You can download the [installer here](https://www.continuum.io/downloads). 

### Why Anaconda?
---
It’s prepackaged with many useful packages and scientific libraries such as:
- Pandas 
- SciPy
- NumPy
- Scikit-learn
- Matplotlib
- and a lot more!

Using a normal terminal, we would need to install all these libraries one by one. 
Using Anaconda makes it a lot easier for us to start coding without the hassle. 

You have 2 options: 
- Run scripts using Jupyter Notebook (recommended)
- Run scripts using Anaconda Prompt

### Jupyter Notebook
Open a command prompt or terminal and type in 
```shell
jupyter notebook
```
This will launch a new browser window (or a new tab) showing the Notebook Dashboard, a sort of control panel that allows (among other things) to select which notebook to open.

Jupyter allows you to run parts of your code incrementally. 
- The `+` button allows you to add another cell. 
- Each cell can contain code, which can be excuted using the `Run Cell` button (which can also be located under `Cell`). 
- You can also insert comments/notes in between code by switching the cell type from `Code` to `Markdown` (located in the dropdown). 

### Running your first Python script in Anaconda
---
Create a python script `script.py` containing:

```shell
print("Hello World!")
```

Search for and open the Anaconda Prompt (it should look like a command prompt or terminal), navigate to you directory, and type in:

``` shell
python script.py
```

And voila! If no error occurs, you're good to go. :)