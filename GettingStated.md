## Getting Started with Julia

### Installing Julia

If you have never used Julia before, then the first thing is to get started with Julia! There are two recommended ways to do this. The simplest is to go to the Julia webpage [https://julialang.org/downloads/](https://julialang.org/downloads/) and download the binary. There will be a latest current release binary for every operating system: pop that in and you are good to go. 

Alternatively, you can download Julia via juliaup. Juliaup is a system that makes controlling and updating Julia very simple, and thus this is what we would recommend for most users. To download Juliaup, you can:

1. On Windows, download julia from the Julia store
2. On MacOSX, Linux, or FreeBSD, you can use the command `curl -fsSL https://install.julialang.org | sh`

Once juliaup is installed, you can manage your Julia installation from the terminal. The main commands to know are:

1. `juliaup help`: shows all of the commands
2. `juliaup update`: updates your Julia version
3. `juliaup default release`: defaults your Julia to use the latest released version.
4. `juliaup default lts`: defaults your Julia to use the long-term stable (LTS) release, currently 1.10.x.

We recommend that you stick to the current default `release`, though production scenarios may want to use LTS. Either way, once this is setup you can just use the command `julia` in your terminal and it should pop right up.

### Using the Package Manager

To use the package manager in Julia, simply hit `]`. You will see your terminal marker change to a blue `pkg>`. The main commands to know in the package manager mode are:

1. `help`: shows all of the commands
2. `st`: show all of your currently installed packages
3. `add PackageX`: adds package X
4. `rm PackageX`: removes package X
5. `activate MyEnv`: activates an environment
6. `instantiate .`: instantiates an environment at a given point.

You can also use the Julia Pkg.jl package, for example `using Pkg; Pkg.add("MyPackage")`.

### Using Pluto Notebooks

Pluto notebooks, such as the one these lecture notes are written in, are a notebook system designed specifically for Julia. Unlike Jupyter notebooks (which, reminder stands for Julia Python R notebooks!), Pluto notebooks are made with an emphasis on reproducible science. As such, it fixes a lot of the issues that arise with the irreproducibility commonly complained about with Jupyter notebooks, such as:

1. Being environment dependent: Jupyter notebook environments depend on your current installation and packages. Pluto notebooks tie the package management into the file. This means that a Pluto notebook is its own package environment, keeps track of package versions, and when opened will automatically install all of the packages to the right version to make sure it fully reproduces on the new computer.
2. Being run-order dependent: in a Jupyter notebook the user is in control of execution and may evaluate blocks out of order. This means that if you pick up a Jupyter notebook and have all of the same packages you can run the notebook from top to bottom and might not get the same answer as what was previously rendered. This is not possible with Pluto (except for differences in psudorandom numbers of course), as Pluto is a fully reactive execution engine, and thus any change in the entire notebook ensures that all downstream cells are automatically updated. This means it has more limitations than a normal Julia program, for example every name can only be used exactly once, but it means that every Pluto notebook is always guaranteed to be up to date.

Thus for reproducibility we will be using Pluto notebooks for these notes.

To get started with Pluto, simply install Pluto (`using Pkg; Pkg.add("Pluto")`) and then get it started: `using Pluto; Pluto.run()`. This will open the Pluto runner in your browser, and you can put the URL for a Pluto notebook in to open it on your computer. When you click edit it will make the document then live.

Note: when you first make this notebook live, it will download all of the required packages. This notebook has a lot of things in there so that might take awhile! Be patient as it's installing ~200 big packages, but they will be reused for the later lectures.
