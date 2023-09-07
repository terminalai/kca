# Keras Core Addons

[![PyPI Latest Release](https://img.shields.io/pypi/v/kca.svg)](https://pypi.org/project/kca/)

[//]: # ([![PyPI Downloads]&#40;https://static.pepy.tech/badge/kca&#41;]&#40;https://pepy.tech/project/kca&#41;)

Keras Core Addons is a repository of contributions that conform to well-established API patterns, but implement new 
functionality not available in Keras Core. Keras Core natively supports a large number of operators, layers, metrics, 
losses, and optimizers. However, in a fast moving field like ML, there are many interesting new developments that cannot 
be integrated into core Keras Core (because their broad applicability is not yet clear, or it is mostly used by a 
smaller subset of the community).

Unlike the package this is inspired by (Tensorflow Addons), Keras Core Addons maintains a near similar structure to 
Keras Core, with the `activations`, `layers` and `losses` structure being continued. This is for potential adoption into
Keras Core being as seamless as possible.

Setup and Installation
-------------

### Installing from PyPI

Yes, we have published `kca` on PyPI! To install `kca` and all its dependencies, the easiest method would be to use 
`pip` to query PyPI. This should, by default, be present in your Python installation. To, install run the following 
command in a terminal or Command Prompt / Powershell:

```bash
$ pip install kca
```

Depending on the OS, you might need to use `pip3` instead. If the command is not found, you can choose to use the
following command too:

```bash
$ python -m pip install kca
```

Here too, `python` or `pip` might be replaced with `py` or `python3` and `pip3` depending on the OS and installation 
configuration. If you have any issues with this, it is always helpful to consult 
[Stack Overflow](https://stackoverflow.com/).

### Installing from Source

To install from source, you need to get the following:

#### Git

Git is needed to install this repository. This is not completely necessary as you can also install the zip file for this 
repository and store it on a local drive manually. To install Git, follow 
[this guide](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git).

After you have successfully installed Git, you can run the following command in a terminal / Command Prompt etc:

```bash
$ git clone https://github.com/terminalai/kca.git
```

This stores a copy in the folder `kca`. You can then navigate into it using `cd kca`.

#### Poetry

This project can be used easily via a tool know as Poetry. This allows you to easily reflect edits made in the original 
source code! To install `poetry`, you can also install it using `pip` by typing in the command as follows:

```bash
$ pip install poetry
```

Again, if you have any issues with `pip`, check out [here](#installing-from-pypi).

After this, you can use the following command to install this library:

```bash
$ poetry install
```