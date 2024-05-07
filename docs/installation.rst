********************************************************************************
Installation
********************************************************************************

This page provides a guide for installing the ``ARA`` plugin on your system.
The plugin can be installed using an installer, or a package manager for Python such as ``pip`` or ``conda``.

The following instructions will guide you through each method.
Alternatively, you can clone the ``aixd_ara`` source code directly from
our `repository <https://github.com/gramaziokohler/aixd_ara>`_.

One-click installer
===================

The one-click installer is the easiest way to install the plugin on your system:

* Download the ZIP files.
* Extract them in any forlder.
* Run the installer file.

Installation using pip
======================

The most popular package manager for Python is ``pip``.
You can use it to install packages from the Python Package Index and other indexes.

**Step 1: Update pip**

It is good practice to ensure that you are using the latest version of ``pip``.
To update ``pip``, run the following command:


.. code-block:: bash

    python -m pip install --upgrade pip

**Step 2: Install plugin**

To install `ara_aixd` using ``pip``, execute the following command:

.. code-block:: bash

    pip install ara_aixd

And then install the plugin in Rhino/Grasshopper using the following command:

.. code-block:: bash

    python -m compas_rhino.install -v 7.0


Installation using conda
========================

``conda`` is an open-source package management system and environment
 management system that runs on Windows, macOS, and Linux.
 It’s very popular in the realm of scientific computing.

**Step 1: Create a conda environment (Optional)**

It's often beneficial to create a new environment for your project. This can be done using the following command:

.. code-block:: bash

    conda create --name project_name python=3.x


Replace *project_name* with your desired environment name and *3.x* with the
specific version of Python you want to use.
This package requires python 3.9 or a higher.

Activate the new environment by running:

.. code-block:: bash

    conda activate project_name

**Step 2: Install plugin**

To install ``aixd_ara`` in your ``conda`` environment, use the following command:

.. code-block:: bash

    conda install -c conda-forge aixd_ara

Finally, install the plugin in Rhino/GH using the following command:

.. code-block:: bash

    python -m compas_rhino.install -v 7.0


Verify installation
===================

After installation, you can verify that the plugin has been successfully installed by running:

.. code-block:: bash

    python -c "import aixd_ara; print(aixd_ara.__version__)"


If everything worked out correctly, the version of the installed plugin will be printed on
the screen, and you can start using the plugin in your projects.
