"""The shipped model files, installed as noisestate.examples so that ns.example() finds them.

This directory is the top-level examples/ of the repository; pyproject maps it into the package for
the wheel, so the .yaml files travel with an install and ns.example(name) returns a path to one.

The .py files here are example SCRIPTS, meant to be read.  They travel too, and since this is a
package they are importable as noisestate.examples.<name> -- each does its work under a function or
a __main__ guard, so importing one costs nothing.  They are not part of the public API and their
contents may change with the examples they build.
"""
