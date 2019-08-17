"""
This file serves to check whether the environment works as expected from the perspective of the
Problem Specification.
"""

# this should live in ackrep_core
import yaml
from ipydex import IPS

fname = "metadata.yml"
with open(fname, 'r') as stream:
    txt = stream.read()

txt = txt.replace("{|", "")
txt = txt.replace("|}", "")
try:
    d = (yaml.safe_load(txt))
    print(d)
except yaml.YAMLError as exc:
    print(exc)


# IPS()
