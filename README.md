This Python3 script takes in a Gromacs .gro file and rotates it so that a vector, 
defined by two Gromacs index groups, gets aligned to one of the three main axes (x,
y, or z). The user can also specify a group that is used to move its center-of-mass
to the box center, the origin, or a specified coordinate.

USAGE:

```bash
$ ./align_vector_to_box_axis.py [-h] -f GRO -n NDX -g1 GROUP1 -g2 GROUP2 [--axis {x,y,z}] [-gc GROUP_CENTER]
                                   [--center-target CENTER_TARGET] -o OUTPUT
```

OPTIONS:

```bash
  -h, --help            show this help message and exit
  -f GRO, --gro GRO     Input .gro file
  -n NDX, --ndx NDX     Input .ndx file
  -g1 GROUP1, --group1 GROUP1
                        First group name in .ndx
  -g2 GROUP2, --group2 GROUP2
                        Second group name in .ndx
  --axis {x,y,z}        Target axis for alignment (default: x)
  -gc GROUP_CENTER, --group-center GROUP_CENTER
                        Third group name in .ndx whose COM will be moved to the chosen target
  --center-target CENTER_TARGET
                        Target position for third-group COM: "box", "origin", or "x,y,z" in nm
  -o OUTPUT, --output OUTPUT
                        Output rotated .gro file
```
               
