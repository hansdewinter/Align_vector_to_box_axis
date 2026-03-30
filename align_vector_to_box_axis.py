#!/usr/bin/env python3
"""
align_vector_axis_gro.py

Rotate a GROMACS .gro structure so that the vector defined by the COMs
of two index groups points along a chosen Cartesian axis (+x, +y, or +z),
then optionally recenter using the COM of a third group.

Examples
--------
Align group A -> group B to +z and center the protein COM in the box:

    python align_vector_axis_gro.py \
        -f input.gro \
        -n index.ndx \
        -g1 GROUP_A \
        -g2 GROUP_B \
        --axis z \
        -gc PROTEIN \
        --center-target box \
        -o aligned.gro

Align to +x and place the third-group COM at the origin:

    python align_vector_axis_gro.py \
        -f input.gro \
        -n index.ndx \
        -g1 GROUP_A \
        -g2 GROUP_B \
        --axis x \
        -gc PROTEIN \
        --center-target origin \
        -o aligned_x.gro
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np


ATOMIC_MASSES = {
    "H": 1.008,
    "C": 12.011,
    "N": 14.007,
    "O": 15.999,
    "P": 30.974,
    "S": 32.06,
    "F": 18.998,
    "NA": 22.990,
    "MG": 24.305,
    "P": 30.974,
    "CL": 35.45,
    "K": 39.098,
    "CA": 40.078,
    "MN": 54.938,
    "FE": 55.845,
    "CU": 63.546,
    "ZN": 65.38,
    "BR": 79.904,
    "I": 126.904,
}


@dataclass
class Atom:
    resid: int
    resname: str
    atomname: str
    atomnr: int
    xyz: np.ndarray
    vel: np.ndarray | None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("-f", "--gro", required=True, help="Input .gro file")
    p.add_argument("-n", "--ndx", required=True, help="Input .ndx file")
    p.add_argument("-g1", "--group1", required=True, help="First group name in .ndx")
    p.add_argument("-g2", "--group2", required=True, help="Second group name in .ndx")
    p.add_argument(
        "--axis",
        choices=["x", "y", "z"],
        default="x",
        help="Target axis for alignment (default: x)",
    )
    p.add_argument(
        "-gc",
        "--group-center",
        help="Third group name in .ndx whose COM will be moved to the chosen target",
    )
    p.add_argument(
        "--center-target",
        default="box",
        help='Target position for third-group COM: "box", "origin", or "x,y,z" in nm',
    )
    p.add_argument("-o", "--output", required=True, help="Output rotated .gro file")
    return p.parse_args()


def read_gro(path: str) -> Tuple[str, List[Atom], np.ndarray]:
    with open(path, "r", encoding="utf-8") as fh:
        lines = fh.readlines()

    if len(lines) < 3:
        raise ValueError("Invalid .gro file: too few lines")

    title = lines[0].rstrip("\n")
    natoms = int(lines[1].strip())
    atom_lines = lines[2:2 + natoms]
    box_line = lines[2 + natoms].split()

    atoms: List[Atom] = []
    for line in atom_lines:
        resid = int(line[0:5])
        resname = line[5:10].strip()
        atomname = line[10:15].strip()
        atomnr = int(line[15:20])
        x = float(line[20:28])
        y = float(line[28:36])
        z = float(line[36:44])

        vel = None
        if len(line) >= 68:
            try:
                vx = float(line[44:52])
                vy = float(line[52:60])
                vz = float(line[60:68])
                vel = np.array([vx, vy, vz], dtype=float)
            except ValueError:
                vel = None

        atoms.append(
            Atom(
                resid=resid,
                resname=resname,
                atomname=atomname,
                atomnr=atomnr,
                xyz=np.array([x, y, z], dtype=float),
                vel=vel,
            )
        )

    box = np.array([float(x) for x in box_line], dtype=float)
    return title, atoms, box


def write_gro(path: str, title: str, atoms: List[Atom], box: np.ndarray) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(f"{title}\n")
        fh.write(f"{len(atoms):5d}\n")
        for a in atoms:
            if a.vel is None:
                fh.write(
                    f"{a.resid:5d}{a.resname:<5s}{a.atomname:>5s}{a.atomnr:5d}"
                    f"{a.xyz[0]:8.3f}{a.xyz[1]:8.3f}{a.xyz[2]:8.3f}\n"
                )
            else:
                fh.write(
                    f"{a.resid:5d}{a.resname:<5s}{a.atomname:>5s}{a.atomnr:5d}"
                    f"{a.xyz[0]:8.3f}{a.xyz[1]:8.3f}{a.xyz[2]:8.3f}"
                    f"{a.vel[0]:8.4f}{a.vel[1]:8.4f}{a.vel[2]:8.4f}\n"
                )
        fh.write(" ".join(f"{x:10.5f}" for x in box) + "\n")


def read_ndx(path: str) -> Dict[str, List[int]]:
    groups: Dict[str, List[int]] = {}
    current = None
    header_re = re.compile(r"^\[\s*(?P<name>[^\]]+?)\s*\](?:\s*[;#].*)?$")

    with open(path, "r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line:
                continue

            # Header line: [ GroupName ] (optionally followed by comment)
            m = header_re.match(line)
            if m:
                current = m.group("name").strip()
                groups[current] = []
                continue

            if current is None:
                continue

            # Only keep integer tokens; ignore accidental junk
            for token in line.split():
                if token in {"[", "]"}:
                    continue
                try:
                    groups[current].append(int(token))
                except ValueError:
                    print(
                        f"Warning: skipping non-integer token '{token}' "
                        f"in {path} at line {lineno}"
                    )

    return groups

def infer_element(atomname: str) -> str:
    name = re.sub(r"[^A-Za-z]", "", atomname).upper()
    if not name:
        return "C"

    for elem in ("CL", "NA", "MG", "ZN", "FE", "CU", "MN", "BR", "CA"):
        if name.startswith(elem):
            return elem

    return name[0]


def atom_mass(atomname: str) -> float:
    elem = infer_element(atomname)
    return ATOMIC_MASSES.get(elem, 12.011)


def center_of_mass(atoms: List[Atom], atom_numbers: List[int]) -> np.ndarray:
    lookup = {a.atomnr: a for a in atoms}
    masses = []
    coords = []

    for nr in atom_numbers:
        if nr not in lookup:
            raise ValueError(f"Atom number {nr} from index file not found in .gro")
        a = lookup[nr]
        masses.append(atom_mass(a.atomname))
        coords.append(a.xyz)

    masses_arr = np.array(masses, dtype=float)
    coords_arr = np.array(coords, dtype=float)
    return (coords_arr * masses_arr[:, None]).sum(axis=0) / masses_arr.sum()


def rotation_matrix_from_vectors(v_from: np.ndarray, v_to: np.ndarray) -> np.ndarray:
    """
    Return rotation matrix R such that R @ v_from = v_to.
    """
    a = v_from / np.linalg.norm(v_from)
    b = v_to / np.linalg.norm(v_to)

    cross = np.cross(a, b)
    dot = float(np.dot(a, b))

    if np.isclose(dot, 1.0, atol=1e-12):
        return np.eye(3)

    if np.isclose(dot, -1.0, atol=1e-12):
        trial = np.array([1.0, 0.0, 0.0])
        if np.allclose(np.abs(a), trial, atol=1e-6):
            trial = np.array([0.0, 1.0, 0.0])
        axis = np.cross(a, trial)
        axis /= np.linalg.norm(axis)
        return -np.eye(3) + 2.0 * np.outer(axis, axis)

    s = np.linalg.norm(cross)
    k = np.array([
        [0.0, -cross[2], cross[1]],
        [cross[2], 0.0, -cross[0]],
        [-cross[1], cross[0], 0.0],
    ])
    return np.eye(3) + k + k @ k * ((1.0 - dot) / (s * s))


def parse_center_target(spec: str, box: np.ndarray) -> np.ndarray:
    spec = spec.strip().lower()
    if spec == "box":
        if len(box) < 3:
            raise ValueError("Box does not contain at least 3 components")
        return np.array([box[0] / 2.0, box[1] / 2.0, box[2] / 2.0], dtype=float)
    if spec == "origin":
        return np.array([0.0, 0.0, 0.0], dtype=float)

    parts = spec.split(",")
    if len(parts) != 3:
        raise ValueError('center-target must be "box", "origin", or "x,y,z"')
    return np.array([float(parts[0]), float(parts[1]), float(parts[2])], dtype=float)


def axis_to_vector(axis: str) -> np.ndarray:
    mapping = {
        "x": np.array([1.0, 0.0, 0.0], dtype=float),
        "y": np.array([0.0, 1.0, 0.0], dtype=float),
        "z": np.array([0.0, 0.0, 1.0], dtype=float),
    }
    return mapping[axis]


def main() -> None:
    args = parse_args()
    title, atoms, box = read_gro(args.gro)
    groups = read_ndx(args.ndx)

    for g in (args.group1, args.group2):
        if g not in groups:
            raise ValueError(f"Group '{g}' not found in {args.ndx}")

    if args.group_center and args.group_center not in groups:
        raise ValueError(f"Group '{args.group_center}' not found in {args.ndx}")

    com1 = center_of_mass(atoms, groups[args.group1])
    com2 = center_of_mass(atoms, groups[args.group2])

    vec = com2 - com1
    if np.linalg.norm(vec) < 1e-12:
        raise ValueError("The COM-to-COM vector is zero; choose different groups")

    target_vec = axis_to_vector(args.axis)
    R = rotation_matrix_from_vectors(vec, target_vec)

    midpoint = 0.5 * (com1 + com2)

    for a in atoms:
        a.xyz = R @ (a.xyz - midpoint)
        if a.vel is not None:
            a.vel = R @ a.vel

    if args.group_center:
        rotated_center_com = center_of_mass(atoms, groups[args.group_center])
        desired_target = parse_center_target(args.center_target, box)
        shift = desired_target - rotated_center_com
    else:
        shift = midpoint

    for a in atoms:
        a.xyz = a.xyz + shift

    new_com1 = center_of_mass(atoms, groups[args.group1])
    new_com2 = center_of_mass(atoms, groups[args.group2])
    new_vec = new_com2 - new_com1
    new_vec_unit = new_vec / np.linalg.norm(new_vec)

    write_gro(
        args.output,
        title + f" | aligned {args.group1}->{args.group2} to +{args.axis}",
        atoms,
        box,
    )

    print(f"Input file:        {args.gro}")
    print(f"Index file:        {args.ndx}")
    print(f"Group 1:           {args.group1}")
    print(f"Group 2:           {args.group2}")
    print(f"Target axis:       +{args.axis}")
    print(f"Original COM1:     {com1}")
    print(f"Original COM2:     {com2}")
    print(f"Old unit vector:   {vec / np.linalg.norm(vec)}")
    print(f"New unit vector:   {new_vec_unit}")

    if args.group_center:
        final_center_com = center_of_mass(atoms, groups[args.group_center])
        print(f"Center group:      {args.group_center}")
        print(f"Final center COM:  {final_center_com}")
        print(f"Center target:     {parse_center_target(args.center_target, box)}")

    print(f"Output file:       {args.output}")


if __name__ == "__main__":
    main()