"""Utility for handling layered requirements files."""
import argparse
import logging
from pathlib import Path

import networkx as nx
import yaml


LOG = logging.getLogger(__name__)


def load_config(config_file):
    """Parse dependency config."""
    with open(config_file) as fh:
        try:
            config = yaml.safe_load(fh)
        except yaml.YAMLError as ex:
            raise ValueError(f"Problem loading the resource configuration: {repr(ex)}")
    G = nx.DiGraph()
    G.add_nodes_from(config.keys())
    for node_a, node_list in config.items():
        G.add_weighted_edges_from([(node_b, node_a, 1) for node_b in node_list])
    assert nx.is_directed_acyclic_graph(G), "Dependency tree should be a DAG."
    install_order = list(nx.topological_sort(G))
    LOG.debug("Install order: %s", ", ".join(install_order))
    return install_order


def build_all_requirements(install_paths):
    """Combine requirements files into one list."""
    reqs = []
    for path in install_paths:
        reqs += [
            req for req in path.read_text().strip().split("\n") if not (req.startswith("-") or req.startswith("#"))
        ]
    reqs = sorted(set(reqs))
    LOG.debug("Requirements list: %s", ", ".join(reqs))
    return reqs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate requirements-all.in from multiple requirements*.in files.")
    parser.add_argument(
        "requirements_dir",
        type=str,
        help="Directory containing requirements*.in files",
    )
    parser.add_argument("--edit", action="store_true")
    parser.add_argument("--no-edit", dest="edit", action="store_false")
    parser.set_defaults(edit=True)
    args = parser.parse_args()
    requirements_dir = Path(args.requirements_dir)
    assert requirements_dir.is_dir() and requirements_dir.exists()

    install_order = load_config(requirements_dir / "requirements-config.yml")
    install_paths = []
    for name in install_order:
        path = requirements_dir / f"requirements-{name}.in"
        assert path.exists(), f"Configured to generate requirements for '{name}' but file '{path}' does not exist."
        install_paths.append(path)
        print(path.resolve().relative_to(Path.cwd()))

    reqs = build_all_requirements(install_paths)
    if args.edit:
        (requirements_dir / "requirements-all.in").write_text("\n".join(reqs))
