"""Convenience wrapper used during the recorded demonstration."""

from __future__ import annotations

import sys

from main import build_parser


if __name__ == "__main__":
    parser = build_parser()
    arguments = parser.parse_args(["play", *sys.argv[1:]])
    arguments.func(arguments)

