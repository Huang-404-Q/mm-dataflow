"""Operator registry: name -> class, so YAML configs can stay declarative."""
from __future__ import annotations

from typing import Dict, Type

_REGISTRY: Dict[str, type] = {}


def register_op(name: str):
    def deco(cls):
        if name in _REGISTRY:
            raise ValueError(f"duplicate operator name: {name}")
        cls.name = name
        _REGISTRY[name] = cls
        return cls

    return deco


def build_op(name: str, params: dict):
    if name not in _REGISTRY:
        raise KeyError(
            f"unknown operator '{name}'. registered: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name](**(params or {}))


def list_ops() -> Dict[str, type]:
    return dict(_REGISTRY)
