"""Simple global context for mesh, replacing jax.experimental.maps."""

_GLOBAL_MESH = None

def set_global_mesh(mesh):
  global _GLOBAL_MESH
  _GLOBAL_MESH = mesh

def get_global_mesh():
  return _GLOBAL_MESH
