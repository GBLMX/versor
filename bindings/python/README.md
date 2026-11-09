# versor Python binding

Minimal PyBind11 binding for the `vsr::runtime` Clifford algebra layer. It is a
thin data/API boundary, not a Blender dependency. Blender integration should
consume this module (or a future external-service client) and convert the
returned arrays to Blender data blocks.

## Scope

Exposed:

- `Signature`: `e3`, `pga3`, `cga3`, `sta`, `cl`, `euclidean`, `of`;
- `Multivector`: construction, coefficients as NumPy, add/sub/scale, geometric
  product, outer/inner products, reverse, norm, unit, inverse, sandwich, grade;
- `E3`: vector, bivector, rotor, rotate, coordinates;
- `P3`: point, plane, translator, rotor, motor, line operations, meets, distances;
- `e3_rotate_many((N,3), rotor)`: one C++ loop for a batch of vectors.

Not exposed yet:

- sparse/batch/matrix classes;
- field/calculus APIs;
- snapshot/IPC;
- any GA object as a Blender dependency-graph type.

## Build

The extension must be built against the Python ABI of the target host. Blender
5.2.2 uses Python 3.14 on this machine; the system Python is also 3.14, so the
local smoke test works. For a different Blender build, set `Python_EXECUTABLE`
to Blender's Python and rebuild.

```bash
cd bindings/python
cmake -S . -B build \
  -Dpybind11_DIR="$(python3 -m pybind11 --cmakedir)" \
  -DPython_EXECUTABLE=/usr/bin/python3 \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 8
PYTHONPATH=build python3 test_binding.py
```

Blender smoke test:

```bash
blender --background --python-expr '
import sys
sys.path.insert(0, "bindings/python/build")
import vsr_runtime as v
r = v.e3_rotor(0, 0, 1, 1.5707963267948966)
print(v.e3_coords(v.e3_rotate(v.e3_vec(1, 0, 0), r)))
'
```

Expected output is approximately `[0, 1, 0]`.

## Examples

- [`examples/blackhole_3d.py`](examples/blackhole_3d.py) builds a 3D
  black-hole/accretion-disk scene in Blender, uses `e3_rotate_many` for the
  initial orbital tangent, integrates Paczyński–Wiita particle orbits, and bakes
  the result into shape keys. The physics notes and references are in
  [`examples/README.md`](examples/README.md).

## Engineering rules

- Do not pass `Multivector`/`Rotor`/`Motor` into Blender custom properties or
  dependency-graph sockets; convert to arrays/quaternions/matrices at this
  boundary.
- Keep large transfers in NumPy/C++ loops. `e3_rotate_many` is the first batch
  example; extend the same pattern rather than looping over Python objects.
- Long-running kernels should release the GIL before this binding becomes a
  hot path. The current batch function is deliberately small and synchronous.
- The `vsr::runtime` headers are header-only; this module does not link the
  legacy `libvsr` graphics library.
