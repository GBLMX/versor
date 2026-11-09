#!/usr/bin/env python3
"""Build a 3D black-hole / accretion-disk scene in Blender from the reference PNG.

The disk is not a rotating texture: every glowing particle is integrated with a
Paczynski-Wiita pseudo-Newtonian potential, which reproduces the relativistic
ISCO and precession qualitatively while staying cheap enough to bake.  The
initial tangential directions can be produced by the Clifford-algebra binding
(``vsr_runtime.e3_rotate_many``); if the binding is absent the example falls
back to the equivalent NumPy cross product.

Run headless:

    blender --background --python blackhole_3d.py -- \
        --out /tmp/black_hole_3d.blend \
        --render /tmp/black_hole_3d.png

The saved .blend contains the baked particle animation (absolute shape keys),
so it renders without this script or a Python handler.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector

# ---------------------------------------------------------------------------
# Optional Clifford-algebra runtime.  The example is intentionally usable
# without the compiled binding; the fallback is the same operation.
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
for candidate in (_HERE / "build", _HERE.parent / "build", _HERE.parent.parent / "build"):
    if candidate.is_dir():
        sys.path.insert(0, str(candidate))
try:
    import vsr_runtime as vsr  # type: ignore
except Exception:
    vsr = None

# ---------------------------------------------------------------------------
# Physics parameters.  Scene units use r_s = 1, M = r_s/2 = 0.5.
# For Schwarzschild: r_ph = 3M = 1.5 r_s, r_ISCO = 6M = 3 r_s, and the
# critical photon impact parameter is b_c = 3*sqrt(3) M = 2.598 r_s.
# ---------------------------------------------------------------------------
M = 0.5
RS = 1.0
PHOTON_SPHERE = 3.0 * M
CRITICAL_IMPACT_PARAMETER = 3.0 * math.sqrt(3.0) * M
ISCO = 6.0 * M
DISK_INNER = ISCO
DISK_OUTER = 7.5
FRAMES = 120
FPS = 24
PHYSICS_DT_PER_FRAME = 0.06
SUBSTEPS = 3
PARTICLES = 4200
STARS = 1600
SEED = 7
CAMERA_LOCATION = (0.0, 3.0, 24.0)
CAMERA_FOV_DEG = 50.0


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="/tmp/black_hole_3d.blend")
    p.add_argument("--render", default="/tmp/black_hole_3d.png")
    p.add_argument("--no-render", action="store_true")
    p.add_argument("--frames", type=int, default=FRAMES)
    p.add_argument("--samples", type=int, default=32)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--render-frame", type=int, default=1)
    return p.parse_args(argv)


def tangential_directions(radial: np.ndarray) -> np.ndarray:
    """Return unit prograde tangent vectors for radial vectors ``(x, 0, z)``."""
    norm = np.linalg.norm(radial, axis=1)
    safe_norm = np.where(norm > 0.0, norm, 1.0)
    if vsr is not None:
        try:
            rotor = vsr.e3_rotor(0.0, 1.0, 0.0, math.pi / 2.0)
            rotated = np.asarray(vsr.e3_rotate_many(radial.astype(np.float64), rotor))
            # ``e3_rotate_many`` rotates the whole radial vector; the rotor
            # preserves magnitude, so divide by |r| to get a unit tangent.
            rotated[:, 1] = 0.0
            return rotated / safe_norm[:, None]
        except Exception:
            pass
    tangent = np.empty_like(radial)
    tangent[:, 0] = -radial[:, 2]
    tangent[:, 1] = 0.0
    tangent[:, 2] = radial[:, 0]
    return tangent / safe_norm[:, None]


def acceleration(pos: np.ndarray) -> np.ndarray:
    """Paczynski-Wiita acceleration in arbitrary units."""
    r2 = np.einsum("ij,ij->i", pos, pos)
    r = np.sqrt(np.maximum(r2, 1e-12))
    denom = np.maximum(r - RS, 0.18)
    scale = -M / (r * denom * denom)
    return pos * scale[:, None]


def initial_state(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    # A little more area near the inner disk makes the hot ring read clearly.
    u = rng.random(n) ** 0.75
    radius = DISK_INNER + (DISK_OUTER - DISK_INNER) * u
    angle = rng.uniform(0.0, 2.0 * math.pi, n)
    x = radius * np.cos(angle)
    z = radius * np.sin(angle)
    # Thin disk with a small flared/thickened tail.
    thickness = 0.025 + 0.022 * (radius - DISK_INNER) / (DISK_OUTER - DISK_INNER)
    y = rng.normal(0.0, 1.0, n) * thickness

    pos = np.column_stack((x, y, z))
    radial = np.column_stack((x, np.zeros(n), z))
    tangent = tangential_directions(radial)

    # Circular speed for the PW potential: v = sqrt(M r) / (r - r_s).
    speed = np.sqrt(M * radius) / (radius - RS)
    vel = tangent * speed[:, None]
    # Small eccentricity and vertical motion: gives precession and a live disk.
    vel[:, 0] += rng.normal(0.0, 0.006, n) * tangent[:, 2]
    vel[:, 2] -= rng.normal(0.0, 0.006, n) * tangent[:, 0]
    vel[:, 1] += rng.normal(0.0, 0.010, n)
    return pos, vel


def simulate(
    pos0: np.ndarray,
    vel0: np.ndarray,
    frames: int,
    dt_per_frame: float,
    substeps: int,
) -> np.ndarray:
    """Velocity-Verlet integration; returns positions with shape (frames, n, 3)."""
    dt = dt_per_frame / substeps
    pos = pos0.copy()
    vel = vel0.copy()
    out = np.empty((frames, pos.shape[0], 3), dtype=np.float64)
    for frame in range(frames):
        out[frame] = pos
        for _ in range(substeps):
            a1 = acceleration(pos)
            vel += 0.5 * a1 * dt
            pos += vel * dt
            a2 = acceleration(pos)
            vel += 0.5 * a2 * dt
            # Numerical safety: a particle that grazes the horizon is put back
            # on a stable circular orbit instead of poisoning the whole bake.
            r = np.linalg.norm(pos, axis=1)
            bad = r < RS * 1.25
            if np.any(bad):
                angle = np.arctan2(pos[bad, 2], pos[bad, 0])
                rr = np.maximum(r[bad], DISK_INNER)
                pos[bad, 0] = rr * np.cos(angle)
                pos[bad, 2] = rr * np.sin(angle)
                pos[bad, 1] = 0.0
                tangent = tangential_directions(pos[bad])
                vel[bad] = tangent * (np.sqrt(M * rr) / (rr - RS))[:, None]
    return out


def disk_colors(rng: np.random.Generator, n: int, radius: np.ndarray) -> np.ndarray:
    # Shakura-Sunyaev zero-torque thin-disk temperature proxy:
    #   T(x) = T0 * x^(-3/4) * (1 - sqrt(1/x))^(1/4),  x = r/r_in.
    # The inner edge is dark in the strict zero-torque model; keep a small
    # floor so the particle ring remains visible in the viewport/render.
    x = np.maximum(radius / DISK_INNER, 1.0001)
    temp = x ** -0.75 * np.maximum(1.0 - np.sqrt(1.0 / x), 0.0) ** 0.25
    temp = temp / max(float(temp.max()), 1e-9)
    hot = 0.25 + 0.75 * np.clip(temp, 0.0, 1.0)
    # Hot core: blue-white.  Cool outer disk: deep orange.
    c = np.column_stack(
        (
            0.45 + 0.55 * hot,
            0.18 + 0.75 * hot,
            0.04 + 0.90 * hot,
            np.ones(n),
        )
    )
    jitter = 0.92 + 0.16 * rng.random(n)
    c[:, :3] *= jitter[:, None]
    return np.clip(c, 0.0, 1.0)


def action_fcurves(action: bpy.types.Action):
    """Yield F-curves for both legacy and Blender 5 layered actions."""
    legacy = getattr(action, "fcurves", None)
    if legacy is not None:
        yield from legacy
        return
    for layer in action.layers:
        for strip in layer.strips:
            for channelbag in getattr(strip, "channelbags", ()):
                yield from channelbag.fcurves


def add_particle_mesh(
    name: str,
    positions: np.ndarray,
    colors: np.ndarray,
    size: float,
    material: bpy.types.Material,
) -> bpy.types.Object:
    """One mesh containing an octahedron per particle, baked as shape keys."""
    frames, n, _ = positions.shape
    offsets = np.array(
        [
            [0.0, size, 0.0],
            [size, 0.0, 0.0],
            [0.0, 0.0, size],
            [-size, 0.0, 0.0],
            [0.0, 0.0, -size],
            [0.0, -size, 0.0],
        ],
        dtype=np.float64,
    )
    base_vertices = (positions[0][:, None, :] + offsets[None, :, :]).reshape(-1, 3)

    faces = []
    for p in range(n):
        b = p * 6
        faces.extend(
            [
                (b + 0, b + 2, b + 1),
                (b + 0, b + 3, b + 2),
                (b + 0, b + 4, b + 3),
                (b + 0, b + 1, b + 4),
                (b + 5, b + 1, b + 2),
                (b + 5, b + 2, b + 3),
                (b + 5, b + 3, b + 4),
                (b + 5, b + 4, b + 1),
            ]
        )

    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(base_vertices.tolist(), [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    obj.data.materials.append(material)

    attr = mesh.color_attributes.new(name="Col", type="FLOAT_COLOR", domain="POINT")
    flat_colors = np.repeat(colors, 6, axis=0).astype(np.float32)
    attr.data.foreach_set("color", flat_colors.reshape(-1))

    basis = obj.shape_key_add(name="Basis", from_mix=False)
    basis.data.foreach_set("co", base_vertices.astype(np.float32).reshape(-1))
    keys = obj.data.shape_keys
    assert keys is not None
    keys.use_relative = True

    for frame in range(frames):
        key = obj.shape_key_add(name=f"F{frame + 1:03d}", from_mix=False)
        frame_vertices = (positions[frame][:, None, :] + offsets[None, :, :]).reshape(-1)
        key.data.foreach_set("co", frame_vertices.astype(np.float32))
        # Linear cross-fade between adjacent baked positions.  Every key must
        # explicitly hold zero until the frame before it becomes active;
        # otherwise Blender evaluates a future key as its first value (1.0)
        # from frame 1 onward and all particle positions get summed.
        if frame > 0:
            key.value = 0.0
            key.keyframe_insert(data_path="value", frame=frame)
        key.value = 1.0
        key.keyframe_insert(data_path="value", frame=frame + 1)
        key.value = 0.0
        key.keyframe_insert(data_path="value", frame=frame + 2)

    if keys.animation_data and keys.animation_data.action:
        for fcurve in action_fcurves(keys.animation_data.action):
            for point in fcurve.keyframe_points:
                point.interpolation = "LINEAR"
    return obj


def emission_material(name: str, strength: float) -> bpy.types.Material:
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    emission = nodes.new("ShaderNodeEmission")
    attr = nodes.new("ShaderNodeAttribute")
    attr.attribute_name = "Col"
    emission.inputs["Strength"].default_value = strength
    links.new(attr.outputs["Color"], emission.inputs["Color"])
    links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return mat


def camera_basis() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Camera position plus right/up/forward unit vectors."""
    cam = np.asarray(CAMERA_LOCATION, dtype=np.float64)
    forward = -cam
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.array([0.0, 1.0, 0.0]))
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    return cam, right, up, forward


def lensed_star_positions(n: int, rng: np.random.Generator) -> np.ndarray:
    """Backward-integrate camera rays and place stars on their escape directions.

    This uses the Cartesian null-geodesic form used by AstroFries/BlackHole_STD:

        d^2 r / dlambda^2 = -1.5 * M * h^2 * r / |r|^5,
        h = |r x v| = const.

    The result is a camera-dependent lensed star field, not a true
    parallax-correct skybox.  Rays that fall below the horizon are dropped.
    """
    cam, right, up, forward = camera_basis()
    count = max(n * 2, n + 32)
    tan_half = math.tan(math.radians(CAMERA_FOV_DEG) * 0.5)
    uv = rng.uniform(-tan_half, tan_half, (count, 2))
    dirs = forward[None, :] + uv[:, 0, None] * right[None, :] + uv[:, 1, None] * up[None, :]
    dirs /= np.linalg.norm(dirs, axis=1)[:, None]

    pos = np.broadcast_to(cam, dirs.shape).copy()
    vel = dirs.copy()
    h = np.linalg.norm(np.cross(pos, vel), axis=1)
    captured = np.zeros(count, dtype=bool)
    escaped = np.zeros(count, dtype=bool)

    for _ in range(8000):
        r = np.linalg.norm(pos, axis=1)
        active = (~captured) & (~escaped) & (r < 175.0)
        if not np.any(active):
            break
        # Small steps near the photon sphere, long steps in the weak field.
        dt = np.where(r < 10.0, 0.006, 0.05)
        a1 = -1.5 * M * (h * h)[:, None] * pos / (r[:, None] ** 5)
        vel[active] += 0.5 * a1[active] * dt[active, None]
        pos[active] += vel[active] * dt[active, None]
        r = np.linalg.norm(pos, axis=1)
        a2 = -1.5 * M * (h * h)[:, None] * pos / (r[:, None] ** 5)
        vel[active] += 0.5 * a2[active] * dt[active, None]
        captured |= r < 1.0
        escaped |= (r > 150.0) & (np.einsum("ij,ij->i", pos, vel) > 0.0)

    valid = escaped & ~captured
    directions = vel[valid] / np.linalg.norm(vel[valid], axis=1)[:, None]
    if len(directions) > n:
        directions = directions[:n]
    if len(directions) < n:
        # A few rays can be captured or numerically lost; fill with an
        # unlensed sphere so the scene always has a usable star count.
        extra = n - len(directions)
        z = rng.uniform(-1.0, 1.0, extra)
        a = rng.uniform(0.0, 2.0 * math.pi, extra)
        rad = np.sqrt(np.maximum(0.0, 1.0 - z * z))
        fallback = np.column_stack((rad * np.cos(a), z, rad * np.sin(a)))
        directions = np.vstack((directions, fallback))
    distance = rng.uniform(180.0, 340.0, len(directions))
    print(f"[blackhole] lensed stars: {len(directions)} rays escaped")
    return cam[None, :] + directions * distance[:, None]


def add_star_field(rng: np.random.Generator, material: bpy.types.Material) -> None:
    pos = lensed_star_positions(STARS, rng)
    n = len(pos)
    colors = np.column_stack(
        (
            rng.uniform(0.55, 1.0, n),
            rng.uniform(0.65, 1.0, n),
            rng.uniform(0.85, 1.0, n),
            np.ones(n),
        )
    )
    # A few warm stars keep the field from reading as pure blue noise.
    warm = rng.random(n) < 0.18
    colors[warm, 0] = 1.0
    colors[warm, 1] = rng.uniform(0.58, 0.82, warm.sum())
    colors[warm, 2] = rng.uniform(0.30, 0.58, warm.sum())

    size = 0.55
    offsets = np.array(
        [
            [0.0, size, 0.0],
            [size, 0.0, 0.0],
            [0.0, 0.0, size],
            [-size, 0.0, 0.0],
            [0.0, 0.0, -size],
            [0.0, -size, 0.0],
        ]
    )
    verts = (pos[:, None, :] + offsets[None, :, :]).reshape(-1, 3)
    faces = []
    for p in range(n):
        b = p * 6
        faces.extend(
            [
                (b + 0, b + 2, b + 1),
                (b + 0, b + 3, b + 2),
                (b + 0, b + 4, b + 3),
                (b + 0, b + 1, b + 4),
                (b + 5, b + 1, b + 2),
                (b + 5, b + 2, b + 3),
                (b + 5, b + 3, b + 4),
                (b + 5, b + 4, b + 1),
            ]
        )
    mesh = bpy.data.meshes.new("StarField")
    mesh.from_pydata(verts.tolist(), [], faces)
    mesh.update()
    obj = bpy.data.objects.new("StarField", mesh)
    bpy.context.scene.collection.objects.link(obj)
    obj.data.materials.append(material)
    attr = mesh.color_attributes.new(name="Col", type="FLOAT_COLOR", domain="POINT")
    flat = np.repeat(colors, 6, axis=0).astype(np.float32)
    attr.data.foreach_set("color", flat.reshape(-1))


def add_world() -> None:
    scene = bpy.context.scene
    world = bpy.data.worlds.new("CosmicBackground")
    scene.world = world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    nodes.clear()

    output = nodes.new("ShaderNodeOutputWorld")
    background = nodes.new("ShaderNodeBackground")
    tex = nodes.new("ShaderNodeTexCoord")
    noise = nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 1.6
    noise.inputs["Detail"].default_value = 7.0
    noise.inputs["Roughness"].default_value = 0.62
    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = 0.25
    ramp.color_ramp.elements[0].color = (0.0015, 0.002, 0.010, 1.0)
    ramp.color_ramp.elements[1].position = 0.82
    ramp.color_ramp.elements[1].color = (0.055, 0.015, 0.095, 1.0)
    links.new(tex.outputs["Generated"], noise.inputs["Vector"])
    links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"], background.inputs["Color"])
    background.inputs["Strength"].default_value = 0.32
    links.new(background.outputs["Background"], output.inputs["Surface"])


def add_black_hole() -> None:
    bpy.ops.mesh.primitive_uv_sphere_add(
        segments=64, ring_count=32, radius=1.0, location=(0.0, 0.0, 0.0)
    )
    obj = bpy.context.object
    obj.name = "EventHorizon"
    mat = bpy.data.materials.new("EventHorizonMaterial")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    bsdf = nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (0.0005, 0.0005, 0.0007, 1.0)
        if "Roughness" in bsdf.inputs:
            bsdf.inputs["Roughness"].default_value = 0.85
        if "Specular IOR Level" in bsdf.inputs:
            bsdf.inputs["Specular IOR Level"].default_value = 0.08
    obj.data.materials.append(mat)


def add_photon_ring() -> bpy.types.Object:
    # A camera-facing emissive ring is an explicit visual approximation of the
    # lensed photon sphere.  It is not a physical solid.
    bpy.ops.mesh.primitive_torus_add(
        major_radius=CRITICAL_IMPACT_PARAMETER,
        minor_radius=0.012,
        major_segments=160,
        minor_segments=8,
        location=(0.0, 0.0, 0.0),
        rotation=(0.0, 0.0, 0.0),
    )
    obj = bpy.context.object
    obj.name = "PhotonRing"
    # Make the ring face the camera instead of lying in the XY plane.
    obj.rotation_euler = Vector(CAMERA_LOCATION).to_track_quat("Z", "Y").to_euler()
    mat = bpy.data.materials.new("PhotonRingMaterial")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    out = nodes.new("ShaderNodeOutputMaterial")
    em = nodes.new("ShaderNodeEmission")
    em.inputs["Color"].default_value = (0.62, 0.82, 1.0, 1.0)
    em.inputs["Strength"].default_value = 2.6
    links.new(em.outputs["Emission"], out.inputs["Surface"])
    obj.data.materials.append(mat)
    return obj


def add_gas_disk() -> None:
    """A smooth, noisy emission annulus underneath the simulated particles."""
    radial_steps = 72
    angular_steps = 192
    vertical_steps = 3
    half_height = 0.16
    verts = []
    for i in range(radial_steps + 1):
        t = i / radial_steps
        radius = DISK_INNER + (DISK_OUTER - DISK_INNER) * t
        for k in range(vertical_steps + 1):
            height = -half_height + 2.0 * half_height * k / vertical_steps
            for j in range(angular_steps):
                angle = 2.0 * math.pi * j / angular_steps
                verts.append(
                    (radius * math.cos(angle), height, radius * math.sin(angle))
                )
    faces = []
    for i in range(radial_steps):
        for k in range(vertical_steps):
            for j in range(angular_steps):
                jn = (j + 1) % angular_steps
                def idx(ii, kk, jj):
                    return (ii * (vertical_steps + 1) + kk) * angular_steps + jj
                a = idx(i, k, j)
                b = idx(i, k, jn)
                c = idx(i + 1, k, jn)
                d = idx(i + 1, k, j)
                faces.append((a, b, c, d))
                a = idx(i, k + 1, j)
                b = idx(i, k + 1, jn)
                c = idx(i + 1, k + 1, jn)
                d = idx(i + 1, k + 1, j)
                faces.append((a, b, c, d))
    mesh = bpy.data.meshes.new("GasDisk")
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    obj = bpy.data.objects.new("GasDisk", mesh)
    bpy.context.scene.collection.objects.link(obj)

    mat = bpy.data.materials.new("GasDiskMaterial")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    out = nodes.new("ShaderNodeOutputMaterial")
    tex = nodes.new("ShaderNodeTexCoord")
    noise = nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 2.2
    noise.inputs["Detail"].default_value = 9.0
    noise.inputs["Roughness"].default_value = 0.7
    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = 0.28
    ramp.color_ramp.elements[0].color = (0.02, 0.004, 0.001, 1.0)
    ramp.color_ramp.elements[1].position = 0.78
    ramp.color_ramp.elements[1].color = (1.0, 0.34, 0.045, 1.0)
    em = nodes.new("ShaderNodeEmission")
    em.inputs["Strength"].default_value = 0.55
    links.new(tex.outputs["Generated"], noise.inputs["Vector"])
    links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"], em.inputs["Color"])
    links.new(em.outputs["Emission"], out.inputs["Surface"])
    obj.data.materials.append(mat)
    obj.rotation_mode = "XYZ"
    obj.rotation_euler[1] = 0.0
    obj.keyframe_insert("rotation_euler", index=1, frame=1)
    obj.rotation_euler[1] = 2.0 * math.pi
    obj.keyframe_insert("rotation_euler", index=1, frame=FRAMES)


def add_lensed_arcs() -> None:
    """Two screen-facing arcs approximating the secondary lensed disk images.

    These are geometry-level stand-ins, not a ray-traced null-geodesic image.
    """
    def arc(name: str, radius: float, a0: float, a1: float, color, strength: float):
        curve = bpy.data.curves.new(name, "CURVE")
        curve.dimensions = "3D"
        curve.resolution_u = 1
        curve.bevel_depth = 0.035
        curve.bevel_resolution = 2
        spline = curve.splines.new("POLY")
        count = 48
        spline.points.add(count - 1)
        for i in range(count):
            a = a0 + (a1 - a0) * i / (count - 1)
            spline.points[i].co = (
                radius * math.cos(a),
                radius * math.sin(a),
                0.0,
                1.0,
            )
        obj = bpy.data.objects.new(name, curve)
        bpy.context.scene.collection.objects.link(obj)
        mat = bpy.data.materials.new(name + "Material")
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        nodes.clear()
        out = nodes.new("ShaderNodeOutputMaterial")
        em = nodes.new("ShaderNodeEmission")
        em.inputs["Color"].default_value = color
        em.inputs["Strength"].default_value = strength
        links.new(em.outputs["Emission"], out.inputs["Surface"])
        obj.data.materials.append(mat)

    arc("LensedUpperArc", 2.95, math.radians(24), math.radians(156),
        (1.0, 0.82, 0.52, 1.0), 2.2)
    arc("LensedLowerArc", 2.55, math.radians(204), math.radians(336),
        (0.55, 0.66, 1.0, 1.0), 0.9)


def add_refraction_lens(shells: int = 64, outer_radius: float = 15.0) -> None:
    """Nested refractive shells, following the Alaskan FX Blender tutorial.

    This is not a geodesic solver.  Each shell bends light a little; the nested
    spheres fake the cumulative bending of a gravitational lens.  It is a
    visual technique, deliberately separate from the ray-traced saddle-point
    formula in the README.
    """
    mat = bpy.data.materials.new("LensRefraction")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    out = nodes.new("ShaderNodeOutputMaterial")
    refraction = nodes.new("ShaderNodeBsdfRefraction")
    refraction.inputs["Roughness"].default_value = 0.0
    refraction.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    tex = nodes.new("ShaderNodeTexCoord")
    length = nodes.new("ShaderNodeVectorMath")
    length.operation = "LENGTH"
    sub = nodes.new("ShaderNodeMath")
    sub.operation = "SUBTRACT"
    sub.inputs[1].default_value = 1.0
    mul = nodes.new("ShaderNodeMath")
    mul.operation = "MULTIPLY"
    mul.inputs[1].default_value = 0.35
    add = nodes.new("ShaderNodeMath")
    add.operation = "ADD"
    add.inputs[1].default_value = 1.0
    links.new(tex.outputs["Object"], length.inputs[0])
    links.new(length.outputs["Value"], sub.inputs[0])
    links.new(sub.outputs["Value"], mul.inputs[0])
    links.new(mul.outputs["Value"], add.inputs[0])
    links.new(add.outputs["Value"], refraction.inputs["IOR"])
    links.new(refraction.outputs["BSDF"], out.inputs["Surface"])

    for i in range(shells):
        t = i / max(shells - 1, 1)
        radius = 1.02 + (outer_radius - 1.02) * (t ** 2.0)
        bpy.ops.mesh.primitive_uv_sphere_add(
            segments=32, ring_count=16, radius=radius, location=(0.0, 0.0, 0.0)
        )
        obj = bpy.context.object
        obj.name = f"LensShell{i:02d}"
        obj.data.materials.append(mat)
        bpy.ops.object.shade_smooth()

    scene = bpy.context.scene
    scene.cycles.transmission_bounces = 128
    scene.cycles.max_bounces = 128


def add_camera() -> None:
    scene = bpy.context.scene
    cam_data = bpy.data.cameras.new("BlackHoleCamera")
    cam = bpy.data.objects.new("BlackHoleCamera", cam_data)
    scene.collection.objects.link(cam)
    cam.location = CAMERA_LOCATION
    direction = Vector((0.0, 0.0, 0.0)) - cam.location
    cam.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    cam_data.lens = 35.0
    cam_data.sensor_width = 36.0
    cam_data.angle = math.radians(50.0)
    scene.camera = cam


def setup_render(args: argparse.Namespace) -> None:
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = args.samples
    scene.cycles.use_denoising = True
    scene.render.resolution_x = args.width
    scene.render.resolution_y = args.width
    scene.render.resolution_percentage = 100
    scene.render.fps = FPS
    scene.frame_start = 1
    scene.frame_end = args.frames
    scene.render.image_settings.file_format = "PNG"
    scene.view_settings.view_transform = "AgX"

    # A little bloom makes 3200 tiny emitters read as a continuous hot disk.
    # Blender 5.x moved compositor trees to a node-group assigned to the scene;
    # the group output replaces the old CompositorNodeComposite.
    scene.use_nodes = True
    tree = bpy.data.node_groups.new("BlackHoleCompositor", "CompositorNodeTree")
    tree.interface.new_socket("Image", in_out="OUTPUT", socket_type="NodeSocketColor")
    scene.compositing_node_group = tree
    rlayers = tree.nodes.new("CompositorNodeRLayers")
    glare = tree.nodes.new("CompositorNodeGlare")
    if "Type" in glare.inputs:
        glare.inputs["Type"].default_value = "Fog Glow"
        glare.inputs["Quality"].default_value = "High"
        glare.inputs["Threshold"].default_value = 0.7
        glare.inputs["Size"].default_value = 0.7
    else:  # Blender < 5.2 compatibility
        glare.glare_type = "FOG_GLOW"
        glare.quality = "HIGH"
        glare.threshold = 0.7
        glare.size = 7
    output = tree.nodes.new("NodeGroupOutput")
    tree.links.new(rlayers.outputs["Image"], glare.inputs["Image"])
    tree.links.new(glare.outputs["Image"], output.inputs["Image"])


def main() -> None:
    args = parse_args()
    bpy.ops.wm.read_factory_settings(use_empty=True)

    scene = bpy.context.scene
    scene.render.fps = FPS
    scene.frame_end = args.frames

    rng = np.random.default_rng(SEED)
    pos0, vel0 = initial_state(PARTICLES, rng)
    radius = np.linalg.norm(pos0[:, (0, 2)], axis=1)
    colors = disk_colors(rng, PARTICLES, radius)
    print(
        f"[blackhole] integrating {PARTICLES} particles for {args.frames} frames, "
        f"vsr={'yes' if vsr is not None else 'no'}"
    )
    positions = simulate(
        pos0, vel0, args.frames, PHYSICS_DT_PER_FRAME, SUBSTEPS
    )

    add_world()
    add_black_hole()
    photon_ring = add_photon_ring()
    add_gas_disk()
    add_lensed_arcs()
    add_refraction_lens()

    disk_mat = emission_material("DiskEmission", 3.0)
    add_particle_mesh("AccretionDisk", positions, colors, size=0.070, material=disk_mat)

    star_mat = emission_material("StarEmission", 2.0)
    add_star_field(rng, star_mat)
    add_camera()
    setup_render(args)

    bpy.ops.wm.save_as_mainfile(filepath=args.out)
    print(f"[blackhole] saved {args.out}")

    if not args.no_render:
        scene.frame_set(min(max(args.render_frame, 1), args.frames))
        scene.render.filepath = args.render
        bpy.ops.render.render(write_still=True)
        print(f"[blackhole] rendered {args.render}")


if __name__ == "__main__":
    main()
