import argparse
import os
import sys
import time

import bpy
from mathutils import Vector

# ---------------------------------------------------------------------------
# Fast vs Legacy rendering
# ---------------------------------------------------------------------------
# The CLI can trigger two quality modes:
#   * "fast" (always executed) keeps the user-requested resolution, enables
#     adaptive sampling with ~128 Cycles samples, and uses a baseline hair
#     density multiplier of 1.0 for short render times.
#   * "legacy" (optional comparison pass) defaults to double resolution, uses
#     ~512 Cycles samples with adaptive sampling disabled, and boosts the hair
#     density multiplier to accentuate individual stitches. Renders take longer
#     but preserve the crisp detail of the older pipeline.
# Geometry generation is identical between modes; only sampling, resolution,
# and hair density differ so you can trade speed for quality per output.
# ---------------------------------------------------------------------------

# Locate the importer code file
_SCRIPT_DIR = os.path.dirname(__file__)
_LOCAL_ADDON_FILE = os.path.join(_SCRIPT_DIR, "ImporterScript", "__init__.py")
ADDON_FILE = (
    _LOCAL_ADDON_FILE
    if os.path.exists(_LOCAL_ADDON_FILE)
    else "/opt/blender/4.3/scripts/addons/ImporterScript/__init__.py"
)

with open(ADDON_FILE, "r", encoding="utf-8") as f:
    ADDON_CODE = compile(f.read(), ADDON_FILE, "exec")

exec(ADDON_CODE, globals(), globals())
print(f"[render_still] Using importer from {ADDON_FILE}")

if hasattr(bpy.types, "ImportEmbroideryData"):
    try:
        bpy.utils.unregister_class(bpy.types.ImportEmbroideryData)
    except Exception as exc:
        print(f"[render_still] Warning: failed to unregister prior ImportEmbroideryData: {exc}")

try:
    bpy.types.TOPBAR_MT_file_import.remove(menu_func_import)
except Exception:
    pass

if "register" in globals():
    register()
else:
    raise RuntimeError("ImporterScript register() not available after exec")

_original_create_material = None


def _cap_colors(colors, max_elements=32):
    if len(colors) <= max_elements:
        return colors
    if max_elements <= 1:
        return [colors[0]]
    max_elements = min(max_elements, max(1, len(colors)))
    max_elements = min(max_elements, max(1, len(colors)))
    capped = []
    last_index = len(colors) - 1
    for i in range(max_elements):
        position = (last_index * i) / (max_elements - 1)
        index = int(round(position))
        index = max(0, min(last_index, index))
        capped.append(colors[index])
    return capped


def create_material_patch():
    global _original_create_material
    original = globals().get("create_material")
    if original is None:
        return
    if getattr(original, "_is_patched", False):
        return

    _original_create_material = original

    def _patched_create_material():
        global thread_colors
        colors = globals().get("thread_colors", [])
        if colors:
            capped_colors = _cap_colors(colors, 32)
            if len(capped_colors) != len(colors):
                print(
                    f"[render_still] Capping thread colors from {len(colors)} to {len(capped_colors)} to meet color ramp limits."
                )
            thread_colors = capped_colors
        else:
            print("[render_still] No thread colors found when creating material; proceeding without capping.")
        try:
            return _original_create_material()
        finally:
            if colors:
                thread_colors = colors

    _patched_create_material._is_patched = True
    globals()["create_material"] = _patched_create_material


create_material_patch()


def ensure_importer_registered():
    if not hasattr(bpy.ops.import_scene, "embroidery"):
        exec(ADDON_CODE, globals(), globals())
        create_material_patch()


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Render a single PES design to a PNG image.")
    parser.add_argument("-i", "--input_pes", required=True, help="Input PES file name or absolute path.")
    parser.add_argument("-o", "--output_image", required=True, help="Output PNG file name or absolute path.")
    parser.add_argument("--resolution", type=int, default=1024, help="Square resolution for the render.")
    parser.add_argument(
        "--camera",
        default="TopView",
        help="Name of the camera to use (defaults to TopView from BlenderSetup.blend).",
    )
    parser.add_argument(
        "--thread_thickness",
        type=float,
        default=0.2,
        help="Thread thickness passed to the importer (defaults to 0.2).",
    )
    parser.add_argument(
        "--fast_samples",
        type=int,
        default=None,
        help="Optional Cycles sample override for the primary (fast) render.",
    )
    parser.add_argument(
        "--legacy_output",
        default=None,
        help="Optional second output PNG for a legacy high-quality render.",
    )
    parser.add_argument(
        "--legacy_resolution",
        type=int,
        default=None,
        help="Resolution to use for the legacy render (defaults to primary resolution * 2).",
    )
    parser.add_argument(
        "--legacy_samples",
        type=int,
        default=None,
        help="Optional Cycles sample override for the legacy render.",
    )
    parser.add_argument(
        "--blend_file",
        default=None,
        help="Optional .blend file to reload for each render when running persistently.",
    )
    return parser.parse_args(argv)


def resolve_path(base_dir, relative_or_abs, fallback_folder):
    if os.path.isabs(relative_or_abs):
        return os.path.abspath(relative_or_abs)
    candidate = os.path.join(base_dir, fallback_folder, relative_or_abs)
    return os.path.abspath(candidate)


def ensure_folder(path):
    folder = os.path.dirname(path)
    if folder and not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)


def convert_collection_to_mesh(collection):
    bpy.ops.object.select_all(action="DESELECT")
    for obj in collection.objects:
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        bpy.ops.object.convert(target="MESH")
        obj.select_set(False)
    bpy.context.view_layer.update()


def recenter_collection(collection):
    min_coord = Vector((float("inf"), float("inf"), float("inf")))
    max_coord = Vector((float("-inf"), float("-inf"), float("-inf")))

    for obj in collection.objects:
        for corner in obj.bound_box:
            world_corner = obj.matrix_world @ Vector(corner)
            min_coord.x = min(min_coord.x, world_corner.x)
            min_coord.y = min(min_coord.y, world_corner.y)
            min_coord.z = min(min_coord.z, world_corner.z)
            max_coord.x = max(max_coord.x, world_corner.x)
            max_coord.y = max(max_coord.y, world_corner.y)
            max_coord.z = max(max_coord.z, world_corner.z)

    center = (min_coord + max_coord) / 2

    for obj in collection.objects:
        obj.location = obj.location - center

    return min_coord, max_coord


def set_origin_to_bounds(collection):
    bpy.context.scene.cursor.location = (0.0, 0.0, 0.0)
    for obj in collection.objects:
        bpy.ops.object.select_all(action="DESELECT")
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        bpy.ops.object.origin_set(type="ORIGIN_CURSOR", center="BOUNDS")


def scale_collection(collection, min_coord, max_coord, desired_dimension=0.127):
    obj_width = max_coord.x - min_coord.x
    obj_length = max_coord.y - min_coord.y
    obj_height = max_coord.z - min_coord.z
    biggest_dimension = max(obj_width, obj_length, obj_height)

    if biggest_dimension <= 0:
        return

    scale_factor = desired_dimension / biggest_dimension
    for obj in collection.objects:
        obj.scale *= scale_factor
    bpy.context.view_layer.update()


def realign_to_table(collection):
    min_coord = Vector((float("inf"), float("inf"), float("inf")))
    for obj in collection.objects:
        for corner in obj.bound_box:
            world_corner = obj.matrix_world @ Vector(corner)
            min_coord.z = min(min_coord.z, world_corner.z)

    z_offset = -min_coord.z
    if abs(z_offset) > 1e-7:
        for obj in collection.objects:
            obj.location.z += z_offset
    bpy.context.view_layer.update()


def add_particle_hair(collection, density_scale=1.0):
    reference_width = 0.0670
    reference_length = 0.0670
    reference_area = reference_width * reference_length
    reference_hair_count = 300

    for obj in collection.objects:
        if not obj.data or len(obj.data.vertices) == 0:
            continue

        bpy.ops.object.select_all(action="DESELECT")
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.particle_system_add()

        psys = obj.particle_systems[-1]
        pset = psys.settings
        pset.type = "HAIR"
        pset.hair_length = 0.001
        pset.count = 500
        pset.use_advanced_hair = True
        pset.root_radius = 0.001
        pset.tip_radius = 0.001
        pset.rendered_child_count = 0
        pset.child_type = "NONE"

        bbox = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
        x_coords = [v.x for v in bbox]
        y_coords = [v.y for v in bbox]
        obj_width = max(x_coords) - min(x_coords)
        obj_length = max(y_coords) - min(y_coords)

        obj_area = obj_width * obj_length
        scale_factor = obj_area / reference_area if reference_area else 1.0
        hair_count = int(reference_hair_count * scale_factor * max(density_scale, 0.0))
        pset.count = max(hair_count, 1)


def adjust_thread_material():
    thread_mat = bpy.data.materials.get("ThreadMaterial")
    if not thread_mat or not thread_mat.use_nodes or not thread_mat.node_tree:
        return

    bsdf_node = None
    for node in thread_mat.node_tree.nodes:
        if node.type == "BSDF_PRINCIPLED":
            bsdf_node = node
            break

    if not bsdf_node:
        return

    bsdf_node.inputs["Metallic"].default_value = 0.5
    bsdf_node.inputs["Roughness"].default_value = 0.35
    bsdf_node.inputs[18].default_value = 1.0
    bsdf_node.inputs[19].default_value = 0.3
    bsdf_node.inputs[20].default_value = 1.5


def enable_gpu_rendering(scene, samples_override=None, adaptive_sampling=True):
    cycles_prefs = bpy.context.preferences.addons["cycles"].preferences

    gpu_backend = None
    for compute_backend in ("OPTIX", "CUDA"):
        try:
            cycles_prefs.compute_device_type = compute_backend
            cycles_prefs.get_devices()
        except Exception as exc:
            print(f"Skipping backend {compute_backend}: {exc}")
            continue

        gpu_found = False
        for device in cycles_prefs.devices:
            if device.type == compute_backend:
                device.use = True
                gpu_found = True
                print(f"Enabled GPU ({compute_backend}): {device.name}")

        if gpu_found:
            gpu_backend = compute_backend
            break

    if not gpu_backend:
        print("GPU not found, defaulting to CPU.")
        scene.cycles.device = "CPU"
    else:
        scene.cycles.device = "GPU"

    scene.render.engine = "CYCLES"
    scene.cycles.samples = samples_override if samples_override is not None else 128
    scene.cycles.use_adaptive_sampling = adaptive_sampling
    scene.cycles.adaptive_threshold = 0.01
    scene.cycles.max_bounces = 6
    scene.cycles.use_denoising = True
    if gpu_backend == "OPTIX":
        scene.cycles.denoiser = "OPTIX"
    else:
        scene.cycles.denoiser = "OPENIMAGEDENOISE"
    scene.render.use_persistent_data = True
    scene.cycles.use_persistent_data = True
    scene.cycles.max_subdivisions = 4
    scene.render.image_settings.compression = 0


def configure_compositor_for_white_background(scene):
    scene.use_nodes = True
    scene.render.use_compositing = True
    tree = scene.node_tree

    alpha_over_node = None
    for node in tree.nodes:
        if node.type == "ALPHAOVER":
            alpha_over_node = node
            break

    if not alpha_over_node:
        print("Alpha Over node not found; ensuring world background is white instead.")
        if scene.world and scene.world.use_nodes:
            background = scene.world.node_tree.nodes.get("Background")
            if background:
                background.inputs[0].default_value = (1, 1, 1, 1)
        return

    rgb_node = None
    for node in tree.nodes:
        if node.type == "RGB":
            rgb_node = node
            break

    if not rgb_node:
        rgb_node = tree.nodes.new(type="CompositorNodeRGB")
        rgb_node.location = alpha_over_node.location.x - 200, alpha_over_node.location.y
        tree.links.new(rgb_node.outputs[0], alpha_over_node.inputs[1])

    rgb_node.outputs[0].default_value = (1, 1, 1, 1)


def set_active_camera(scene, camera_name):
    camera_obj = bpy.data.objects.get(camera_name)
    if camera_obj and camera_obj.type == "CAMERA":
        scene.camera = camera_obj
    else:
        print(f"Camera {camera_name} not found. Using existing active camera.")


def hide_collections_for_clean_render():
    for collection_name in ("Environment_Table", "Environment_Anim"):
        collection = bpy.data.collections.get(collection_name)
        if collection:
            collection.hide_render = True


def render_single_image(
    input_pes_path,
    output_image_path,
    resolution=1024,
    camera="TopView",
    thread_thickness=0.2,
    blend_file=None,
    quality="fast",
    cycles_samples=None,
):
    metrics = {}
    total_start = time.perf_counter()

    if blend_file:
        reload_start = time.perf_counter()
        bpy.ops.wm.open_mainfile(filepath=blend_file)
        metrics["scene_reload"] = time.perf_counter() - reload_start
        ensure_importer_registered()

    ensure_folder(output_image_path)
    ensure_importer_registered()

    import_start = time.perf_counter()
    bpy.ops.import_scene.embroidery(
        filepath=input_pes_path,
        show_jump_wires=True,
        do_create_material=True,
        create_collection=True,
        line_depth="GEOMETRY_NODES",
        thread_thickness=thread_thickness,
    )
    metrics["import"] = time.perf_counter() - import_start

    collection_name = os.path.basename(input_pes_path)
    collection = bpy.data.collections.get(collection_name)
    if not collection or not collection.objects:
        raise RuntimeError(f"No collection found or empty collection for {collection_name}")

    prep_start = time.perf_counter()
    convert_collection_to_mesh(collection)
    min_coord, max_coord = recenter_collection(collection)
    set_origin_to_bounds(collection)
    scale_collection(collection, min_coord, max_coord)
    realign_to_table(collection)
    hair_density = 1.0 if quality == "fast" else 1.6
    add_particle_hair(collection, density_scale=hair_density)
    adjust_thread_material()
    metrics["prep"] = time.perf_counter() - prep_start

    configure_start = time.perf_counter()
    scene = bpy.context.scene
    desired_samples = cycles_samples
    if desired_samples is None:
        desired_samples = 128 if quality == "fast" else 512
    adaptive = quality == "fast"
    enable_gpu_rendering(scene, samples_override=desired_samples, adaptive_sampling=adaptive)
    configure_compositor_for_white_background(scene)
    hide_collections_for_clean_render()
    set_active_camera(scene, camera)
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100
    scene.render.filepath = output_image_path
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "8"
    bpy.context.scene.cycles.samples = desired_samples
    metrics["configure"] = time.perf_counter() - configure_start

    render_start = time.perf_counter()
    bpy.ops.render.render(write_still=True)
    metrics["render"] = time.perf_counter() - render_start

    metrics["total"] = time.perf_counter() - total_start
    return metrics


def main():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else sys.argv[1:]
    args = parse_args(argv)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_pes = resolve_path(script_dir, args.input_pes, "input_PES")
    output_image = resolve_path(script_dir, args.output_image, "output")

    print(f"Importing PES file from: {input_pes}")
    print(f"Image will be saved to: {output_image}")

    metrics = render_single_image(
        input_pes_path=input_pes,
        output_image_path=output_image,
        resolution=args.resolution,
        camera=args.camera,
        thread_thickness=args.thread_thickness,
        blend_file=args.blend_file,
        quality="fast",
        cycles_samples=args.fast_samples,
    )

    print(f"Render metrics: {metrics}")
    print(f"Saved image: {output_image}")

    if args.legacy_output:
        legacy_output = resolve_path(script_dir, args.legacy_output, "output")
        legacy_resolution = (
            args.legacy_resolution if args.legacy_resolution else max(args.resolution * 2, args.resolution)
        )
        legacy_blend = args.blend_file if args.blend_file else os.path.join(script_dir, "BlenderSetup.blend")
        print(f"Starting legacy render for comparison at resolution {legacy_resolution} -> {legacy_output}")
        legacy_metrics = render_single_image(
            input_pes_path=input_pes,
            output_image_path=legacy_output,
            resolution=legacy_resolution,
            camera=args.camera,
            thread_thickness=args.thread_thickness,
            blend_file=legacy_blend,
            quality="legacy",
            cycles_samples=args.legacy_samples,
        )
        print(f"Legacy render metrics: {legacy_metrics}")
        print(f"Legacy image saved: {legacy_output}")


if __name__ == "__main__":
    main()
