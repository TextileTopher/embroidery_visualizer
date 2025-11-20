import bpy
import sys
import os
import argparse
import math
import time
import shutil
import subprocess
from mathutils import Vector
from mathutils import Matrix

# Locate the importer code file
script_dir = os.path.dirname(__file__)
local_addon_file = os.path.join(script_dir, "ImporterScript", "__init__.py")
addon_file = (
    local_addon_file
    if os.path.exists(local_addon_file)
    else "/opt/blender/4.3/scripts/addons/ImporterScript/__init__.py"
)

embroidery_center = Vector((0.0, 0.0, 0.0))
embroidery_diagonal = 0.127

with open(addon_file, 'r', encoding='utf-8') as f:
    addon_code = compile(f.read(), addon_file, "exec")

exec(addon_code, globals(), globals())
print(f"[Embroidery Visualizer] Using importer from {addon_file}")

if hasattr(bpy.types, "ImportEmbroideryData"):
    try:
        bpy.utils.unregister_class(bpy.types.ImportEmbroideryData)
    except Exception as exc:
        print(f"[Embroidery Visualizer] Warning: failed to unregister prior ImportEmbroideryData: {exc}")

try:
    bpy.types.TOPBAR_MT_file_import.remove(menu_func_import)
except Exception:
    pass

if "register" in globals():
    register()
else:
    raise RuntimeError("ImporterScript register() not available after exec")

# Blender adds its own arguments before ours, so we find '--' and slice from there.
if "--" in sys.argv:
    argv = sys.argv[sys.argv.index("--") + 1:]
else:
    argv = sys.argv[1:]  # If no '--', take all but the script name

# Set up argparse
parser = argparse.ArgumentParser(description="Import and process a PES file in Blender.")
parser.add_argument("-i", "--input_pes", required=True, help="Input PES file path")
parser.add_argument("-o", "--output_image", required=True, help="Output image file path")
parser.add_argument("-v", "--output_video", help="Output video file path")
parser.add_argument(
    "--skip_video",
    action="store_true",
    help="Skip animation rendering even if an output_video path is provided.",
)
parser.add_argument(
    "--backgrounds",
    default="white",
    help="Comma-separated list of backgrounds to render (choices: white, black, side_white).",
)
parser.add_argument(
    "--resolutions",
    default="1024",
    help="Comma-separated list of square resolutions (e.g., '1024,2048').",
)
parser.add_argument(
    "--show_jump_wires",
    action="store_true",
    help="Render jump stitches between sections (slower).",
)
parser.add_argument(
    "--enable_hair",
    action="store_true",
    help="Enable thread fuzz particle hair.",
)
parser.add_argument(
    "--video_quality",
    choices=["fast", "high", "ultra"],
    default="high",
    help="Animation quality preset (fast, high, or ultra).",
)
parser.add_argument(
    "--video_framing",
    choices=["zoomed", "full"],
    default="full",
    help="Choose between the tight CameraAnim framing (zoomed) or the original full-scene view.",
)

args = parser.parse_args(argv)
script_dir = os.path.dirname(__file__)

change_log_entries = []
requested_ultra_video = bool(
    args.output_video and args.video_quality and args.video_quality.lower() == "ultra"
)


def log_video_change(message: str):
    """Capture notable adjustments for diagnostics."""
    entry = f"{message}"
    change_log_entries.append(entry)
    print(f"[Embroidery Visualizer] {message}", flush=True)


backgrounds = [bg.strip() for bg in args.backgrounds.split(",") if bg.strip()]
if not backgrounds:
    backgrounds = ["white"]
resolution_tokens = [res.strip() for res in args.resolutions.split(",") if res.strip()]
if not resolution_tokens:
    resolution_tokens = ["1024"]
resolutions = []
for token in resolution_tokens:
    try:
        resolutions.append(str(int(token)))
    except ValueError:
        print(f"[Embroidery Visualizer] Ignoring invalid resolution token '{token}'.")
if not resolutions:
    resolutions = ["1024"]
##################################removed these two lines of code for new batch script
if os.path.isabs(args.input_pes):
    input_pes = os.path.abspath(args.input_pes)
else:
    candidate = os.path.abspath(os.path.join(script_dir, args.input_pes))
    if os.path.exists(candidate):
        input_pes = candidate
    else:
        pes_dir = os.path.join(script_dir, "input_PES")
        input_pes = os.path.abspath(os.path.join(pes_dir, args.input_pes))

if os.path.isabs(args.output_image):
    output_image = os.path.abspath(args.output_image)
else:
    output_image = os.path.abspath(os.path.join(script_dir, "output", args.output_image))
os.makedirs(os.path.dirname(output_image), exist_ok=True)

if args.output_video:
    if os.path.isabs(args.output_video):
        output_video = os.path.abspath(args.output_video)
    else:
        output_video = os.path.abspath(os.path.join(script_dir, "output", args.output_video))
    os.makedirs(os.path.dirname(output_video), exist_ok=True)
else:
    output_video = None

print("Attempting to import:", input_pes, flush=True)
print("Absolute path:", os.path.abspath(input_pes), flush=True)
print("File exists?", os.path.exists(input_pes), flush=True)
bpy.context.scene.render.use_lock_interface = False
import_start = time.perf_counter()
print("[Embroidery Visualizer] Importing PES via importer operator...", flush=True)
# Now proceed as before
bpy.ops.import_scene.embroidery(
    filepath=input_pes,
    show_jump_wires=args.show_jump_wires,
    do_create_material=True,
    create_collection=True,  # Ensure collection is created so we can center objects
    line_depth='GEOMETRY_NODES',
    thread_thickness=0.2,
)
print(
    f"[Embroidery Visualizer] Import finished in {time.perf_counter() - import_start:.2f}s",
    flush=True,
)

#########################################################################################################

# After import, assume the collection is named after the PES file:
collection_name = os.path.basename(input_pes)
collection = bpy.data.collections.get(collection_name)

if collection and len(collection.objects) > 0:
    conversion_start = time.perf_counter()
    print(
        f"[Embroidery Visualizer] Converting {len(collection.objects)} objects to meshes...",
        flush=True,
    )
    bpy.ops.object.select_all(action='DESELECT')
    for obj in collection.objects:
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        bpy.ops.object.convert(target='MESH')
        obj.select_set(False)

    bpy.context.view_layer.update()
    print(
        f"[Embroidery Visualizer] Mesh conversion took {time.perf_counter() - conversion_start:.2f}s",
        flush=True,
    )

    # Calculate the combined bounding box center of all objects
    min_coord = Vector((float('inf'), float('inf'), float('inf')))
    max_coord = Vector((float('-inf'), float('-inf'), float('-inf')))

    bbox_start = time.perf_counter()
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

    # #Adjust the 3D cursor in the position of table top z axis
    # bpy.context.scene.cursor.location = (0, 0, table_max_vert)

    # Center object in the world 0,0,0
    print("Calculated center:", center)
    for obj in collection.objects:
        old_loc = obj.location.copy()
        obj.location = obj.location - center
        print(f"Moved {obj.name} from {old_loc} to {obj.location}")

    #Center Origin for each object
    for obj in collection.objects:
        bpy.ops.object.select_all(action='DESELECT')
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        bpy.ops.object.origin_set(type='ORIGIN_CURSOR', center='BOUNDS')

    # Define and store the biggest dimensions of the imported object
    obj_width = (max_coord.x - min_coord.x)
    obj_length = (max_coord.y - min_coord.y)
    obj_height = (max_coord.z - min_coord.z)
    biggest_dimension = max(obj_width, obj_length, obj_height)
    print("the biggest dimension is:", biggest_dimension)

    embroidery_center = center.copy()
    embroidery_diagonal = max(biggest_dimension, 0.01)

    # Define the reference dimension you want all models to conform to
    desired_dimension = 0.127

    # Compute the scale factor
    if biggest_dimension > 0:
        scale_factor = desired_dimension / biggest_dimension
        print(f"Scaling objects by a factor of {scale_factor} to fit into {desired_dimension}m")
    else:
        scale_factor = 1.0
        print("No scaling needed (biggest_dimension is zero or invalid)")

    # Apply the scale factor to all objects in the collection
    for obj in collection.objects:
        # Since objects are centered at the origin, scaling is straightforward
        obj.scale *= scale_factor

    bpy.context.view_layer.update()
    print(
        f"[Embroidery Visualizer] Bounding box + scaling completed in {time.perf_counter() - bbox_start:.2f}s",
        flush=True,
    )

    # Re-initialize min/max coords before the second calculation
    min_coord = Vector((float('inf'), float('inf'), float('inf')))
    max_coord = Vector((float('-inf'), float('-inf'), float('-inf')))

    for obj in collection.objects:
        for corner in obj.bound_box:
            world_corner = obj.matrix_world @ Vector(corner)
            min_coord.x = min(min_coord.x, world_corner.x)
            min_coord.y = min(min_coord.y, world_corner.y)
            min_coord.z = min(min_coord.z, world_corner.z)
            max_coord.x = max(max_coord.x, world_corner.x)
            max_coord.y = max(max_coord.y, world_corner.y)
            max_coord.z = max(max_coord.z, world_corner.z)

    # # Align the table's top to the collection's bottom
    z_offset = -min_coord.z  # If min_coord.z is negative, we shift up; if positive, we shift down
    if abs(z_offset) > 1e-7:  # Just a tiny tolerance check
        for obj in collection.objects:
            obj.location.z += z_offset

    bpy.context.view_layer.update()

    # bpy.context.view_layer.update()

else:
    print(f"No collection found or empty collection for {collection_name}")

# Reference area and hair density
reference_width = 0.0670
reference_length = 0.0670
reference_area = reference_width * reference_length
reference_hair_count = 300

hair_required = args.enable_hair or requested_ultra_video
if requested_ultra_video and not args.enable_hair:
    print(
        "[Embroidery Visualizer] Enabling thread fuzz automatically for ultra-high-quality video.",
        flush=True,
    )

def set_pset_attr(pset, attr_name, value):
    if hasattr(pset, attr_name):
        setattr(pset, attr_name, value)

if hair_required:
    hair_start = time.perf_counter()
    # Add hair particle system to each object in the collection
    for obj in collection.objects:
        # Check if the object has geometry (vertices)
        if not obj.data or len(obj.data.vertices) == 0:
            print(f"Skipping object {obj.name}: No geometry found.")
            continue

        # Select and make the object active
        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj

        # Add a particle system
        bpy.ops.object.particle_system_add()

        # Access the particle system and settings
        psys = obj.particle_systems[-1]
        pset = psys.settings
        pset.type = 'HAIR'
        pset.hair_length = 0.001
        pset.count = 500  # Temporary, will calculate precisely below
        pset.use_advanced_hair = True
        pset.root_radius = 0.001
        pset.tip_radius = 0.001
        pset.rendered_child_count = 0
        pset.child_type = 'NONE'

        # Calculate object’s bounding box dimensions in local space
        # We assume object is roughly flat, so just width and length
        bbox = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
        x_coords = [v.x for v in bbox]
        y_coords = [v.y for v in bbox]
        obj_width = (max(x_coords) - min(x_coords))
        obj_length = (max(y_coords) - min(y_coords))

        # Calculate hair count proportionally
        obj_area = obj_width * obj_length
        scale_factor = obj_area / reference_area
        hair_count = int(reference_hair_count * scale_factor)
        if requested_ultra_video:
            hair_count = int(hair_count * 1.4)
        pset.count = max(hair_count, 200)

        if requested_ultra_video:
            pset.child_type = 'INTERPOLATED'
            pset.rendered_child_count = 20
            set_pset_attr(pset, "child_length", 0.85)
            set_pset_attr(pset, "child_radius", 0.6)
            set_pset_attr(pset, "clump_factor", 0.15)
            set_pset_attr(pset, "roughness_1", 0.02)
            set_pset_attr(pset, "roughness_endpoint", 0.01)
            set_pset_attr(pset, "roughness_2", 0.005)
            set_pset_attr(pset, "use_kink", True)
            set_pset_attr(pset, "kink", 'CURL')
            set_pset_attr(pset, "kink_amplitude", 0.00025)
            set_pset_attr(pset, "kink_frequency", 120.0)
        else:
            pset.child_type = 'NONE'
            pset.rendered_child_count = 0

    print(
        f"[Embroidery Visualizer] Particle hair assignment finished in {time.perf_counter() - hair_start:.2f}s",
        flush=True,
    )
else:
    print("[Embroidery Visualizer] Hair particle pass skipped (enable with --enable_hair).", flush=True)

thread_mat = bpy.data.materials.get("ThreadMaterial")
if thread_mat:
    print("Found ThreadMaterial, adjusting parameters.")
    # Ensure material is using nodes
    if thread_mat.use_nodes and thread_mat.node_tree:
        bsdf_node = None
        # Find the Principled BSDF node
        for node in thread_mat.node_tree.nodes:
            if node.type == 'BSDF_PRINCIPLED':
                bsdf_node = node
                break

        if bsdf_node:
            # Metallic
            bsdf_node.inputs["Metallic"].default_value = 0.5
            # Roughness
            bsdf_node.inputs["Roughness"].default_value = 0.35
            # Coat (Clearcoat)
            bsdf_node.inputs[18].default_value = 1.0
            bsdf_node.inputs[19].default_value = 0.3
            # IOR
            bsdf_node.inputs[20].default_value = 1.5

#########################################################################################################

# Now the PES embroidery is imported and centered into the BlenderSetup scene.
# Camera and lights are pre-set in BlenderSetup.blend.

# ------------------------------------------------------------
# Render Settings (example)
# ------------------------------------------------------------
# A helper function to set render resolution
def set_resolution(scene, width, height):
    scene.render.resolution_x = width
    scene.render.resolution_y = height

# A helper function to set the active camera
def set_active_camera(scene, camera_name):
    cam = bpy.data.objects.get(camera_name)
    if cam and cam.type == 'CAMERA':
        scene.camera = cam
    else:
        print(f"Camera {camera_name} not found or not a camera")

def ensure_ultra_fill_light(scene):
    """Create or update a warm area light for the ultra preset."""
    light_name = "UltraFillLight"
    light_obj = bpy.data.objects.get(light_name)
    if not light_obj:
        light_data = bpy.data.lights.new(light_name, type='AREA')
        light_obj = bpy.data.objects.new(light_name, light_data)
        scene.collection.objects.link(light_obj)
    light_data = light_obj.data
    light_data.energy = 1500
    light_data.color = (1.0, 0.95, 0.85)
    light_data.shape = 'RECTANGLE'
    light_data.size = 1.2
    light_data.size_y = 0.8
    light_obj.location = (0.8, -0.8, 0.6)
    light_obj.rotation_euler = (math.radians(70), 0.0, math.radians(40))
    return light_obj

# A helper function to render and save an image (composite output)
def render_image(scene, filepath):
    scene.render.filepath = filepath
    start = time.perf_counter()
    bpy.ops.render.render(write_still=True)
    print(
        f"[Embroidery Visualizer] Rendered {filepath} in {time.perf_counter() - start:.2f}s",
        flush=True,
    )


def encode_ultra_video_from_frames(frame_dir, output_path, fps, start_frame):
    """Transcode a 16-bit PNG sequence into the requested MP4."""
    ffmpeg_binary = shutil.which("ffmpeg")
    if not ffmpeg_binary:
        raise RuntimeError("ffmpeg binary not found on PATH; cannot encode ultra video.")
    frame_pattern = os.path.join(frame_dir, "frame_%04d.png")
    cmd = [
        ffmpeg_binary,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-framerate",
        f"{fps:.6f}",
        "-start_number",
        str(start_frame),
        "-i",
        frame_pattern,
        "-c:v",
        "libx264",
        "-preset",
        "slow",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        output_path,
    ]
    subprocess.run(cmd, check=True)

# ------------------------------------------------------------
# Setup compositing for background
# ------------------------------------------------------------
#scene = bpy.context.scene
#scene.render.engine = 'CYCLES'
#scene.cycles.samples = 100
scene = bpy.context.scene
original_render_resolution = (
    int(scene.render.resolution_x),
    int(scene.render.resolution_y),
)
original_resolution_percentage = scene.render.resolution_percentage


def enable_gpu_rendering(scene, samples_override=512, adaptive_sampling=False):
    cycles_prefs = bpy.context.preferences.addons["cycles"].preferences

    gpu_backend = None
    for compute_backend in ("OPTIX", "CUDA"):
        try:
            cycles_prefs.compute_device_type = compute_backend
            cycles_prefs.get_devices()
        except Exception as exc:
            print(f"[Embroidery Visualizer] Skipping backend {compute_backend}: {exc}")
            continue

        gpu_found = False
        for device in cycles_prefs.devices:
            if device.type == compute_backend:
                device.use = True
                gpu_found = True
                print(f"[Embroidery Visualizer] Enabled GPU ({compute_backend}): {device.name}")

        if gpu_found:
            gpu_backend = compute_backend
            break

    if not gpu_backend:
        print("[Embroidery Visualizer] GPU not found, defaulting to CPU.")
        scene.cycles.device = "CPU"
    else:
        scene.cycles.device = "GPU"

    scene.render.engine = "CYCLES"
    scene.cycles.samples = samples_override
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
    return gpu_backend


gpu_backend = enable_gpu_rendering(scene, samples_override=512, adaptive_sampling=False)
cycles_prefs = bpy.context.preferences.addons["cycles"].preferences
print(
    "[Embroidery Visualizer] Cycles render device:",
    scene.cycles.device,
    "| backend:",
    gpu_backend or "CPU",
)
for device in cycles_prefs.devices:
    print(
        f"[Embroidery Visualizer] Device {device.name} "
        f"type={device.type} use={getattr(device, 'use', False)}"
    )

# Optimize render tiles for GPUs while retaining compatibility with CPU fallback
scene.cycles.tile_x = 256
scene.cycles.tile_y = 256
scene.cycles.tile_size = 4096  # Increased for modern GPUs

# Set output resolution and scaling
scene.render.resolution_percentage = 100  # Ensure full resolution is used

# Ensure compositing is used
scene.use_nodes = True
scene.render.use_compositing = True

view_settings = scene.view_settings
print(
    "[Embroidery Visualizer] View settings -> "
    f"Transform: {view_settings.view_transform}, "
    f"Look: {view_settings.look}, "
    f"Exposure: {view_settings.exposure:.2f}, "
    f"Gamma: {view_settings.gamma:.2f}"
)


######################################
tree = scene.node_tree

# Find the Alpha Over node
alpha_over_node = None
for node in tree.nodes:
    if node.type == 'ALPHAOVER':
        alpha_over_node = node
        break

if not alpha_over_node:
    raise RuntimeError("Alpha Over node not found in the compositing setup!")

original_background_socket = None
if alpha_over_node.inputs[1].links:
    original_background_socket = alpha_over_node.inputs[1].links[0].from_socket


def set_input_link(input_socket, output_socket):
    """Ensure the given input only receives data from the provided output."""
    for link in list(input_socket.links):
        tree.links.remove(link)
    tree.links.new(output_socket, input_socket)


# Find or create an RGB node for setting the background color
rgb_node = None
for node in tree.nodes:
    if node.type == 'RGB':
        rgb_node = node
        break

if not rgb_node:
    rgb_node = tree.nodes.new(type='CompositorNodeRGB')
    rgb_node.location = alpha_over_node.location.x - 200, alpha_over_node.location.y

# Canvas texture setup for off-white woven backdrop
canvas_texture_path = os.path.join(script_dir, "assets", "textures", "canvas_offwhite.png")
canvas_mix_node = None
canvas_texture_available = False

if os.path.exists(canvas_texture_path):
    try:
        canvas_image = bpy.data.images.load(canvas_texture_path, check_existing=True)
        canvas_texture_available = True
    except Exception as exc:
        print(f"[Embroidery Visualizer] Warning: failed to load canvas texture '{canvas_texture_path}': {exc}")
        canvas_image = None
else:
    canvas_image = None
    print(
        f"[Embroidery Visualizer] Canvas texture '{canvas_texture_path}' not found. "
        "Falling back to flat color background."
    )

if canvas_image:
    canvas_image_node = tree.nodes.get("CanvasTextureImage")
    if not canvas_image_node:
        canvas_image_node = tree.nodes.new(type='CompositorNodeImage')
        canvas_image_node.name = "CanvasTextureImage"
        canvas_image_node.label = "Canvas Texture"
        canvas_image_node.location = alpha_over_node.location.x - 600, alpha_over_node.location.y - 200
    canvas_image_node.image = canvas_image

    canvas_mix_node = tree.nodes.get("CanvasTextureMix")
    if not canvas_mix_node:
        canvas_mix_node = tree.nodes.new(type='CompositorNodeMixRGB')
        canvas_mix_node.name = "CanvasTextureMix"
        canvas_mix_node.label = "Canvas Blend"
        canvas_mix_node.blend_type = 'MULTIPLY'
        canvas_mix_node.location = alpha_over_node.location.x - 350, alpha_over_node.location.y - 100

    set_input_link(canvas_mix_node.inputs[1], rgb_node.outputs[0])
    set_input_link(canvas_mix_node.inputs[2], canvas_image_node.outputs[0])


def use_canvas_background():
    """Prefer the woven canvas mix if available, else default to the RGB node."""
    if canvas_mix_node:
        set_input_link(alpha_over_node.inputs[1], canvas_mix_node.outputs[0])
    else:
        set_input_link(alpha_over_node.inputs[1], rgb_node.outputs[0])


def use_solid_background():
    """Force a flat color background."""
    set_input_link(alpha_over_node.inputs[1], rgb_node.outputs[0])

def restore_original_background():
    """Reconnect the compositor to the pre-script configuration."""
    if original_background_socket:
        set_input_link(alpha_over_node.inputs[1], original_background_socket)
    else:
        set_input_link(alpha_over_node.inputs[1], rgb_node.outputs[0])

# Default link so the compositor has something before iteration begins
use_canvas_background()

# ------------------------------------------------------------
# Assume 'output_image' and 'output_video' come from outside (e.g., PowerShell arguments)
# ------------------------------------------------------------
# Example:
# output_image = "D:/renders/output.png"
# output_video = "D:/renders/output.mp4"

output_image_base = output_image.rsplit('.', 1)[0]  # Remove file extension if any
video_quality = args.video_quality.lower() if args.video_quality else "high"
video_framing = args.video_framing.lower() if args.video_framing else "full"
legacy_high_video = video_quality == "high"
ultra_high_video = video_quality == "ultra"

# Lighting experiment toggle (set False to immediately revert to the previous hidden-light setup).
ULTRA_LIGHTING_EXPERIMENT = True

# Define postfixes dynamically from requested combos
environment_table = bpy.data.collections.get("Environment_Table")

postfixes = [(bg, res) for bg in backgrounds for res in resolutions]

for bg, res in postfixes:
    if environment_table:
        environment_table.hide_render = True
    # Determine the camera based on bg
    if bg == "white":
        # White background from TopView
        set_active_camera(scene, "TopView")
        rgb_node.outputs[0].default_value = (0.96, 0.93, 0.87, 1.0)  # Warm off-white
        if canvas_mix_node:
            canvas_mix_node.inputs[0].default_value = 0.72
        use_canvas_background()
    elif bg == "black":
        # Black background from TopView
        set_active_camera(scene, "TopView")
        rgb_node.outputs[0].default_value = (0,0,0,1)  # Pure black background
        use_solid_background()
    elif bg == "side_white":
        # White background from SideView
        set_active_camera(scene, "SideView")
        rgb_node.outputs[0].default_value = (0.95, 0.92, 0.86, 1.0)
        if canvas_mix_node:
            canvas_mix_node.inputs[0].default_value = 0.8
        use_canvas_background()
        if environment_table:
            environment_table.hide_render = False
    else:
        print(f"[Embroidery Visualizer] Unknown background '{bg}', skipping.")
        continue

    try:
        width = int(res)
        height = int(res)
    except ValueError:
        print(f"[Embroidery Visualizer] Invalid resolution '{res}', skipping.")
        continue
    set_resolution(scene, width, height)

    unique_output_image = f"{output_image_base}_{bg}_{res}.png"
    print(f"Rendering: {unique_output_image}")
    render_image(scene, unique_output_image)


if output_video and not args.skip_video:
    print("[Embroidery Visualizer] Starting animation render...", flush=True)
    restore_original_background()
    if video_quality == "fast":
        VIDEO_SAMPLES = 32
        ADAPTIVE = False
    elif legacy_high_video:
        VIDEO_SAMPLES = 10
        ADAPTIVE = False
        log_video_change(
            "High-quality video render reverting to legacy settings "
            f"(samples={VIDEO_SAMPLES}, resolution={original_render_resolution[0]}x{original_render_resolution[1]})."
        )
        if video_framing == "zoomed":
            log_video_change("Legacy high-quality video path ignores --video_framing=zoomed.")
    elif ultra_high_video:
        VIDEO_SAMPLES = 256
        ADAPTIVE = True
        log_video_change(
            "Ultra-high-quality video render enabled (1.5x native resolution, adaptive sampling, thread hair)."
        )
        if video_framing == "zoomed":
            log_video_change("Ultra-high-quality video path ignores --video_framing=zoomed in favor of native camera framing.")
    else:
        VIDEO_SAMPLES = 256
        ADAPTIVE = True

    if legacy_high_video:
        video_resolution = original_render_resolution
    elif ultra_high_video:
        video_resolution = (
            min(int(original_render_resolution[0] * 1.5), 3840),
            min(int(original_render_resolution[1] * 1.5), 2160),
        )
    elif video_quality == "fast" and video_framing == "full":
        video_resolution = original_render_resolution
    elif video_framing == "zoomed":
        video_resolution = (1280, 720) if video_quality == "fast" else (1920, 1080)
    else:
        video_resolution = (1024, 1024) if video_quality == "fast" else (2048, 2048)

    scene.render.engine = 'CYCLES'
    scene.render.use_border = False
    scene.render.use_crop_to_border = False
    set_resolution(scene, *video_resolution)
    scene.render.resolution_percentage = (
        original_resolution_percentage if legacy_high_video else 100
    )
    scene.cycles.samples = VIDEO_SAMPLES
    scene.cycles.use_adaptive_sampling = ADAPTIVE
    scene.cycles.adaptive_threshold = 0.005 if ultra_high_video else (
        0.01 if ADAPTIVE else scene.cycles.adaptive_threshold
    )
    ultra_frame_dir = None
    if ultra_high_video:
        ultra_frame_dir = os.path.splitext(output_video)[0] + "_frames"
        os.makedirs(ultra_frame_dir, exist_ok=True)
        scene.render.filepath = os.path.join(ultra_frame_dir, "frame_")
        scene.render.use_file_extension = True
        scene.render.image_settings.file_format = 'PNG'
        scene.render.image_settings.color_mode = 'RGB'
        scene.render.image_settings.color_depth = '16'
        scene.render.image_settings.compression = 0
        log_video_change(f"Capturing 16-bit PNG frames before encoding (stored in {ultra_frame_dir}).")
    else:
        scene.render.filepath = output_video
        scene.render.image_settings.file_format = 'FFMPEG'
        scene.render.ffmpeg.format = 'MPEG4'
        scene.render.ffmpeg.codec = 'H264'
        scene.render.ffmpeg.constant_rate_factor = 'HIGH'
        scene.render.ffmpeg.ffmpeg_preset = 'GOOD'
        scene.render.ffmpeg.video_bitrate = 0
        scene.render.image_settings.color_mode = 'RGB'
        scene.render.image_settings.color_depth = '8'
    scene.frame_start = 0
    scene.frame_end = 160

    scene.render.use_motion_blur = False
    if ultra_high_video:
        log_video_change("Motion blur disabled to keep every frame tack-sharp.")
    if hasattr(scene.cycles, "motion_blur_position"):
        scene.cycles.motion_blur_position = 'START' if ultra_high_video else 'CENTER'
    if hasattr(scene.cycles, "motion_blur_shutter"):
        scene.cycles.motion_blur_shutter = 0.3 if ultra_high_video else 0.5

    if ultra_high_video:
        scene.cycles.sample_clamp_direct = 0.0
        scene.cycles.sample_clamp_indirect = 0.0
        scene.cycles.use_denoising = False
        first_layer = scene.view_layers[0] if scene.view_layers else None
        if first_layer and hasattr(first_layer, "cycles"):
            first_layer.cycles.use_denoising = False
        log_video_change("Disabled denoising and clamp thresholds so thread micro-detail survives the 256-sample pass.")

    camera_anim = bpy.data.objects.get("CameraAnim")
    if camera_anim and camera_anim.type == "CAMERA":
        camera_data = camera_anim.data
        if ultra_high_video:
            camera_data.dof.use_dof = False
            log_video_change("Depth of field disabled so ultra videos match the crisper high-quality focus.")
        elif not args.enable_hair:
            camera_data.dof.use_dof = False

    environment_table = bpy.data.collections.get("Environment_Table")
    lights_collection = bpy.data.collections.get("Lights")
    animation_collection = bpy.data.collections.get("Environment_Anim")
    if animation_collection:
        animation_collection.hide_render = False
    fill_light = bpy.data.objects.get("UltraFillLight")
    if ultra_high_video and ULTRA_LIGHTING_EXPERIMENT:
        log_video_change("Ultra lighting experiment ON: table, rim lights, and UltraFillLight are visible for richer highlights.")
        if environment_table:
            environment_table.hide_render = False
        if lights_collection:
            lights_collection.hide_render = False
        fill_light = ensure_ultra_fill_light(scene)
        fill_light.hide_render = False
    else:
        if ultra_high_video:
            log_video_change("Ultra video reuses legacy lighting (table and extra lights hidden).")
        if environment_table:
            environment_table.hide_render = True
        if lights_collection:
            lights_collection.hide_render = True
        if fill_light:
            fill_light.hide_render = True

    world_nodes = bpy.data.worlds.get("World")
    if world_nodes and world_nodes.node_tree:
        mapping_node = world_nodes.node_tree.nodes.get("Mapping")
        background_node = world_nodes.node_tree.nodes.get("Background")
        if mapping_node:
            mapping_node.inputs[2].default_value[2] = 1.09956
        if background_node:
            background_node.inputs[1].default_value = 1.1
    scene_nodes = bpy.data.scenes.get("Scene")
    if scene_nodes and scene_nodes.node_tree:
        mix_node = scene_nodes.node_tree.nodes.get("Mix")
        if mix_node:
            mix_node.inputs[0].default_value = 0

    set_active_camera(scene, "CameraAnim")
    bpy.ops.render.render(animation=True, write_still=True)
    if ultra_high_video and ultra_frame_dir:
        fps = scene.render.fps / scene.render.fps_base if scene.render.fps_base else scene.render.fps
        try:
            encode_ultra_video_from_frames(
                ultra_frame_dir,
                output_video,
                fps,
                scene.frame_start,
            )
            log_video_change(
                f"FFmpeg encoded ultra MP4 from 16-bit frames (stored at {ultra_frame_dir})."
            )
        except Exception as exc:
            log_video_change(f"FFmpeg encoding failed: {exc}")
            raise
else:
    print("[Embroidery Visualizer] Skipping animation render.", flush=True)

if change_log_entries:
    logs_dir = os.path.join(script_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    log_path = os.path.join(logs_dir, "high_quality_video_revert.log")
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(log_path, "a", encoding="utf-8") as log_file:
        log_file.write(f"[{timestamp}] input={os.path.basename(input_pes)} output_video={output_video or 'n/a'}\n")
        for entry in change_log_entries:
            log_file.write(f" - {entry}\n")
