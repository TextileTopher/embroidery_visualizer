import sys
import bpy
from mathutils import Vector

from bpy.types import Operator
from bpy_extras.io_utils import ImportHelper
from pyembroidery import read

# from pyembroidery import write_png, write_svg
from math import floor
from os import path

z_height = 0.0002
scale = 10000.0
section_lift = 0.00002
MAX_THREAD_COLORS = 32
DEFAULT_THREAD_COLOR = (0.8, 0.8, 0.8)

NO_COMMAND = -1
STITCH = 0
JUMP = 1
TRIM = 2
STOP = 3
END = 4
COLOR_CHANGE = 5
NEEDLE_SET = 9

show_jumpwires = True
thread_colors = []
thread_color_cap_applied = False


def srgb_channel_to_linear(channel: float) -> float:
    """Convert a normalized sRGB channel into scene-linear space."""
    channel = max(0.0, min(1.0, channel))
    if channel <= 0.04045:
        return channel / 12.92
    return ((channel + 0.055) / 1.055) ** 2.4


def srgb_bytes_to_linear(r: int, g: int, b: int):
    """Return a tuple of scene-linear floats from byte RGB inputs."""
    normalized = (r / 255.0, g / 255.0, b / 255.0)
    return tuple(srgb_channel_to_linear(channel) for channel in normalized)


def describe_thread(thread):
    """Best-effort description for debug logging."""
    for attr in ("description", "name", "catalog_number"):
        value = getattr(thread, attr, None)
        if value:
            return str(value)
    return ""


def log_thread_palette(entries, declared_count, capped):
    """Emit human-readable information about the current palette."""
    shown = len(entries)
    cap_text = "yes" if capped else "no"
    print(
        f"[Embroidery Importer] Thread palette preview "
        f"({shown}/{declared_count} colors shown; cap_applied={cap_text})"
    )
    for entry in entries:
        idx = entry["index"]
        srgb = entry["srgb_255"]
        linear = entry["linear"]
        name = entry["name"]
        prefix = f"  #{idx + 1:02d}"
        if name:
            prefix += f" ({name})"
        rounded_linear = tuple(round(channel, 4) for channel in linear)
        print(
            f"{prefix}: sRGB {srgb} -> linear {rounded_linear}"
        )
    if declared_count > shown:
        print(
            f"[Embroidery Importer] {declared_count - shown} additional palette "
            f"entries not listed (cap enforced)."
        )


def truncate(f, n):
    return floor(f * 10**n) / 10**n


def create_material():
    """Creates a material with a color ramp based on the thread colors
    Base name of the material is ThreadMaterial, which Blender will append
    with a number if a material with this name already exists"""

    global thread_color_cap_applied

    colors = thread_colors if thread_colors else [DEFAULT_THREAD_COLOR]
    color_count = len(colors)

    material = bpy.data.materials.new(name="ThreadMaterial")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links

    nodes.clear()  # Clear existing nodes

    # Nodes are created in the same order as how they are linked in the node editor
    # Add an Attribute node; we store the thread number in an attribute which this node retrieves
    attribute_node = nodes.new(type="ShaderNodeAttribute")
    attribute_node.attribute_type = "OBJECT"
    attribute_node.attribute_name = "thread_index"
    attribute_node.location = (-900, 0)

    # Add a Math node; we will use this to divide the thread number by the number of thread colors to
    # find its position in the color ramp
    math_node_divide = nodes.new(type="ShaderNodeMath")
    math_node_divide.operation = "DIVIDE"
    math_node_divide.location = (-700, 0)

    math_node_add = nodes.new(type="ShaderNodeMath")
    math_node_add.operation = "ADD"
    math_node_add.inputs[0].default_value = 0.01  # Set the multiplier value
    math_node_add.location = (-500, 0)

    # Add a Color ramp node; this has a color for each of our threads
    color_ramp_node = nodes.new(type="ShaderNodeValToRGB")
    color_ramp_node.location = (-300, 0)
    color_ramp = color_ramp_node.color_ramp
    color_ramp.interpolation = "CONSTANT"

    if not thread_color_cap_applied:
        for index, color in enumerate(colors):
            color_stop = color_ramp.elements.new(
                truncate(1.0 / max(color_count, 1) * index, 3)
            )
            color_stop.color = (color[0], color[1], color[2], 1.0)

        if len(color_ramp.elements) > 0:
            color_ramp.elements.remove(color_ramp.elements[0])
        if len(color_ramp.elements) > 0:
            color_ramp.elements.remove(color_ramp.elements[-1])
        math_node_divide.inputs[1].default_value = max(color_count, 1)
    else:
        while len(color_ramp.elements) > 1:
            color_ramp.elements.remove(color_ramp.elements[-1])
        color_ramp.elements[0].position = 0.0
        color_ramp.elements[0].color = (
            colors[0][0],
            colors[0][1],
            colors[0][2],
            1.0,
        )
        for index in range(1, color_count):
            position = truncate(index / max(color_count - 1, 1), 3)
            new_element = color_ramp.elements.new(position)
            new_element.color = (
                colors[index][0],
                colors[index][1],
                colors[index][2],
                1.0,
            )
        math_node_divide.inputs[1].default_value = max(color_count, 1)

    # Add a Principled BSDF node
    bsdf_node = nodes.new(type="ShaderNodeBsdfPrincipled")
    bsdf_node.location = (0, 0)

    # Add an Output node
    output_node = nodes.new(type="ShaderNodeOutputMaterial")
    output_node.location = (300, 0)

    # Connect the Attribute node to the Math node
    links.new(attribute_node.outputs["Fac"], math_node_divide.inputs[0])
    # Connect the Math node to the Color Ramp node
    links.new(math_node_divide.outputs["Value"], math_node_add.inputs[1])
    # Connect the Math node to the Color Ramp node
    links.new(math_node_add.outputs["Value"], color_ramp_node.inputs["Fac"])
    # Connect the Color Ramp node to the Base Color input of the Principled BSDF node
    links.new(color_ramp_node.outputs["Color"], bsdf_node.inputs["Base Color"])
    # Connect the Principled BSDF node to the Output node
    links.new(bsdf_node.outputs["BSDF"], output_node.inputs["Surface"])

    return material


def create_line_depth_geometry_nodes(filename, material):
    nodeName = f"{filename}_embroidery_GN"
    if nodeName in bpy.data.node_groups:
        return bpy.data.node_groups[nodeName]

    threadgeometrynodes = bpy.data.node_groups.new(
        type="GeometryNodeTree", name=nodeName
    )
    threadgeometrynodes.color_tag = "NONE"
    threadgeometrynodes.description = ""
    threadgeometrynodes.is_modifier = True

    # threadgeometrynodes interface
    # Socket Geometry
    geometry_socket = threadgeometrynodes.interface.new_socket(
        name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry"
    )
    geometry_socket.attribute_domain = "POINT"
    # Socket Geometry
    geometry_socket_1 = threadgeometrynodes.interface.new_socket(
        name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry"
    )
    geometry_socket_1.attribute_domain = "POINT"

    # initialize threadgeometrynodes nodes
    # node Group Input
    group_input = threadgeometrynodes.nodes.new("NodeGroupInput")
    group_input.name = "Group Input"

    # node Group Output
    group_output = threadgeometrynodes.nodes.new("NodeGroupOutput")
    group_output.name = "Group Output"
    group_output.is_active_output = True

    # node Curve to Mesh
    curve_to_mesh = threadgeometrynodes.nodes.new("GeometryNodeCurveToMesh")
    curve_to_mesh.name = "Curve to Mesh"
    # Fill Caps
    curve_to_mesh.inputs[2].default_value = False
    # node Curve Circle
    curve_circle = threadgeometrynodes.nodes.new("GeometryNodeCurvePrimitiveCircle")
    curve_circle.name = "Curve Circle"
    curve_circle.mode = "RADIUS"
    # Resolution
    curve_circle.inputs[0].default_value = 4
    # Radius
    curve_circle.inputs[4].default_value = 0.0002

    # node Set Material
    set_material = threadgeometrynodes.nodes.new("GeometryNodeSetMaterial")
    set_material.name = "Set Material"
    # Selection
    set_material.inputs[1].default_value = True
    if material.name in bpy.data.materials:
        set_material.inputs[2].default_value = bpy.data.materials[material.name]
    # Set locations
    group_input.location = (-360.0, 80.0)
    group_output.location = (220.0, 80.0)
    curve_to_mesh.location = (-140.0, 80.0)
    curve_circle.location = (-360.0, -20.0)
    set_material.location = (40.0, 80.0)

    # Set dimensions
    group_input.width, group_input.height = 140.0, 100.0
    group_output.width, group_output.height = 140.0, 100.0
    curve_to_mesh.width, curve_to_mesh.height = 140.0, 100.0
    curve_circle.width, curve_circle.height = 140.0, 100.0
    set_material.width, set_material.height = 140.0, 100.0
    # initialize threadgeometrynodes links
    # group_input.Geometry -> curve_to_mesh.Curve
    threadgeometrynodes.links.new(group_input.outputs[0], curve_to_mesh.inputs[0])
    # curve_circle.Curve -> curve_to_mesh.Profile Curve
    threadgeometrynodes.links.new(curve_circle.outputs[0], curve_to_mesh.inputs[1])
    # set_material.Geometry -> group_output.Geometry
    threadgeometrynodes.links.new(set_material.outputs[0], group_output.inputs[0])
    # curve_to_mesh.Mesh -> set_material.Geometry
    threadgeometrynodes.links.new(curve_to_mesh.outputs[0], set_material.inputs[0])
    return threadgeometrynodes


def draw_stitch(curve_data, x1, y1, x2, y2):
    """Draw a single stitch"""
    spline = curve_data.splines.new("NURBS")
    spline.points.add(4)
    spline.points[0].co = (x1, y1, 0, 1)
    spline.points[1].co = (x1, y1, z_height, 1)
    spline.points[2].co = ((x2 + x1) / 2, (y2 + y1) / 2, z_height, 1)
    spline.points[3].co = (x2, y2, z_height, 1)
    spline.points[4].co = (x2, y2, 0, 1)
    spline.use_endpoint_u = True  # do this AFTER setting the points


def parse_embroidery_data(
    context,
    filepath,
    show_jumpwires,
    do_create_material,
    line_depth,
    thread_thickness,
    create_collection,
):

    filename = ""
    report_type = "INFO"
    report_message = ""
    error_message = ""

    try:
        filename = path.basename(filepath)
        pattern = read(filepath)
    except Exception as e:
        report_message = "Error reading file"
        report_type = "ERROR"
        return report_message, report_type
    global thread_colors
    global thread_color_cap_applied

    declared_thread_count = len(pattern.threadlist)
    thread_color_cap_applied = False
    if declared_thread_count > MAX_THREAD_COLORS:
        thread_color_cap_applied = True
        print(
            f"[Embroidery Importer] Warning: File declares {declared_thread_count} thread colors; "
            f"limiting to first {MAX_THREAD_COLORS} for Blender material support."
        )

    if do_create_material:
        palette_entries = []
        for idx, thread in enumerate(pattern.threadlist):
            srgb_bytes = (
                int(thread.get_red()),
                int(thread.get_green()),
                int(thread.get_blue()),
            )
            linear = srgb_bytes_to_linear(*srgb_bytes)
            palette_entries.append(
                {
                    "index": idx,
                    "srgb_255": srgb_bytes,
                    "linear": linear,
                    "name": describe_thread(thread),
                }
            )

        if not palette_entries:
            thread_colors = [DEFAULT_THREAD_COLOR]
        else:
            truncated_entries = palette_entries[:MAX_THREAD_COLORS]
            thread_colors = [entry["linear"] for entry in truncated_entries]
            log_thread_palette(truncated_entries, declared_thread_count, thread_color_cap_applied)

    thread_index = 0  # start at the first thread
    sections = []  # list of sections, each section is a list of stitches

    def clamp_thread_index(index: int) -> int:
        if MAX_THREAD_COLORS <= 0:
            return index
        if do_create_material:
            if not thread_colors:
                return 0
            return min(index, len(thread_colors) - 1)
        return min(index, MAX_THREAD_COLORS - 1)

    section = {"thread_index": clamp_thread_index(thread_index), "stitches": [], "is_jump": False}

    for stitch in pattern.stitches:
        x = float(stitch[0]) / scale
        y = -float(stitch[1]) / scale
        c = int(stitch[2])

        if c == STITCH:  # stitch and jump both draw a thread
            section["stitches"].append([x, y])

        elif c == JUMP:
            if show_jumpwires:
                section["stitches"].append([x, y])
            else:
                # Skip the jump wire
                if section["stitches"]:
                    # if our last section contained any stitches,
                    # close the section and start a new one
                    sections.append(section)
                    section = {
                        "thread_index": clamp_thread_index(thread_index),
                        "stitches": [],
                    }

        elif c == COLOR_CHANGE:  # color change, move to the next thread
            sections.append(section)  # end our previous section
            thread_index += 1
            section = {
                "thread_index": clamp_thread_index(thread_index),
                "stitches": [],
            }

        elif (
            c == TRIM
        ):  # trim moves to the next section without a line between the old and new position
            sections.append(section)  # end our previous section
            section = {
                "thread_index": clamp_thread_index(thread_index),
                "stitches": [],
            }

        elif c == END:  # end of a section?
            sections.append(section)
            section = {
                "thread_index": clamp_thread_index(thread_index),
                "stitches": [],
            }

        else:  # unhandled/unknown commands
            print("[Embroidery Importer] Unknown command: ", c)
            sections.append(section)  # end our previous section
            section = {
                "thread_index": clamp_thread_index(thread_index),
                "stitches": [],
            }
            section["stitches"].append([x, y])

    if do_create_material:
        material = create_material()  # create our material

    # collection = None
    if create_collection:
        collection = bpy.data.collections.new(filename)
        bpy.context.scene.collection.children.link(collection)
        bpy.context.view_layer.active_layer_collection = (
            bpy.context.view_layer.layer_collection.children[collection.name]
        )

    for section_index, section in enumerate(sections):
        bpy.ops.curve.primitive_nurbs_path_add()
        curve_obj = bpy.context.object
        curve_obj.location.z = section_lift * section_index
        curve_obj["thread_index"] = section["thread_index"]

        if do_create_material and line_depth != "GEOMETRY_NODES":
            curve_obj.data.materials.append(material)

        curve_data = curve_obj.data
        curve_data.use_path = False
        curve_data.splines.clear()

        if line_depth == "BEVEL":
            curve_data.use_fill_caps = True
            curve_data.bevel_depth = thread_thickness
            curve_data.bevel_resolution = 4
        elif line_depth == "GEOMETRY_NODES":
            GN = create_line_depth_geometry_nodes(filename, material)
            curve_obj.modifiers.new("Geometry Nodes", "NODES")
            curve_obj.modifiers["Geometry Nodes"].node_group = GN

        for stitch_index, stitch in enumerate(section["stitches"]):
            if stitch_index == 0:
                continue
            draw_stitch(
                curve_data,
                section["stitches"][stitch_index - 1][0],
                section["stitches"][stitch_index - 1][1],
                section["stitches"][stitch_index][0],
                section["stitches"][stitch_index][1],
            )

        # If we created a collection, move the new curve into it
        if collection:
            # Unlink from current and link to our new collection
            if curve_obj.name in bpy.context.scene.collection.objects:
                bpy.context.scene.collection.objects.unlink(curve_obj)
            if curve_obj.name not in collection.objects:
                collection.objects.link(curve_obj)

        curve_obj.data = curve_data

    if not do_create_material:
        report_message = f"Imported {len(pattern.stitches)} stitches"
        return report_message, report_type

    report_message = f"Imported {len(pattern.stitches)} stitches with {len(pattern.threadlist)} threads"
    return report_message, report_type


class ImportEmbroideryData(Operator, ImportHelper):
    """Import embroidery data"""

    bl_idname = "import_scene.embroidery"
    bl_label = "Import Embroidery"

    filter_glob: bpy.props.StringProperty(
        # these are all types supported by pyembroidery
        default="*.pes;*.dst;*.exp;*.jef;*.vp3;*.10o;*.100;*.bro;*.dat;*.dsb;*.dsz;*.emd;*.exy;*.fxy;*.gt;*.hus;*.inb;*.jpx;*.ksm;*.max;*.mit;*.new;*.pcd;*.pcm;*.pcq;*.pcs;*.pec;*.phb;*.phc;*.sew;*.shv;*.stc;*.stx;*.tap;*.tbf;*.u01;*.xxx;*.zhs;*.zxy;*.gcode",
        options={"HIDDEN"},
        maxlen=255,
    )  # type: ignore

    import_scale: bpy.props.FloatProperty(
        name="Scale",
        description="Scale the imported data",
        default=10000.0,
        min=0.0001,
        max=1000.0,
        options={"HIDDEN"},
    )  # type: ignore

    thread_thickness: bpy.props.FloatProperty(
        name="Thread Thickness (mm)",
        description="Thickness of the thread in milimeters",
        default=0.2,
        min=0.01,
        max=2.00,
        options={"HIDDEN"},
    )  # type: ignore

    show_jump_wires: bpy.props.BoolProperty(
        name="Import jump wires",
        description="Include or exclude jump wires from the design",
        default=True,
        options={"HIDDEN"},
    )  # type: ignore

    do_create_material: bpy.props.BoolProperty(
        name="Create material",
        description="Create a material based on the thread information in the file",
        default=True,
        options={"HIDDEN"},
    )  # type: ignore

    create_collection: bpy.props.BoolProperty(
        name="Create a collection",
        description="Create a new collection for the created objects",
        default=True,
        options={"HIDDEN"},
    )  # type: ignore

    line_depth: bpy.props.EnumProperty(
        name="Line type",
        description="Choose what type of lines to use for the embroidery",
        items=[
            ("NO_THICKNESS", "No thickness (curve only)", "Only curves, no thickness"),
            (
                "GEOMETRY_NODES",
                "Using geometry nodes",
                "Create a geometry node setup to add thickness. Most versatile.",
            ),
            ("BEVEL", "Using bevel", "Adds thickness through the bevel property"),
        ],
        default="GEOMETRY_NODES",
        options={"HIDDEN"},
    )  # type: ignore

    def draw(self, context):
        layout = self.layout
        layout.label(text="Import Embroidery Options")
        layout.prop(self, "show_jump_wires")
        layout.prop(self, "do_create_material")
        layout.prop(self, "create_collection")

        col = layout.column(align=True)
        col.label(text="Thickness type:")
        col.prop(self, "line_depth", expand=True)

        row = layout.row()
        row.active = self.line_depth in ["GEOMETRY_NODES", "BEVEL"]
        row.prop(self, "thread_thickness", text="Thread Thickness (mm)")

    def execute(self, context):
        thread_thickness = self.thread_thickness / 1000.0

        report_message, report_type = parse_embroidery_data(
            context,
            self.filepath,
            self.show_jump_wires,
            self.do_create_material,
            self.line_depth,
            thread_thickness,
            self.create_collection,
        )

        self.report({report_type}, report_message)
        return {"FINISHED"}


classes = [
    ImportEmbroideryData,
]


def menu_func_import(self, context):
    self.layout.operator(ImportEmbroideryData.bl_idname, text="Embroidery Import")


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.TOPBAR_MT_file_import.append(menu_func_import)


def unregister():
    for cls in classes:
        bpy.utils.unregister_class(cls)
    bpy.types.TOPBAR_MT_file_import.remove(menu_func_import)
