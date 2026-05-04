import math
from pathlib import Path

OUT = Path.home() / "btp_3d_fresh" / "experiments" / "test.argos"

LIB_CONTROLLER = "/Users/harshvardhanmohite/btp_3d_fresh/build/libbtp_controller.dylib"
LIB_LOOP = "/Users/harshvardhanmohite/btp_3d_fresh/build/libbtp_loop.dylib"

start = (-8.0, -8.0, 1.5)
ring_r = 1.25

buildings = [
    ("bld_0",  -6.2, -5.2, 1.1, 1.2, 1.4, 2.2),
    ("bld_1",  -3.8, -6.4, 1.0, 1.1, 1.0, 2.0),
    ("bld_2",  -1.3, -4.8, 1.4, 1.4, 1.0, 2.8),
    ("bld_3",   1.8, -6.0, 1.2, 1.2, 1.3, 2.4),

    ("bld_4",  -7.0, -1.5, 1.3, 1.1, 1.4, 2.6),
    ("bld_5",  -4.4, -0.6, 1.6, 1.3, 1.1, 3.2),
    ("bld_6",  -1.4, -1.8, 1.2, 1.0, 1.6, 2.4),
    ("bld_7",   2.4, -1.2, 1.5, 1.5, 1.1, 3.0),

    ("bld_8",  -5.8,  2.9, 1.2, 1.4, 1.1, 2.4),
    ("bld_9",  -2.8,  3.4, 1.7, 1.1, 1.6, 3.4),
    ("bld_10",  0.7,  2.4, 1.3, 1.2, 1.3, 2.6),
    ("bld_11",  4.0,  2.8, 1.5, 1.4, 1.2, 3.0),

    ("bld_12",  2.0,  6.5, 1.2, 1.2, 1.1, 2.4),
    ("bld_13",  6.0,  5.6, 1.4, 1.3, 1.3, 2.8),
]

air_obstacles = [
    ("air_0", -5.0, -2.8, 2.2, 0.30, 0.30, 0.25),
    ("air_1", -2.3,  0.9, 2.4, 0.28, 0.28, 0.24),
    ("air_2",  0.5, -3.0, 2.1, 0.25, 0.25, 0.22),
    ("air_3",  2.5,  1.2, 2.5, 0.30, 0.30, 0.25),
    ("air_4",  5.2,  3.5, 2.3, 0.26, 0.26, 0.22),
    ("air_5",  0.0,  6.0, 2.2, 0.25, 0.25, 0.22),
]

def proto_box(pid, x, y, z, sx, sy, sz, mass="8.0", movable="false"):
    return f'''    <prototype id="{pid}" movable="{movable}">
      <body position="{x:.3f},{y:.3f},{z:.3f}"/>
      <links ref="b">
        <link id="b" geometry="box" mass="{mass}" size="{sx:.3f},{sy:.3f},{sz:.3f}"/>
      </links>
    </prototype>
'''

def proto_sphere(pid, x, y, z, r, mass, ctrl=None, movable="true"):
    ctrl_line = f'      <controller config="{ctrl}"/>\n' if ctrl else ""
    return f'''    <prototype id="{pid}" movable="{movable}">
      <body position="{x:.3f},{y:.3f},{z:.3f}"/>
{ctrl_line}      <links ref="b">
        <link id="b" geometry="sphere" mass="{mass}" radius="{r:.3f}"/>
      </links>
    </prototype>
'''

xml = '''<?xml version="1.0" ?>
<argos-configuration>

  <framework>
    <system threads="0"/>
    <experiment length="3000000" ticks_per_second="10" random_seed="1"/>
  </framework>

  <physics_engines>
    <dynamics3d id="dyn3d"/>
  </physics_engines>

  <media></media>

  <controllers>
    <infl id="infl" library="''' + LIB_CONTROLLER + '''">
      <actuators/>
      <sensors/>
      <params/>
    </infl>

    <dec id="dec" library="''' + LIB_CONTROLLER + '''">
      <actuators/>
      <sensors/>
      <params/>
    </dec>
  </controllers>

  <loop_functions library="''' + LIB_LOOP + '''" label="loop"/>

  <arena size="22,22,5" center="0,0,2.5">

    <!-- No floor: avoids ARGoS floor/source XML issue -->

    <!-- Ground building obstacles -->
'''

for b in buildings:
    xml += proto_box(*b, mass="10.0", movable="false")

xml += '''
    <!-- Small moving aerial obstacles. Their motion is controlled from loop_functions. -->
'''

for a in air_obstacles:
    xml += proto_box(*a, mass="0.3", movable="true")

xml += '''
    <!-- Start marker -->
'''
xml += proto_sphere("start_marker", start[0], start[1], 0.25, 0.18, "0.1", None, "false")

xml += '''
    <!-- Goal marker -->
'''
xml += proto_sphere("goal_marker", 8.0, 8.0, 0.25, 0.18, "0.1", None, "false")

xml += '''
    <!-- Influential core -->
'''

core_offsets = [
    (0.00, 0.00),
    (0.25, 0.00),
    (-0.25, 0.00),
    (0.00, 0.25),
    (0.00, -0.25),
]

for i, (dx, dy) in enumerate(core_offsets):
    xml += proto_sphere(
        f"infl_{i}",
        start[0] + dx,
        start[1] + dy,
        start[2],
        0.06,
        "0.15",
        "infl",
        "true"
    )

xml += '''
    <!-- 40 decoys in protection ring -->
'''

for j in range(40):
    th = 2.0 * math.pi * j / 40.0
    x = start[0] + ring_r * math.cos(th)
    y = start[1] + ring_r * math.sin(th)

    xml += proto_sphere(
        f"dec_{j}",
        x,
        y,
        start[2],
        0.035,
        "0.06",
        "dec",
        "true"
    )

xml += '''
  </arena>

  <visualization>
    <qt-opengl>
      <camera>
        <placement position="0,-18,14" look_at="0,0,1.5"/>
      </camera>
    </qt-opengl>
  </visualization>

</argos-configuration>
'''

OUT.write_text(xml)
print(f"Wrote: {OUT}")
