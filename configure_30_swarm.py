from pathlib import Path
import math
import re

ROOT = Path.home() / "btp_3d_fresh"
ARGOS_FILE = ROOT / "experiments" / "test.argos"
CPP_FILE = ROOT / "src" / "btp_loop_functions.cpp"

LIB_CONTROLLER = "/Users/harshvardhanmohite/btp_3d_fresh/build/libbtp_controller.dylib"
LIB_LOOP = "/Users/harshvardhanmohite/btp_3d_fresh/build/libbtp_loop.dylib"

# ============================================================
# SWARM CONFIG
# ============================================================

TOTAL_SWARM = 30
N_INFL = 4
N_DECOY = TOTAL_SWARM - N_INFL

START = (-7.0, -7.0, 1.50)
GOAL = (7.0, 7.0, 1.50)

RING_R = 1.35

# Reduced obstacle environment:
# 6 ground buildings, placed with wide navigation gaps
GROUND_BUILDINGS = [
    # id, x, y, z, sx, sy, sz
    ("bld_0", -4.8, -4.0, 1.1, 1.2, 1.4, 2.2),
    ("bld_1", -1.3, -4.5, 1.2, 1.3, 1.1, 2.4),
    ("bld_2", -5.2,  0.0, 1.3, 1.2, 1.5, 2.6),
    ("bld_3", -1.5,  1.2, 1.4, 1.5, 1.1, 2.8),
    ("bld_4",  2.3,  0.0, 1.3, 1.4, 1.2, 2.6),
    ("bld_5",  4.8,  4.2, 1.2, 1.3, 1.4, 2.4),
]

# Small aerial obstacles
AIR_OBSTACLES = [
    ("air_0", -3.5, -1.5, 2.25, 0.28, 0.28, 0.22),
    ("air_1",  0.8,  2.8, 2.35, 0.26, 0.26, 0.22),
    ("air_2",  4.0,  1.2, 2.20, 0.25, 0.25, 0.20),
]

WAYPOINTS = [
    (-7.0, -7.0, 1.50),
    (-6.8, -3.0, 1.50),
    (-4.0, -1.2, 1.50),
    (-4.2,  2.8, 1.50),
    (-1.0,  4.5, 1.50),
    ( 2.5,  3.2, 1.50),
    ( 5.0,  5.5, 1.50),
    ( 7.0,  7.0, 1.50),
]


# ============================================================
# XML GENERATION
# ============================================================

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


def write_argos():
    sx, sy, sz = START
    gx, gy, gz = GOAL

    xml = f'''<?xml version="1.0" ?>
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
    <infl id="infl" library="{LIB_CONTROLLER}">
      <actuators/>
      <sensors/>
      <params/>
    </infl>

    <dec id="dec" library="{LIB_CONTROLLER}">
      <actuators/>
      <sensors/>
      <params/>
    </dec>
  </controllers>

  <loop_functions library="{LIB_LOOP}" label="loop"/>

  <arena size="18,18,5" center="0,0,2.5">

    <!-- Reduced building obstacle environment: 6 buildings + 3 small aerial obstacles -->
    <!-- No floor: avoids ARGoS floor/source XML issues -->

'''

    for b in GROUND_BUILDINGS:
        xml += proto_box(*b, mass="10.0", movable="false")

    xml += '''    <!-- Small moving aerial obstacles; motion controlled by loop_functions.cpp -->

'''

    for a in AIR_OBSTACLES:
        xml += proto_box(*a, mass="0.3", movable="true")

    xml += '''    <!-- Start and goal markers -->

'''
    xml += proto_sphere("start_marker", sx, sy, 0.25, 0.18, "0.1", None, "false")
    xml += proto_sphere("goal_marker", gx, gy, 0.25, 0.18, "0.1", None, "false")

    xml += '''    <!-- Influential agents: 4 / 30 = 13.33 percent of swarm -->

'''

    core_offsets = [
        (0.00, 0.00),
        (0.28, 0.00),
        (-0.28, 0.00),
        (0.00, 0.28),
    ]

    for i, (dx, dy) in enumerate(core_offsets):
        xml += proto_sphere(
            f"infl_{i}",
            sx + dx,
            sy + dy,
            sz,
            0.065,
            "0.15",
            "infl",
            "true"
        )

    xml += '''    <!-- Decoy agents: 26 / 30 = 86.67 percent of swarm -->

'''

    for j in range(N_DECOY):
        th = 2.0 * math.pi * j / N_DECOY
        x = sx + RING_R * math.cos(th)
        y = sy + RING_R * math.sin(th)

        xml += proto_sphere(
            f"dec_{j}",
            x,
            y,
            sz,
            0.035,
            "0.06",
            "dec",
            "true"
        )

    xml += '''  </arena>

  <visualization>
    <qt-opengl>
      <camera>
        <placement position="0,-15,12" look_at="0,0,1.5"/>
      </camera>
    </qt-opengl>
  </visualization>

</argos-configuration>
'''

    ARGOS_FILE.write_text(xml)
    print(f"Wrote {ARGOS_FILE}")
    print(f"Swarm: {N_INFL} influentials + {N_DECOY} decoys = {TOTAL_SWARM}")
    print(f"Influential percentage = {100.0 * N_INFL / TOTAL_SWARM:.2f}%")


# ============================================================
# C++ PATCHING
# ============================================================

def cpp_vec(x, y, z):
    return f"CVector3({x:.3f}, {y:.3f}, {z:.3f})"


def make_cpp_obstacles_block():
    lines = []
    lines.append('   void InitObstacles() {')
    lines.append('      obstacles.clear();')
    lines.append('')
    lines.append('      /*')
    lines.append('         Reduced obstacle environment for 30-agent swarm.')
    lines.append('         6 ground buildings + 3 small dynamic aerial obstacles.')
    lines.append('         Wide route spacing avoids choking/narrow-throat behavior.')
    lines.append('      */')

    for oid, x, y, z, sx, sy, sz in GROUND_BUILDINGS:
        half = (sx / 2.0, sy / 2.0, sz / 2.0)
        lines.append(
            f'      obstacles.push_back({{"{oid}",  {cpp_vec(x, y, z)}, {cpp_vec(*half)}, false}});'
        )

    lines.append('')
    lines.append('      /*')
    lines.append('         Small dynamic aerial obstacles.')
    lines.append('      */')

    dynamic_params = [
        ("air_0", (1.0, 0.0, 0.18), 0.020, 0.0),
        ("air_1", (0.0, 1.0, 0.18), 0.022, 1.1),
        ("air_2", (0.9, 0.0, 0.15), 0.018, 2.2),
    ]

    for (oid, x, y, z, sx, sy, sz), (_, amp, freq, phase) in zip(AIR_OBSTACLES, dynamic_params):
        half = (sx / 2.0, sy / 2.0, sz / 2.0)
        lines.append(
            f'      obstacles.push_back({{"{oid}", {cpp_vec(x, y, z)}, {cpp_vec(*half)}, true,'
        )
        lines.append(
            f'                           {cpp_vec(*amp)}, {freq:.3f}, {phase:.3f}}});'
        )

    lines.append('   }')
    return "\n".join(lines)


def make_cpp_waypoints_block():
    lines = []
    lines.append('   void InitWaypoints() {')
    lines.append('      waypoints.clear();')
    lines.append('')
    lines.append('      /*')
    lines.append('         Longer route through the reduced obstacle field.')
    lines.append('         Waypoints are placed in wide corridors, not tiny throats.')
    lines.append('      */')

    for x, y, z in WAYPOINTS:
        lines.append(f'      waypoints.push_back({cpp_vec(x, y, z)});')

    lines.append('')
    lines.append('      current_wp = 1;')
    lines.append('')
    lines.append('      initial_goal_dist = XYOnly(goal - start).Length();')
    lines.append('      if(initial_goal_dist < 1e-6) {')
    lines.append('         initial_goal_dist = 1.0;')
    lines.append('      }')
    lines.append('')
    lines.append('      LOG << "Loaded " << waypoints.size()')
    lines.append('          << " waypoints | initial_goal_dist=" << initial_goal_dist')
    lines.append('          << std::endl;')
    lines.append('   }')
    return "\n".join(lines)


def patch_cpp():
    s = CPP_FILE.read_text()

    # Mission setup
    s = re.sub(
        r'Real flight_z\s*=\s*[-0-9.]+;',
        'Real flight_z = 1.50;',
        s
    )

    s = re.sub(
        r'CVector3 start\s*=\s*CVector3\([^)]+\);',
        f'CVector3 start = {cpp_vec(*START)};',
        s
    )

    s = re.sub(
        r'CVector3 goal\s*=\s*CVector3\([^)]+\);',
        f'CVector3 goal  = {cpp_vec(*GOAL)};',
        s
    )

    # Ring/band for 26 decoys around 4 influentials
    replacements = {
        r'Real goal_accept_radius\s*=\s*[-0-9.]+;': 'Real goal_accept_radius = 1.35;',
        r'Real wp_accept_radius\s*=\s*[-0-9.]+;': 'Real wp_accept_radius   = 0.95;',
        r'Real rmin\s*=\s*[-0-9.]+;': 'Real rmin  = 0.85;',
        r'Real rmax\s*=\s*[-0-9.]+;': 'Real rmax  = 1.85;',
        r'Real rring\s*=\s*[-0-9.]+;': 'Real rring = 1.35;',
        r'Real tau\s*=\s*[-0-9.]+;': 'Real tau   = 0.60;',
        r'Real infl_step_max\s*=\s*[-0-9.]+;': 'Real infl_step_max = 0.034;',
        r'Real k_goal_core\s*=\s*[-0-9.]+;': 'Real k_goal_core     = 0.065;',
        r'Real dec_sep_radius\s*=\s*[-0-9.]+;': 'Real dec_sep_radius  = 0.32;',
        r'Real graph_radius\s*=\s*[-0-9.]+;': 'Real graph_radius = 2.05;',
        r'Real xmin\s*=\s*[-0-9.]+;': 'Real xmin = -8.7;',
        r'Real xmax\s*=\s*[-0-9.]+;': 'Real xmax =  8.7;',
        r'Real ymin\s*=\s*[-0-9.]+;': 'Real ymin = -8.7;',
        r'Real ymax\s*=\s*[-0-9.]+;': 'Real ymax =  8.7;',
        r'UInt32 replan_period\s*=\s*[0-9]+;': 'UInt32 replan_period = 45;',
        r'UInt32 horizon\s*=\s*[0-9]+;': 'UInt32 horizon = 5;',
        r'UInt32 population_size\s*=\s*[0-9]+;': 'UInt32 population_size = 4;',
    }

    for pattern, repl in replacements.items():
        s = re.sub(pattern, repl, s)

    # Replace InitObstacles
    s = re.sub(
        r'   void InitObstacles\(\) \{.*?\n   void LinkObstacleVisuals\(\) \{',
        make_cpp_obstacles_block() + '\n\n   void LinkObstacleVisuals() {',
        s,
        flags=re.S
    )

    # Replace InitWaypoints
    s = re.sub(
        r'   void InitWaypoints\(\) \{.*?\n   Real Rand01\(\) const \{',
        make_cpp_waypoints_block() + '\n\n   Real Rand01() const {',
        s,
        flags=re.S
    )

    CPP_FILE.write_text(s)
    print(f"Patched {CPP_FILE}")


def main():
    if not CPP_FILE.exists():
        raise FileNotFoundError(f"Cannot find {CPP_FILE}")

    write_argos()
    patch_cpp()

    print("\nDone.")
    print("Now rebuild and run:")
    print("  cd ~/btp_3d_fresh/build")
    print("  make -j4")
    print("  cd ~/btp_3d_fresh")
    print("  ~/argos3/build/core/argos3 -c experiments/test.argos")


if __name__ == "__main__":
    main()
