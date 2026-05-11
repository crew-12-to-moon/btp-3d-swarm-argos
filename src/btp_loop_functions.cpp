#include <argos3/core/simulator/loop_functions.h>
#include <argos3/core/simulator/space/space.h>
#include <argos3/plugins/robots/prototype/simulator/prototype_entity.h>
#include <argos3/core/utility/logging/argos_log.h>
#include <argos3/core/utility/math/quaternion.h>
#include <argos3/core/utility/math/angles.h>

#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <limits>
#include <cstdlib>
#include <utility>
#include <array>

using namespace argos;

struct Chromosome {
   Real s      = 0.045;
   Real wcent  = 0.70;
   Real wsep   = 2.20;
   Real wband  = 2.80;
   Real wobs   = 1.60;

   /*
      Step-1 GA update:
      scalar fitness is retained only as a tie-break / reward value.
      Selection is now based on priority-threshold violations first,
      then Pareto rank + crowding distance when thresholds are satisfied.
   */
   Real fitness = -1e9;

   Real q1 = 0.0;   // maximize
   Real q2 = 0.0;   // maximize
   Real q3 = 0.0;   // maximize
   Real q4 = 0.0;   // maximize
   Real q5 = 0.0;   // maximize
   Real q6 = 0.0;   // minimize

   Real V1 = 0.0;
   Real V2 = 0.0;
   Real V3 = 0.0;
   Real V4 = 0.0;
   Real V5 = 0.0;
   Real V6 = 0.0;

   UInt32 pareto_rank = 999999;
   Real crowding_distance = 0.0;
};

struct Agent {
   std::string id;
   CPrototypeEntity* entity = nullptr;
   CVector3 pos;
   CVector3 vel;
   bool influential = false;
   int slot = -1;

   Chromosome active;
   std::vector<Chromosome> population;
};

struct BoxObs {
   std::string id;
   CVector3 center;
   CVector3 half;

   std::vector<CVector3> vertices;
   std::vector<std::array<int, 3>> faces;
   std::vector<CVector3> face_centroids;
   std::vector<CVector3> face_normals;

   bool dynamic = false;
   CVector3 amp;
   Real freq = 0.0;
   Real phase = 0.0;
   CPrototypeEntity* visual = nullptr;
};

class CBTP_LoopFunctions : public CLoopFunctions {

public:
   std::vector<Agent> agents;
   std::vector<BoxObs> obstacles;
   std::vector<CVector3> shell_dirs;

   UInt32 step = 0;
   bool initialized = false;

   std::ofstream csv;
   std::ofstream agent_csv;

   Real last_bridge_distance = 0.0;
   UInt32 last_bridge_pairs = 0;

   /*
      30-swarm setup:
      4 influentials + 26 decoys = 30 total
      Influential fraction = 13.33%
   */
   Real flight_z = 1.50;

   CVector3 start = CVector3(-7.0, -7.0, 1.50);
   CVector3 goal  = CVector3( 7.0,  7.0, 1.50);

   Real goal_accept_radius = 1.35;
   bool mission_complete   = false;

   /*
      Protection band.
      Ring radius is large enough for 26 decoys without over-compression.
   */
   Real rmin  = 1.00;
   Real rmax  = 2.10;
   Real rring = 1.55;
   Real tau   = 0.60;

   /*
      Step-2 3D protection shell:
      Decoy slots are distributed over a Fibonacci sphere around the influential core.
      This replaces the earlier purely planar ring and gives top/bottom protection.
   */
   Real shell_radius = 1.55;
   Real shell_slot_gain = 0.135;
   Real shell_z_scale = 0.75;
   Real zmin = 0.65;
   Real zmax = 2.75;
   Real vertical_band_min = 0.35;

   /*
      Influential motion.
      IMPORTANT FIX:
      influential core cohesion is strengthened so one influential cannot be left behind.
   */
   Real infl_step_max = 0.034;

   Real k_goal_core     = 0.065;
   Real k_core_cohesion = 0.050;
   Real k_infl_sep      = 0.020;
   Real k_obs_infl      = 0.050;

   /*
      Influential-core rescue.
      Prevents one influential from getting isolated due to obstacles or local forces.
   */
   Real max_infl_core_dist = 0.90;
   Real k_infl_core_rescue = 0.090;

   /*
      Protection-first stabilization.
      Slot force is strong, but not so strong that decoys overlap in tight corridors.
   */
   Real k_slot_bias = 0.115;
   Real k_tangent   = 0.0005;

   Real infl_sep_radius = 0.38;
   Real dec_sep_radius  = 0.42;

   /*
      Obstacle SDF settings
   */
   Real obs_safe      = 0.35;
   Real obs_influence = 1.35;

   /*
      Arena clamp for 18 x 18 arena
   */
   Real xmin = -8.7;
   Real xmax =  8.7;
   Real ymin = -8.7;
   Real ymax =  8.7;

   /*
      Connectivity radius
   */
   Real graph_radius = 2.05;

   /*
      Mission progress
   */

   /*
      Mission progress
   */
   Real initial_goal_dist = 1.0;

   /*
      Stable lightweight decentralized GA-MPC.
      GA is a fine tuner, not the dominant ring controller.
   */
   UInt32 replan_period = 80;
   UInt32 horizon = 4;
   UInt32 population_size = 4;

   Real fit_prot   = 1.25;
   Real fit_struct = 0.60;
   Real fit_obs    = 3.20;
   Real fit_move   = 0.35;

   /*
      Generic connectivity reward is secondary.
      Slot reward and protection reward dominate.
   */
   Real fit_conn   = 0.20;
   Real fit_slot   = 2.20;

   /*
      Connectivity break penalty.
   */
   Real fit_conn_break = 2.50;

   Real mutation_rate = 0.30;
   Real mutation_scale = 0.16;

   /*
      Step-1 GA update from updated report:
      q(chi) = [q1, q2, q3, q4]
      q1 = protection reward, q2 = influential structural reward,
      q3 = obstacle penalty, q4 = motion penalty.

      T = [Pmin, Smin, Omax, Mmax].
      q1 and q2 are maximized; q3 and q4 are minimized.
   */
   Real T1_dd   = 0.22;   // min decoy-decoy distance
   Real T2_obs  = 0.15;   // min obstacle clearance
   Real T3_l2   = 0.10;   // lambda2 proxy threshold
   Real T4_prot = 1.00;   // protection threshold
   Real T5_str  = 0.35;   // structural threshold
   Real T6_mov  = 0.0045; // motion penalty max

   UInt32 csv_log_period = 25;
   UInt32 terminal_log_period = 100;

   UInt32 weak_connectivity_steps = 0;

   /*
      Mission completion thresholds.
      Mission is complete only if the core reaches the goal
      AND swarm quality is acceptable.
   */
   Real complete_min_lambda2_proxy = 0.1;
   Real complete_min_decoy_dist    = 0.12;
   Real complete_min_obs_dist      = 0.0;
   Real complete_min_protection    = 0.45;
   Real complete_max_infl_core_dist = 1.10;

   /*
      Adaptive replacement-link connectivity.
      Exact old edges are not preserved.
      Each agent must maintain at least k_conn local links.
   */
   UInt32 k_conn = 3;
   Real k_conn_rescue = 0.150;

   /*
      Step-3 connectivity repair.
      If disk graph splits into components, closest components are pulled together.
      This is non-rigid replacement-link repair, not old-edge preservation.
   */
   Real k_component_bridge = 0.120;
   Real bridge_soft_margin = 0.85;

   void Init(TConfigurationNode&) override {
      LOG << "BTP ARGoS STEP-3.2 LOCAL-K-CONNECTIVITY initialized: priority GA + 3D shell + component bridge repair" << std::endl;

      std::srand(7);

      LoadAgents();
      InitObstacles();
      LinkObstacleVisuals();
      InitShellDirections();
      InitDecoyPopulations();
      InitDefinedFormation();
      OpenCSV();
   }

   void Reset() override {
      agents.clear();
      obstacles.clear();

      step = 0;
      initialized = false;
      initial_goal_dist = 1.0;
      mission_complete = false;
      weak_connectivity_steps = 0;

      if(csv.is_open()) {
         csv.close();
      }
      if(agent_csv.is_open()) {
         agent_csv.close();
      }

      last_bridge_distance = 0.0;
      last_bridge_pairs = 0;

      std::srand(7);

      LoadAgents();
      InitObstacles();
      LinkObstacleVisuals();
      InitShellDirections();
      InitDecoyPopulations();
      InitDefinedFormation();
      OpenCSV();
   }

   void Destroy() override {
      if(csv.is_open()) {
         csv.close();
      }
      if(agent_csv.is_open()) {
         agent_csv.close();
      }
   }

   void OpenCSV() {
      csv.open("btp_argos_metrics.csv");

      csv << "step,"
          << "core_x,"
          << "core_y,"
          << "core_z,"
          << "dist_goal,"
          << "max_infl_core_dist,"
          << "avg_band_error,"
          << "min_decoy_decoy,"
          << "min_obstacle_decoy,"
          << "lambda2_proxy,"
          << "min_neighbor_count,"
          << "weak_connectivity_steps,"
          << "num_connected_components,"
          << "largest_component_fraction,"
          << "bridge_pairs,"
          << "bridge_distance,"
          << "protection_infl_mean,"
          << "protection_decoy_mean,"
          << "struct_infl_mean,"
          << "struct_decoy_mean,"
          << "combined_infl_mean,"
          << "combined_decoy_mean,"
          << "margin_prot_mean,"
          << "margin_prot_robust,"
          << "margin_prot_worst,"
          << "margin_struct_mean,"
          << "margin_struct_robust,"
          << "margin_struct_worst,"
          << "margin_combined_mean,"
          << "margin_combined_robust,"
          << "margin_combined_worst,"
          << "formation_health,"
          << "goal_progress_score,"
          << "criticality_score,"
          << "mean_shell_error,"
          << "spherical_coverage_score,"
          << "vertical_protection_score,"
          << "upper_decoy_count,"
          << "lower_decoy_count,"
          << "mission_complete"
          << "\n";

      agent_csv.open("btp_agent_positions.csv");
      agent_csv << "step,agent_id,type,x,y,z\n";
   }

   void LoadAgents() {
      auto& ents = GetSpace().GetEntitiesByType("prototype");

      for(auto& it : ents) {
         CPrototypeEntity* e = any_cast<CPrototypeEntity*>(it.second);
         std::string id = e->GetId();

         if(id.find("infl_") == 0 || id.find("dec_") == 0) {
            Agent a;
            a.id = id;
            a.entity = e;
            a.pos = e->GetEmbodiedEntity().GetOriginAnchor().Position;
            a.vel = CVector3(0,0,0);
            a.influential = (id.find("infl_") == 0);
            agents.push_back(a);
         }
      }

      std::sort(agents.begin(), agents.end(),
         [](const Agent& a, const Agent& b) {
            return a.id < b.id;
         });

      int slot = 0;
      for(auto& a : agents) {
         if(!a.influential) {
            a.slot = slot;
            slot++;
         }
      }

      LOG << "Loaded agents: " << agents.size()
          << " | influentials=" << CountInfl()
          << " | decoys=" << CountDec()
          << std::endl;
   }

   void InitObstacles() {
      obstacles.clear();

      /*
         Reduced obstacle environment for 30-agent swarm.
         6 ground buildings + 3 small dynamic aerial obstacles.
         Waypoints use wide corridors.
      */
      {
         BoxObs o; o.id = "bld_0"; o.center = CVector3(-4.800, -4.000, 1.100); o.half = CVector3(0.600, 0.700, 1.100); o.dynamic = false;
         BuildBoxPolyhedron(o); obstacles.push_back(o);
      }
      {
         BoxObs o; o.id = "bld_1"; o.center = CVector3(-1.300, -4.500, 1.200); o.half = CVector3(0.650, 0.550, 1.200); o.dynamic = false;
         BuildBoxPolyhedron(o); obstacles.push_back(o);
      }
      {
         BoxObs o; o.id = "bld_2"; o.center = CVector3(-5.200,  0.000, 1.300); o.half = CVector3(0.600, 0.750, 1.300); o.dynamic = false;
         BuildBoxPolyhedron(o); obstacles.push_back(o);
      }
      {
         BoxObs o; o.id = "bld_3"; o.center = CVector3(-1.500,  1.200, 1.400); o.half = CVector3(0.750, 0.550, 1.400); o.dynamic = false;
         BuildBoxPolyhedron(o); obstacles.push_back(o);
      }
      {
         BoxObs o; o.id = "bld_4"; o.center = CVector3( 2.300,  0.000, 1.300); o.half = CVector3(0.700, 0.600, 1.300); o.dynamic = false;
         BuildBoxPolyhedron(o); obstacles.push_back(o);
      }
      {
         BoxObs o; o.id = "bld_5"; o.center = CVector3( 4.800,  4.200, 1.200); o.half = CVector3(0.650, 0.700, 1.200); o.dynamic = false;
         BuildBoxPolyhedron(o); obstacles.push_back(o);
      }

      /*
         Small dynamic aerial obstacles.
      */
      {
         BoxObs o; o.id = "air_0"; o.center = CVector3(-3.500, -1.500, 2.250); o.half = CVector3(0.140, 0.140, 0.110);
         o.dynamic = true; o.amp = CVector3(1.000, 0.000, 0.180); o.freq = 0.020; o.phase = 0.000;
         BuildBoxPolyhedron(o); obstacles.push_back(o);
      }
      {
         BoxObs o; o.id = "air_1"; o.center = CVector3( 0.800,  2.800, 2.350); o.half = CVector3(0.130, 0.130, 0.110);
         o.dynamic = true; o.amp = CVector3(0.000, 1.000, 0.180); o.freq = 0.022; o.phase = 1.100;
         BuildBoxPolyhedron(o); obstacles.push_back(o);
      }
      {
         BoxObs o; o.id = "air_2"; o.center = CVector3( 4.000,  1.200, 2.200); o.half = CVector3(0.125, 0.125, 0.100);
         o.dynamic = true; o.amp = CVector3(0.900, 0.000, 0.150); o.freq = 0.018; o.phase = 2.200;
         BuildBoxPolyhedron(o); obstacles.push_back(o);
      }
   }


   void LinkObstacleVisuals() {
      auto& ents = GetSpace().GetEntitiesByType("prototype");

      for(auto& it : ents) {
         CPrototypeEntity* e = any_cast<CPrototypeEntity*>(it.second);
         std::string id = e->GetId();

         for(auto& o : obstacles) {
            if(o.id == id) {
               o.visual = e;
            }
         }
      }
   }

   Real Rand01() const {
      return static_cast<Real>(std::rand()) / static_cast<Real>(RAND_MAX);
   }

   Real RandRange(Real a, Real b) const {
      return a + (b - a) * Rand01();
   }

   Chromosome RandomChromosome() const {
      Chromosome c;

      /*
         Reduced speed range prevents overshoot and trailing clusters.
         Separation range is increased because 26 decoys need more spacing authority.
      */
      c.s     = RandRange(0.020, 0.055);
      c.wcent = RandRange(0.20, 1.00);
      c.wsep  = RandRange(2.80, 6.00);
      c.wband = RandRange(1.60, 4.00);
      c.wobs  = RandRange(1.10, 3.20);
      c.fitness = -1e9;

      return c;
   }

   Chromosome MutateChromosome(Chromosome c) const {
      auto mutate_scalar = [&](Real x, Real lo, Real hi) {
         if(Rand01() < mutation_rate) {
            Real scale = 1.0 + mutation_scale * RandRange(-1.0, 1.0);
            x *= scale;
         }

         if(x < lo) x = lo;
         if(x > hi) x = hi;
         return x;
      };

      c.s     = mutate_scalar(c.s,     0.018, 0.060);
      c.wcent = mutate_scalar(c.wcent, 0.100, 1.300);
      c.wsep  = mutate_scalar(c.wsep,  2.000, 7.000);
      c.wband = mutate_scalar(c.wband, 1.000, 5.000);
      c.wobs  = mutate_scalar(c.wobs,  0.700, 4.000);
      c.fitness = -1e9;

      return c;
   }

   Chromosome Crossover(const Chromosome& a, const Chromosome& b) const {
      Chromosome c;
      c.s     = (Rand01() < 0.5) ? a.s     : b.s;
      c.wcent = (Rand01() < 0.5) ? a.wcent : b.wcent;
      c.wsep  = (Rand01() < 0.5) ? a.wsep  : b.wsep;
      c.wband = (Rand01() < 0.5) ? a.wband : b.wband;
      c.wobs  = (Rand01() < 0.5) ? a.wobs  : b.wobs;
      c.fitness = -1e9;
      return c;
   }

   void InitDecoyPopulations() {
      for(auto& a : agents) {
         if(a.influential) {
            continue;
         }

         a.population.clear();

         for(UInt32 i = 0; i < population_size; ++i) {
            a.population.push_back(RandomChromosome());
         }

         if(!a.population.empty()) {
            a.active = a.population[0];
         }
      }
   }

   int CountInfl() const {
      int c = 0;
      for(const auto& a : agents) {
         if(a.influential) c++;
      }
      return c;
   }

   int CountDec() const {
      int c = 0;
      for(const auto& a : agents) {
         if(!a.influential) c++;
      }
      return c;
   }

   CVector3 XYOnly(CVector3 v) const {
      v.SetZ(0.0);
      return v;
   }

   CVector3 SetFlightZ(CVector3 p) const {
      p.SetZ(flight_z);
      return p;
   }

   CVector3 ClampZ(CVector3 p) const {
      p.SetZ(std::max(zmin, std::min(zmax, p.GetZ())));
      return p;
   }

   CVector3 NormalizeSafe3D(const CVector3& v) const {
      Real L = v.Length();
      if(L < 1e-9) {
         return CVector3(0,0,0);
      }
      return v / L;
   }

   CVector3 Limit3D(CVector3 v, Real max_mag) const {
      Real L = v.Length();
      if(L > max_mag && L > 1e-9) {
         return (v / L) * max_mag;
      }
      return v;
   }

   CVector3 NormalizeSafeXY(const CVector3& v) const {
      CVector3 u = v;
      u.SetZ(0.0);

      Real L = u.Length();
      if(L < 1e-9) {
         return CVector3(0,0,0);
      }

      return u / L;
   }

   CVector3 LimitXY(CVector3 v, Real max_mag) const {
      v.SetZ(0.0);

      Real L = v.Length();
      if(L > max_mag && L > 1e-9) {
         return (v / L) * max_mag;
      }

      return v;
   }

   CVector3 Clamp(CVector3 p) const {
      p.SetX(std::max(xmin, std::min(xmax, p.GetX())));
      p.SetY(std::max(ymin, std::min(ymax, p.GetY())));
      p.SetZ(std::max(zmin, std::min(zmax, p.GetZ())));
      return p;
   }

   CVector3 ObsCenter(const BoxObs& o) const {
      if(!o.dynamic) {
         return o.center;
      }

      Real s = std::sin(o.freq * static_cast<Real>(step) + o.phase);
      return o.center + o.amp * s;
   }

   // -------------------------------------------------------
   // Geometry helpers (mirror obs.py logic)
   // -------------------------------------------------------

   Real Dot3(const CVector3& a, const CVector3& b) const {
      return a.GetX() * b.GetX() + a.GetY() * b.GetY() + a.GetZ() * b.GetZ();
   }

   CVector3 Cross3(const CVector3& a, const CVector3& b) const {
      return CVector3(
         a.GetY() * b.GetZ() - a.GetZ() * b.GetY(),
         a.GetZ() * b.GetX() - a.GetX() * b.GetZ(),
         a.GetX() * b.GetY() - a.GetY() * b.GetX()
      );
   }

   void ProjectPointToPlane(const CVector3& A,
                            const CVector3& B,
                            const CVector3& C,
                            const CVector3& P,
                            CVector3& Q) const {
      CVector3 AB = B - A;
      CVector3 AC = C - A;
      CVector3 n  = Cross3(AB, AC);
      Real n_len_sq = Dot3(n, n);
      if(n_len_sq < 1e-12) { Q = A; return; }
      CVector3 AP = P - A;
      Real t = Dot3(AP, n) / n_len_sq;
      Q = P - n * t;
   }

   bool PointInTriangle(const CVector3& Q,
                        const CVector3& A,
                        const CVector3& B,
                        const CVector3& C) const {
      CVector3 v0 = C - A, v1 = B - A, v2 = Q - A;
      Real dot00 = Dot3(v0, v0), dot01 = Dot3(v0, v1), dot02 = Dot3(v0, v2);
      Real dot11 = Dot3(v1, v1), dot12 = Dot3(v1, v2);
      Real denom = dot00 * dot11 - dot01 * dot01;
      if(std::fabs(denom) < 1e-12) return false;
      Real u = (dot11 * dot02 - dot01 * dot12) / denom;
      Real v = (dot00 * dot12 - dot01 * dot02) / denom;
      return (u >= -1e-12) && (v >= -1e-12) && (u + v <= 1.0 + 1e-12);
   }

   CVector3 ClosestPointOnSegment(const CVector3& P,
                                  const CVector3& U,
                                  const CVector3& V) const {
      CVector3 UV = V - U;
      Real denom = Dot3(UV, UV);
      if(denom < 1e-12) return U;
      Real t = Dot3(P - U, UV) / denom;
      t = std::max<Real>(0.0, std::min<Real>(1.0, t));
      return U + UV * t;
   }

   CVector3 ClosestPointOnTriangle(const CVector3& P,
                                   const CVector3& A,
                                   const CVector3& B,
                                   const CVector3& C) const {
      CVector3 Q;
      ProjectPointToPlane(A, B, C, P, Q);
      if(PointInTriangle(Q, A, B, C)) return Q;
      CVector3 c1 = ClosestPointOnSegment(P, A, B);
      CVector3 c2 = ClosestPointOnSegment(P, B, C);
      CVector3 c3 = ClosestPointOnSegment(P, C, A);
      Real d1 = (P - c1).Length(), d2 = (P - c2).Length(), d3 = (P - c3).Length();
      if(d1 <= d2 && d1 <= d3) return c1;
      if(d2 <= d1 && d2 <= d3) return c2;
      return c3;
   }

   // -------------------------------------------------------
   // Box -> convex polyhedron builder (called once in InitObstacles)
   // -------------------------------------------------------

   void BuildBoxPolyhedron(BoxObs& o) {
      const Real hx = o.half.GetX(), hy = o.half.GetY(), hz = o.half.GetZ();
      const CVector3 c = o.center;

      o.vertices = {
         CVector3(c.GetX()-hx, c.GetY()-hy, c.GetZ()-hz), // 0
         CVector3(c.GetX()+hx, c.GetY()-hy, c.GetZ()-hz), // 1
         CVector3(c.GetX()+hx, c.GetY()+hy, c.GetZ()-hz), // 2
         CVector3(c.GetX()-hx, c.GetY()+hy, c.GetZ()-hz), // 3
         CVector3(c.GetX()-hx, c.GetY()-hy, c.GetZ()+hz), // 4
         CVector3(c.GetX()+hx, c.GetY()-hy, c.GetZ()+hz), // 5
         CVector3(c.GetX()+hx, c.GetY()+hy, c.GetZ()+hz), // 6
         CVector3(c.GetX()-hx, c.GetY()+hy, c.GetZ()+hz)  // 7
      };

      o.faces = {
         {0,1,2}, {0,2,3}, // bottom
         {4,6,5}, {4,7,6}, // top
         {0,5,1}, {0,4,5}, // y-
         {3,2,6}, {3,6,7}, // y+
         {0,3,7}, {0,7,4}, // x-
         {1,5,6}, {1,6,2}  // x+
      };

      o.face_centroids.clear();
      o.face_normals.clear();
      o.face_centroids.reserve(o.faces.size());
      o.face_normals.reserve(o.faces.size());

      for(const auto& f : o.faces) {
         const CVector3& A = o.vertices[f[0]];
         const CVector3& B = o.vertices[f[1]];
         const CVector3& C = o.vertices[f[2]];
         CVector3 centroid = (A + B + C) / 3.0;
         CVector3 n = NormalizeSafe3D(Cross3(B - A, C - A));
         // Ensure normal points outward
         if(Dot3(c - centroid, n) > 0.0) n = n * -1.0;
         o.face_centroids.push_back(centroid);
         o.face_normals.push_back(n);
      }
   }

   // -------------------------------------------------------
   // Translation helper for dynamic obstacles (shift at query time)
   // -------------------------------------------------------

   void TranslatedObstacleGeometry(const BoxObs& o,
                                   std::vector<CVector3>& vertices,
                                   std::vector<CVector3>& face_centroids,
                                   std::vector<CVector3>& face_normals) const {
      CVector3 shift = ObsCenter(o) - o.center;

      vertices = o.vertices;
      for(auto& v : vertices) v += shift;

      face_centroids = o.face_centroids;
      for(auto& fc : face_centroids) fc += shift;

      face_normals = o.face_normals; // normals unchanged by translation
   }

   // -------------------------------------------------------
   // Convex-polyhedron closest-point (mirrors obs.py)
   // -------------------------------------------------------

   void ClosestPointOnConvexPolyhedron(const CVector3& P,
                                       const std::vector<CVector3>& vertices,
                                       const std::vector<std::array<int,3>>& faces,
                                       CVector3& p_star,
                                       Real& phi,
                                       CVector3& n_hat,
                                       const std::vector<CVector3>* fc_ptr = nullptr,
                                       const std::vector<CVector3>* fn_ptr = nullptr) const {
      std::vector<CVector3> local_fc, local_fn;

      if(fc_ptr == nullptr || fn_ptr == nullptr) {
         local_fc.reserve(faces.size());
         local_fn.reserve(faces.size());
         for(const auto& f : faces) {
            const CVector3& A = vertices[f[0]], &B = vertices[f[1]], &C = vertices[f[2]];
            local_fc.push_back((A + B + C) / 3.0);
            local_fn.push_back(NormalizeSafe3D(Cross3(B - A, C - A)));
         }
         fc_ptr = &local_fc;
         fn_ptr = &local_fn;
      }

      Real best_dist = std::numeric_limits<Real>::max();
      CVector3 best_point(0, 0, 0);

      // First pass: only facing faces
      for(size_t idx = 0; idx < faces.size(); ++idx) {
         if(Dot3(P - (*fc_ptr)[idx], (*fn_ptr)[idx]) <= 0.0) continue;
         const auto& f = faces[idx];
         CVector3 Q = ClosestPointOnTriangle(P, vertices[f[0]], vertices[f[1]], vertices[f[2]]);
         Real d = (P - Q).Length();
         if(d < best_dist) { best_dist = d; best_point = Q; }
      }

      // Fallback: check all faces (handles interior/degenerate case)
      if(best_dist == std::numeric_limits<Real>::max()) {
         for(size_t idx = 0; idx < faces.size(); ++idx) {
            const auto& f = faces[idx];
            CVector3 Q = ClosestPointOnTriangle(P, vertices[f[0]], vertices[f[1]], vertices[f[2]]);
            Real d = (P - Q).Length();
            if(d < best_dist) { best_dist = d; best_point = Q; }
         }
      }

      p_star = best_point;
      phi    = best_dist;
      n_hat  = (phi < 1e-12) ? CVector3(0, 0, 0) : NormalizeSafe3D(P - p_star);
   }

   Real ClosestPointObstacleDistance(const CVector3& p,
                                     const BoxObs& o,
                                     CVector3& p_star,
                                     CVector3& n_hat) const {
      std::vector<CVector3> vertices, face_centroids, face_normals;
      TranslatedObstacleGeometry(o, vertices, face_centroids, face_normals);

      Real phi = 0.0;
      ClosestPointOnConvexPolyhedron(p, vertices, o.faces, p_star, phi, n_hat,
                                     &face_centroids, &face_normals);
      return phi;
   }


   CVector3 ObstacleForce(const CVector3& p, Real gain) const {
      CVector3 f(0,0,0);

      for(const auto& o : obstacles) {
         CVector3 p_star, n_hat;
         Real d = ClosestPointObstacleDistance(p, o, p_star, n_hat);

         Real gap = obs_safe - d;
         if(gap <= 0.0) {
            continue;
         }

         Real mag = gain / std::pow(std::max(gap, 1e-4), 2.0);
         f += mag * n_hat;
      }

      return f;
   }

   CVector3 ProjectOutside(CVector3 p) const {
      p = ClampZ(p);

      for(const auto& o : obstacles) {
         CVector3 p_star, n_hat;
         Real d = ClosestPointObstacleDistance(p, o, p_star, n_hat);

         if(d < obs_safe) {
            p = p_star + n_hat * obs_safe;
         }
      }

      return Clamp(p);
   }

   CVector3 CoreCentroid() const {
      CVector3 c(0,0,0);
      int n = 0;

      for(const auto& a : agents) {
         if(a.influential) {
            c += a.pos;
            n++;
         }
      }

      if(n > 0) {
         c /= n;
      }

      c.SetZ(flight_z);
      return c;
   }

   Real MaxInfluentialDistanceFromCore() const {
      CVector3 core = CoreCentroid();
      Real worst = 0.0;

      for(const auto& a : agents) {
         if(a.influential) {
            worst = std::max(worst, XYOnly(a.pos - core).Length());
         }
      }

      return worst;
   }

   CVector3 CurrentTarget() const {
      return goal;
   }

   Real FormationHealthFactor() const {
      Real pinfl = ProtectionInfluentialMean();
      int min_neighbors = MinNeighborCountCurrent();
      Real min_dd = MinDecoyDistance();
      Real max_infl_dist = MaxInfluentialDistanceFromCore();

      Real health = 1.0;

      /*
         Step 3.2:
         Local k-connectivity now affects health earlier.
         This prevents the core from pulling the shell forward while one agent
         is already close to isolation.
      */
      if(min_neighbors <= 1) {
         health *= 0.45;
      }
      else if(min_neighbors <= static_cast<int>(k_conn)) {
         health *= 0.55;
      }
      else if(min_neighbors <= static_cast<int>(k_conn + 1)) {
         health *= 0.75;
      }

      /*
         Keep protection-sensitive slowdown, but do not fully stall.
      */
      if(pinfl < 0.45) {
         health *= 0.65;
      } else if(pinfl < 0.55) {
         health *= 0.80;
      }

      if(min_dd < 0.08) {
         health *= 0.65;
      } else if(min_dd < 0.14) {
         health *= 0.80;
      }

      if(max_infl_dist > 1.10) {
         health *= 0.60;
      } else if(max_infl_dist > max_infl_core_dist) {
         health *= 0.80;
      }

      /*
         Anti-stall floor:
         even when locally weak, the swarm must keep enough motion to escape
         obstacle-induced bad geometry.
      */
      return std::max<Real>(0.35, Clamp01(health));
   }

   void UpdateGoalTracking() {
      if(mission_complete) {
         return;
      }

      CVector3 core = CoreCentroid();
      Real dist = XYOnly(goal - core).Length();

      if(dist < goal_accept_radius) {
         if(IsMissionSafeToComplete()) {
            mission_complete = true;
            LOG << "MISSION COMPLETE: core reached goal and swarm quality is valid. "
                << "dist_goal=" << dist
                << " goal_accept_radius=" << goal_accept_radius
                << std::endl;
         }
      }
   }

   bool IsMissionSafeToComplete() const {
      Real lambda2 = Lambda2FastProxy();
      Real min_dd = MinDecoyDistance();
      Real min_od = MinObstacleDecoyDistance();
      Real pinfl  = ProtectionInfluentialMean();
      int min_neighbors = MinNeighborCountCurrent();
      Real max_infl_dist = MaxInfluentialDistanceFromCore();
      Real vertical_score = VerticalProtectionScore();

      return
         vertical_score > 0.65 &&
         lambda2 > complete_min_lambda2_proxy &&
         NumConnectedComponents() == 1 &&
         min_neighbors >= static_cast<int>(k_conn) &&
         min_dd  > complete_min_decoy_dist &&
         min_od  > complete_min_obs_dist &&
         pinfl   > complete_min_protection &&
         max_infl_dist < complete_max_infl_core_dist;
   }

   void InitShellDirections() {
      shell_dirs.clear();
      int ndec = CountDec();

      if(ndec <= 0) {
         return;
      }

      /*
         Fibonacci sphere directions.
         Directions are deterministic and approximately uniform over the sphere.
         shell_z_scale slightly compresses vertical spread for easier ARGoS visualization
         while still providing upper and lower protection.
      */
      const Real golden_angle = ARGOS_PI * (3.0 - std::sqrt(5.0));

      for(int j = 0; j < ndec; ++j) {
         Real z = 1.0 - 2.0 * (static_cast<Real>(j) + 0.5) / static_cast<Real>(ndec);
         Real r = std::sqrt(std::max<Real>(0.0, 1.0 - z * z));
         Real th = golden_angle * static_cast<Real>(j);

         CVector3 b(r * std::cos(th),
                    r * std::sin(th),
                    shell_z_scale * z);

         b = NormalizeSafe3D(b);
         shell_dirs.push_back(b);
      }

      LOG << "Initialized " << shell_dirs.size()
          << " Fibonacci shell directions for 3D protection." << std::endl;
   }

   void InitDefinedFormation() {
      /*
         4 influential agents in a compact diamond-like core.
      */
      std::vector<CVector3> core_offsets = {
         CVector3( 0.00,  0.00, 0.00),
         CVector3( 0.30,  0.00, 0.00),
         CVector3(-0.30,  0.00, 0.00),
         CVector3( 0.00,  0.30, 0.00)
      };

      int i = 0;
      for(auto& a : agents) {
         if(!a.influential) continue;

         a.pos = SetFlightZ(start + core_offsets[i % core_offsets.size()]);
         a.vel = CVector3(0,0,0);
         i++;
      }

      int ndec = CountDec();
      int j = 0;

      for(auto& a : agents) {
         if(a.influential) continue;

         CVector3 dir(1,0,0);
         if(j >= 0 && j < static_cast<int>(shell_dirs.size())) {
            dir = shell_dirs[j];
         }

         a.pos = Clamp(start + dir * shell_radius);
         a.vel = CVector3(0,0,0);
         a.slot = j;
         j++;
      }

      ApplyMovement();
      ApplyObstacleVisuals();

      initialized = true;

      LOG << "Defined 4-influential + 26-decoy 3D protection shell initialized around z="
          << flight_z << std::endl;
   }

   CVector3 InfluentialSeparation(const Agent& a) const {
      CVector3 f(0,0,0);

      for(const auto& b : agents) {
         if(!b.influential || a.id == b.id) continue;

         CVector3 d = XYOnly(a.pos - b.pos);
         Real L = d.Length();

         if(L < infl_sep_radius && L > 1e-6) {
            f += NormalizeSafeXY(d) * ((infl_sep_radius - L) / infl_sep_radius);
         }
      }

      return f;
   }

   CVector3 DecoySeparationFromPositions(const CVector3& y,
                                         const std::vector<CVector3>& decoys,
                                         int self) const {
      CVector3 f(0,0,0);

      for(size_t k = 0; k < decoys.size(); ++k) {
         if(static_cast<int>(k) == self) continue;

         CVector3 d = y - decoys[k];
         Real L = d.Length();

         if(L < dec_sep_radius && L > 1e-6) {
            f += NormalizeSafe3D(d) * ((dec_sep_radius - L) / dec_sep_radius);
         }
      }

      return f;
   }

   CVector3 BandForceNearestInfluentialFromPositions(const CVector3& y,
                                                     const std::vector<CVector3>& infls) const {
      Real best = std::numeric_limits<Real>::max();
      CVector3 nearest(0,0,flight_z);

      for(const auto& x : infls) {
         Real d = (y - x).Length();

         if(d < best) {
            best = d;
            nearest = x;
         }
      }

      CVector3 radial = y - nearest;
      Real d = radial.Length();

      if(d < 1e-6) {
         return CVector3(0.05,0,0);
      }

      CVector3 u = radial / d;

      if(d < rmin) {
         Real err = rmin - d;
         return u * err * std::exp(std::min(err / tau, 4.0));
      }

      if(d > rmax) {
         Real err = d - rmax;
         return -u * err * std::exp(std::min(err / tau, 4.0));
      }

      return CVector3(0,0,0);
   }

   void UpdateInfluentials() {
      if(mission_complete) {
         for(auto& a : agents) {
            if(a.influential) {
               a.vel = CVector3(0,0,0);
            }
         }
         return;
      }

      CVector3 core = CoreCentroid();
      CVector3 target = CurrentTarget();

      CVector3 to_target = XYOnly(target - core);
      CVector3 target_dir = NormalizeSafeXY(to_target);

      Real dist = to_target.Length();
      Real slow = std::min(1.0, dist / 1.5);

      /*
         Main behavior:
         core waits/slows if decoys are not protecting correctly
         or if one influential is separating from the core.
      */
      slow *= FormationHealthFactor();

      for(auto& a : agents) {
         if(!a.influential) continue;

         CVector3 f_goal = target_dir * (k_goal_core * slow);
         CVector3 f_sep = InfluentialSeparation(a) * k_infl_sep;
         CVector3 f_cohesion = XYOnly(core - a.pos) * k_core_cohesion;
         CVector3 f_obs = ObstacleForce(a.pos, k_obs_infl);

         CVector3 f = f_goal + f_sep + f_cohesion + f_obs;

         a.vel = LimitXY(f, infl_step_max);
         a.pos = ProjectOutside(a.pos + a.vel);

         /*
            NEW:
            Influential-core rescue.
            If any influential drifts too far from the core, pull it back.
         */
         Real dcore = XYOnly(a.pos - core).Length();

         if(dcore > max_infl_core_dist) {
            CVector3 rescue = LimitXY(core - a.pos, k_infl_core_rescue);
            a.pos = ProjectOutside(a.pos + rescue);
            a.vel *= 0.25;
         }
      }
   }

   CVector3 MeanPosition(const std::vector<CVector3>& pts) const {
      if(pts.empty()) {
         return CVector3(0,0,flight_z);
      }

      CVector3 c(0,0,0);

      for(const auto& p : pts) {
         c += p;
      }

      c /= static_cast<Real>(pts.size());
      c.SetZ(flight_z);

      return c;
   }

   CVector3 SlotTargetByIndex(int dec_idx, const CVector3& core) const {
      int ndec = CountDec();

      if(ndec <= 0) {
         return core;
      }

      CVector3 dir(1,0,0);
      if(dec_idx >= 0 && dec_idx < static_cast<int>(shell_dirs.size())) {
         dir = shell_dirs[dec_idx];
      } else {
         Real th = 2.0 * ARGOS_PI * static_cast<Real>(dec_idx) / std::max(1, ndec);
         dir = NormalizeSafe3D(CVector3(std::cos(th), std::sin(th), 0.0));
      }

      return Clamp(core + dir * shell_radius);
   }

   CVector3 SlotTarget(const Agent& a, const CVector3& core) const {
      return SlotTargetByIndex(a.slot, core);
   }

   CVector3 DecoyForceWithChromosome(const CVector3& y,
                                     const CVector3& v,
                                     int dec_idx,
                                     const Chromosome& c,
                                     const std::vector<CVector3>& infls,
                                     const std::vector<CVector3>& decoys,
                                     const CVector3& slot_target,
                                     const CVector3& core_target_dir) const {
      CVector3 core = MeanPosition(infls);

      CVector3 f_cent = XYOnly(core - y);
      CVector3 f_sep  = DecoySeparationFromPositions(y, decoys, dec_idx);
      CVector3 f_band = BandForceNearestInfluentialFromPositions(y, infls);
      CVector3 f_obs  = ObstacleForce(y, c.wobs);

      /*
         Strong assigned-slot force.
         This is the dominant protection-preserving force.
      */
      CVector3 f_slot = (slot_target - y) * shell_slot_gain;

      CVector3 radial = NormalizeSafeXY(y - core);
      CVector3 tangent(-radial.GetY(), radial.GetX(), 0.0);
      CVector3 f_tangent = tangent * k_tangent;

      /*
         GA tunes local behavior.
         Slot force and band force keep protection dominant.
      */
      CVector3 f =
         f_cent * (0.25 * c.wcent) +
         f_sep  * c.wsep +
         f_band * (1.10 * c.wband) +
         f_obs +
         core_target_dir * 0.006 +
         f_slot +
         f_tangent;

      return Limit3D(v + f, c.s);
   }

   Real BandMembership(Real d) const {
      Real delta = 0.0;

      if(d < rmin) {
         delta = rmin - d;
      } else if(d > rmax) {
         delta = d - rmax;
      }

      return std::exp(-delta / std::max(tau, 1e-6));
   }

   Real CandidateProtectionReward(const CVector3& y, const std::vector<CVector3>& infls) const {
      Real best = 0.0;

      for(const auto& x : infls) {
         Real d = (y - x).Length();
         best = std::max(best, BandMembership(d));
      }

      return best;
   }

   Real CandidateObstaclePenalty(const CVector3& y) const {
      Real pen = 0.0;

      for(const auto& o : obstacles) {
         CVector3 p_star, n_hat;
         Real phi = ClosestPointObstacleDistance(y, o, p_star, n_hat) - obs_safe;

         if(phi < 0.0) {
            pen += 5.0 + (-phi * 10.0);
         }
         else if(phi < obs_influence) {
            pen += (obs_influence - phi) / obs_influence;
         }
      }

      return pen;
   }

   Real CandidateConnectivityReward(const std::vector<CVector3>& infls,
                                    const std::vector<CVector3>& decoys) const {
      std::vector<CVector3> pts;

      for(const auto& x : infls) pts.push_back(x);
      for(const auto& y : decoys) pts.push_back(y);

      int n = static_cast<int>(pts.size());
      if(n < 2) return 0.0;

      int edges = 0;

      for(int i = 0; i < n; ++i) {
         for(int j = i + 1; j < n; ++j) {
            if((pts[i] - pts[j]).Length() <= graph_radius) {
               edges++;
            }
         }
      }

      return Clamp01(static_cast<Real>(edges) / static_cast<Real>(n * 3));
   }

   int NeighborCountInPoints(const std::vector<CVector3>& pts,
                             int self_idx,
                             Real radius) const {
      if(self_idx < 0 || self_idx >= static_cast<int>(pts.size())) {
         return 0;
      }

      int count = 0;

      for(int i = 0; i < static_cast<int>(pts.size()); ++i) {
         if(i == self_idx) {
            continue;
         }

         if((pts[self_idx] - pts[i]).Length() <= radius) {
            count++;
         }
      }

      return count;
   }

   Real CandidateConnectivityBreakPenalty(const std::vector<CVector3>& infls,
                                          const std::vector<CVector3>& decoys,
                                          int dec_idx) const {
      std::vector<CVector3> pts;

      for(const auto& x : infls) {
         pts.push_back(x);
      }

      for(const auto& y : decoys) {
         pts.push_back(y);
      }

      int self_idx = static_cast<int>(infls.size()) + dec_idx;
      int neighbors = NeighborCountInPoints(pts, self_idx, graph_radius);

      if(neighbors >= static_cast<int>(k_conn)) {
         return 0.0;
      }

      Real missing = static_cast<Real>(k_conn) - static_cast<Real>(neighbors);
      return missing / std::max<Real>(1.0, static_cast<Real>(k_conn));
   }

   Real CandidateSlotReward(const CVector3& y, const CVector3& slot) const {
      Real d = (y - slot).Length();
      return std::exp(-d / 0.65);
   }

   std::vector<Real> StructuralScoresFromPoints(const std::vector<CVector3>& pts) const {
      int n = static_cast<int>(pts.size());

      std::vector<Real> S(n, 0.0);

      if(n <= 1) {
         return S;
      }

      std::vector<std::vector<int>> A(n, std::vector<int>(n, 0));
      std::vector<std::vector<Real>> Dmat(n, std::vector<Real>(n, 0.0));

      for(int i = 0; i < n; ++i) {
         for(int j = i + 1; j < n; ++j) {
            Real d = XYOnly(pts[i] - pts[j]).Length();

            Dmat[i][j] = d;
            Dmat[j][i] = d;

            if(d <= graph_radius) {
               A[i][j] = 1;
               A[j][i] = 1;
            }
         }
      }

      std::vector<Real> deg(n, 0.0);
      for(int i = 0; i < n; ++i) {
         for(int j = 0; j < n; ++j) {
            deg[i] += A[i][j];
         }
      }

      /*
         Lightweight dominant eigenvector approximation.
      */
      std::vector<Real> eig(n, 1.0 / static_cast<Real>(n));

      for(int it = 0; it < 20; ++it) {
         std::vector<Real> next(n, 0.0);

         for(int i = 0; i < n; ++i) {
            for(int j = 0; j < n; ++j) {
               next[i] += static_cast<Real>(A[i][j]) * eig[j];
            }
         }

         Real L = 0.0;
         for(int i = 0; i < n; ++i) {
            L += next[i] * next[i];
         }

         L = std::sqrt(std::max(L, 1e-12));

         for(int i = 0; i < n; ++i) {
            eig[i] = std::fabs(next[i] / L);
         }
      }

      std::vector<Real> uniformity(n, 0.0);
      std::vector<Real> angular(n, 0.0);

      for(int i = 0; i < n; ++i) {
         std::vector<Real> lens;
         std::vector<Real> angles;

         for(int j = 0; j < n; ++j) {
            if(A[i][j]) {
               lens.push_back(Dmat[i][j]);

               CVector3 r = XYOnly(pts[j] - pts[i]);
               angles.push_back(std::atan2(r.GetY(), r.GetX()));
            }
         }

         if(lens.empty()) {
            uniformity[i] = 0.0;
            angular[i] = 0.0;
         } else {
            Real mean = 0.0;
            for(Real l : lens) mean += l;
            mean /= static_cast<Real>(lens.size());

            Real var = 0.0;
            for(Real l : lens) {
               Real e = l - mean;
               var += e * e;
            }
            var /= static_cast<Real>(lens.size());

            Real sigma = std::sqrt(var);
            Real cv = sigma / std::max(mean, 1e-6);

            uniformity[i] = 1.0 / (1.0 + cv);

            if(angles.size() < 2) {
               angular[i] = 0.0;
            } else {
               for(auto& a : angles) {
                  while(a < 0.0) a += 2.0 * ARGOS_PI;
                  while(a >= 2.0 * ARGOS_PI) a -= 2.0 * ARGOS_PI;
               }

               std::sort(angles.begin(), angles.end());

               Real max_gap = 0.0;

               for(size_t k = 0; k < angles.size(); ++k) {
                  Real a1 = angles[k];
                  Real a2 = angles[(k + 1) % angles.size()];

                  Real gap = 0.0;

                  if(k == angles.size() - 1) {
                     gap = (a2 + 2.0 * ARGOS_PI) - a1;
                  } else {
                     gap = a2 - a1;
                  }

                  max_gap = std::max(max_gap, gap);
               }

               angular[i] = 1.0 - max_gap / (2.0 * ARGOS_PI);
            }
         }
      }

      std::vector<Real> deg_n = MinMaxNormalize(deg);
      std::vector<Real> eig_n = MinMaxNormalize(eig);
      std::vector<Real> uni_n = MinMaxNormalize(uniformity);
      std::vector<Real> ang_n = MinMaxNormalize(angular);

      for(int i = 0; i < n; ++i) {
         S[i] =
            0.2 * eig_n[i] +
            0.2 * deg_n[i] +
            0.3 * uni_n[i] +
            0.3 * ang_n[i];
      }

      return S;
   }

   Real MeanInfluentialStructuralScore(const std::vector<CVector3>& infls,
                                       const std::vector<CVector3>& decoys) const {
      std::vector<CVector3> pts;
      std::vector<int> is_infl;

      for(const auto& x : infls) {
         pts.push_back(x);
         is_infl.push_back(1);
      }

      for(const auto& y : decoys) {
         pts.push_back(y);
         is_infl.push_back(0);
      }

      std::vector<Real> S = StructuralScoresFromPoints(pts);

      Real sum = 0.0;
      int count = 0;

      for(size_t i = 0; i < S.size(); ++i) {
         if(is_infl[i]) {
            sum += S[i];
            count++;
         }
      }

      if(count == 0) return 0.0;
      return sum / static_cast<Real>(count);
   }

   Real ProtectionInfluentialMeanFromPoints(const std::vector<CVector3>& infls,
                                             const std::vector<CVector3>& decoys) const {
      if(infls.empty() || decoys.empty()) {
         return 0.0;
      }

      int upper = 0;
      int lower = 0;
      CVector3 core = MeanPosition(infls);

      for(const auto& y : decoys) {
         if(y.GetZ() > core.GetZ() + vertical_band_min) upper++;
         if(y.GetZ() < core.GetZ() - vertical_band_min) lower++;
      }

      int target_upper = std::max(1, static_cast<int>(decoys.size()) / 4);
      int target_lower = std::max(1, static_cast<int>(decoys.size()) / 4);
      Real vertical_score = Clamp01(std::min(
         static_cast<Real>(upper) / static_cast<Real>(target_upper),
         static_cast<Real>(lower) / static_cast<Real>(target_lower)
      ));

      Real total = 0.0;

      for(const auto& x : infls) {
         Real sumB = 0.0;

         for(const auto& y : decoys) {
            Real d = (y - x).Length();
            sumB += BandMembership(d);
         }

         Real meanB = sumB / static_cast<Real>(decoys.size());
         Real gamma3d = SphericalCoverageScore(x, decoys);

         total += 0.65 * meanB + 0.25 * gamma3d + 0.10 * vertical_score;
      }

      return total / static_cast<Real>(infls.size());
   }

   Real ViolateMaximize(Real q, Real threshold) const {
      return std::max<Real>(0.0, threshold - q);
   }

   Real ViolateMinimize(Real q, Real threshold) const {
      return std::max<Real>(0.0, q - threshold);
   }

   bool HasPriorityViolation(const Chromosome& c) const {
      const Real eps = 1e-9;
      return (c.V1 > eps || c.V2 > eps || c.V3 > eps || c.V4 > eps || c.V5 > eps || c.V6 > eps);
   }

   bool BetterChromosome(const Chromosome& a, const Chromosome& b) const {
      bool av = HasPriorityViolation(a);
      bool bv = HasPriorityViolation(b);
      const Real eps = 1e-9;

      if(av || bv) {
         if(std::fabs(a.V1 - b.V1) > eps) return a.V1 < b.V1;
         if(std::fabs(a.V2 - b.V2) > eps) return a.V2 < b.V2;
         if(std::fabs(a.V3 - b.V3) > eps) return a.V3 < b.V3;
         if(std::fabs(a.V4 - b.V4) > eps) return a.V4 < b.V4;
         if(std::fabs(a.V5 - b.V5) > eps) return a.V5 < b.V5;
         if(std::fabs(a.V6 - b.V6) > eps) return a.V6 < b.V6;
         return a.fitness > b.fitness;
      }

      if(a.pareto_rank != b.pareto_rank) {
         return a.pareto_rank < b.pareto_rank;
      }

      if(std::fabs(a.crowding_distance - b.crowding_distance) > eps) {
         return a.crowding_distance > b.crowding_distance;
      }

      return a.fitness > b.fitness;
   }

   bool Dominates(const Chromosome& a, const Chromosome& b) const {
      bool no_worse =
         a.q1 >= b.q1 &&
         a.q2 >= b.q2 &&
         a.q3 >= b.q3 &&
         a.q4 >= b.q4 &&
         a.q5 >= b.q5 &&
         a.q6 <= b.q6;

      bool strictly_better =
         a.q1 > b.q1 ||
         a.q2 > b.q2 ||
         a.q3 > b.q3 ||
         a.q4 > b.q4 ||
         a.q5 > b.q5 ||
         a.q6 < b.q6;

      return no_worse && strictly_better;
   }

   Real ObjectiveValue(const Chromosome& c, int obj) const {
      if(obj == 0) return c.q1;
      if(obj == 1) return c.q2;
      if(obj == 2) return c.q3;
      if(obj == 3) return c.q4;
      if(obj == 4) return c.q5;
      return c.q6;
   }

   void ComputeParetoRankAndCrowding(std::vector<Chromosome>& pop) const {
      const int N = static_cast<int>(pop.size());
      if(N == 0) {
         return;
      }

      std::vector<std::vector<int>> dominates_set(N);
      std::vector<int> domination_count(N, 0);
      std::vector<std::vector<int>> fronts;

      for(int i = 0; i < N; ++i) {
         pop[i].pareto_rank = 999999;
         pop[i].crowding_distance = 0.0;

         for(int j = 0; j < N; ++j) {
            if(i == j) continue;

            if(Dominates(pop[i], pop[j])) {
               dominates_set[i].push_back(j);
            } else if(Dominates(pop[j], pop[i])) {
               domination_count[i]++;
            }
         }

         if(domination_count[i] == 0) {
            pop[i].pareto_rank = 0;
         }
      }

      std::vector<int> current;
      for(int i = 0; i < N; ++i) {
         if(pop[i].pareto_rank == 0) {
            current.push_back(i);
         }
      }

      UInt32 rank = 0;
      while(!current.empty()) {
         fronts.push_back(current);
         std::vector<int> next;

         for(int i : current) {
            for(int j : dominates_set[i]) {
               domination_count[j]--;
               if(domination_count[j] == 0) {
                  pop[j].pareto_rank = rank + 1;
                  next.push_back(j);
               }
            }
         }

         current = next;
         rank++;
      }

      /*
         Crowding distance for each front, NSGA-II style.
         q1 and q2 are maximize objectives; q3 and q4 are minimize objectives.
         For crowding distance, the ordering direction does not matter because
         only spread along each objective axis is needed.
      */
      for(const auto& front : fronts) {
         if(front.empty()) continue;

         if(front.size() <= 2) {
            for(int idx : front) {
               pop[idx].crowding_distance = 1e9;
            }
            continue;
         }

         for(int obj = 0; obj < 6; ++obj) {
            std::vector<int> sorted = front;

            std::sort(sorted.begin(), sorted.end(),
               [&](int a, int b) {
                  return ObjectiveValue(pop[a], obj) < ObjectiveValue(pop[b], obj);
               });

            Real fmin = ObjectiveValue(pop[sorted.front()], obj);
            Real fmax = ObjectiveValue(pop[sorted.back()], obj);
            Real denom = std::max<Real>(fmax - fmin, 1e-9);

            pop[sorted.front()].crowding_distance += 1e9;
            pop[sorted.back()].crowding_distance += 1e9;

            for(size_t k = 1; k + 1 < sorted.size(); ++k) {
               Real prev = ObjectiveValue(pop[sorted[k - 1]], obj);
               Real next = ObjectiveValue(pop[sorted[k + 1]], obj);
               pop[sorted[k]].crowding_distance += (next - prev) / denom;
            }
         }
      }
   }

   Chromosome EvaluateChromosomePriority(int dec_idx,
                                         const CVector3& y0,
                                         const CVector3& v0,
                                         const Chromosome& base,
                                         const std::vector<CVector3>& infls0,
                                         const std::vector<CVector3>& decoys0) const {
      Chromosome c = base;

      std::vector<CVector3> infls = infls0;
      std::vector<CVector3> decoys = decoys0;

      CVector3 y = y0;
      CVector3 v = v0;

      Real q1_min = 1e9;
      Real q2_min = 1e9;
      Real q3_min = 1e9;
      Real q4_sum = 0.0;
      Real q5_sum = 0.0;
      Real q6_sum = 0.0;

      Real reward_sum = 0.0;

      CVector3 target = CurrentTarget();
      CVector3 core0 = MeanPosition(infls);
      CVector3 core_target_dir = NormalizeSafeXY(target - core0);

      for(UInt32 h = 0; h < horizon; ++h) {
         CVector3 core = MeanPosition(infls);
         CVector3 dir = NormalizeSafeXY(target - core);

         for(auto& x : infls) {
            CVector3 f = dir * infl_step_max;
            x = ProjectOutside(x + f);
         }

         CVector3 slot_target = SlotTargetByIndex(dec_idx, MeanPosition(infls));

         v = DecoyForceWithChromosome(y, v, dec_idx, c, infls, decoys, slot_target, core_target_dir);
         y = ProjectOutside(y + v);

         if(dec_idx >= 0 && dec_idx < static_cast<int>(decoys.size())) {
            decoys[dec_idx] = y;
         }

         Real dd_step = 1e9;
         for(size_t i = 0; i < decoys.size(); ++i) {
            if(static_cast<int>(i) == dec_idx) continue;
            dd_step = std::min(dd_step, (y - decoys[i]).Length());
         }

         Real obs_step = 1e9;
         for(const auto& o : obstacles) {
            CVector3 p_star, n_hat;
            obs_step = std::min(obs_step, ClosestPointObstacleDistance(y, o, p_star, n_hat));
         }

         Real l2_step = CandidateConnectivityReward(infls, decoys) * 3.0;

         Real pinfl_reward = ProtectionInfluentialMeanFromPoints(infls, decoys);
         Real popt_decoy = CandidateProtectionReward(y, infls);
         Real prot_step = pinfl_reward + popt_decoy;
         Real str_step = MeanInfluentialStructuralScore(infls, decoys);
         Real mov_step = v.Length() * v.Length();

         q1_min = std::min(q1_min, dd_step);
         q2_min = std::min(q2_min, obs_step);
         q3_min = std::min(q3_min, l2_step);
         q4_sum += prot_step;
         q5_sum += str_step;
         q6_sum += mov_step;

         Real conn_break_pen = CandidateConnectivityBreakPenalty(infls, decoys, dec_idx);
         Real slot_reward = CandidateSlotReward(y, slot_target);

         Real reward =
            fit_prot   * prot_step +
            fit_struct * str_step +
            fit_conn   * l2_step +
            fit_slot   * slot_reward -
            fit_conn_break * conn_break_pen -
            fit_obs    * CandidateObstaclePenalty(y) -
            fit_move   * mov_step;

         reward_sum += reward;
      }

      Real H = static_cast<Real>(std::max<UInt32>(1, horizon));

      c.q1 = q1_min;
      c.q2 = q2_min;
      c.q3 = q3_min;
      c.q4 = q4_sum / H;
      c.q5 = q5_sum / H;
      c.q6 = q6_sum / H;
      c.fitness = reward_sum / H;

      c.V1 = ViolateMaximize(c.q1, T1_dd);
      c.V2 = ViolateMaximize(c.q2, T2_obs);
      c.V3 = ViolateMaximize(c.q3, T3_l2);
      c.V4 = ViolateMaximize(c.q4, T4_prot);
      c.V5 = ViolateMaximize(c.q5, T5_str);
      c.V6 = ViolateMinimize(c.q6, T6_mov);

      return c;
   }

   int TournamentSelectIndex(const std::vector<Chromosome>& pop) const {
      if(pop.empty()) {
         return 0;
      }

      int i = std::rand() % pop.size();
      int j = std::rand() % pop.size();

      if(BetterChromosome(pop[i], pop[j])) {
         return i;
      }

      return j;
   }

   void ReplanAllDecoys() {
      if(mission_complete) {
         return;
      }

      if(step % replan_period != 0) {
         return;
      }

      std::vector<CVector3> infls;
      std::vector<CVector3> decoys;

      for(const auto& a : agents) {
         if(a.influential) infls.push_back(a.pos);
         else decoys.push_back(a.pos);
      }

      int dec_idx = 0;

      for(auto& a : agents) {
         if(a.influential) {
            continue;
         }

         for(auto& c : a.population) {
            c = EvaluateChromosomePriority(dec_idx, a.pos, a.vel, c, infls, decoys);
         }

         ComputeParetoRankAndCrowding(a.population);

         std::sort(a.population.begin(), a.population.end(),
            [this](const Chromosome& p, const Chromosome& q) {
               return BetterChromosome(p, q);
            });

         if(!a.population.empty()) {
            a.active = a.population[0];
         }

         std::vector<Chromosome> next;

         /*
            Elitism: keep best candidate according to priority/Pareto comparator.
         */
         if(a.population.size() >= 1) {
            next.push_back(a.population[0]);
         }

         /*
            Tournament selection now uses the same priority/Pareto comparator.
         */
         while(next.size() < population_size) {
            int i1 = TournamentSelectIndex(a.population);
            int i2 = TournamentSelectIndex(a.population);

            const Chromosome& p1 = a.population[i1];
            const Chromosome& p2 = a.population[i2];

            Chromosome child = Crossover(p1, p2);
            child = MutateChromosome(child);

            next.push_back(child);
         }

         a.population = next;
         dec_idx++;
      }
   }

   void UpdateDecoys() {
      ReplanAllDecoys();

      CVector3 core = CoreCentroid();
      CVector3 target = CurrentTarget();
      CVector3 core_target_dir = NormalizeSafeXY(target - core);

      std::vector<CVector3> infls;
      std::vector<CVector3> decoys;

      for(const auto& a : agents) {
         if(a.influential) infls.push_back(a.pos);
         else decoys.push_back(a.pos);
      }

      int dec_idx = 0;

      for(auto& a : agents) {
         if(a.influential) {
            continue;
         }

         CVector3 slot = SlotTarget(a, core);

         CVector3 vnew = DecoyForceWithChromosome(
            a.pos,
            a.vel,
            dec_idx,
            a.active,
            infls,
            decoys,
            slot,
            core_target_dir
         );

         a.vel = vnew;
         a.pos = ProjectOutside(a.pos + a.vel);

         /*
            Slot rescue:
            If a decoy drifted from its assigned ring sector, pull it back.
         */
         CVector3 to_slot = slot - a.pos;
         Real slot_dist = to_slot.Length();

         if(slot_dist > 0.95) {
            CVector3 rescue = Limit3D(to_slot * 0.40, 0.080);
            a.pos = ProjectOutside(a.pos + rescue);
            a.vel *= 0.25;
         }

         /*
            Core-envelope rescue:
            Prevent disconnected trailing subgroups.
         */
         Real dcore = XYOnly(a.pos - core).Length();

         if(dcore > rmax + 0.65) {
            CVector3 pull = Limit3D(slot - a.pos, 0.090);
            a.pos = ProjectOutside(a.pos + pull);
            a.vel *= 0.20;
         }

         /*
            Adaptive replacement-link rescue:
            If this decoy has fewer than k_conn links, reconnect it in a
            protection-aware way. Slot/core dominate, not nearest clump.
         */
         int n_neighbors = NeighborCountCurrent(a.pos, a.id);
         if(n_neighbors <= static_cast<int>(k_conn + 1)) {
            CVector3 rescue = ConnectivityRescueForce(a.pos, a.id, slot, core, n_neighbors);
            a.pos = ProjectOutside(a.pos + rescue);

            /*
               Urgent cases should damp the old velocity more aggressively
               so the weak agent does not keep drifting away.
            */
            if(n_neighbors <= 1) {
               a.vel *= 0.15;
            }
            else if(n_neighbors <= static_cast<int>(k_conn)) {
               a.vel *= 0.25;
            }
            else {
               a.vel *= 0.35;
            }
         }

         if(dec_idx >= 0 && dec_idx < static_cast<int>(decoys.size())) {
            decoys[dec_idx] = a.pos;
         }

         dec_idx++;
      }
   }

   Real AngularUniformity(const std::vector<Real>& angles_in) const {
      if(angles_in.size() < 2) {
         return 0.0;
      }

      std::vector<Real> angles = angles_in;

      for(auto& a : angles) {
         while(a < 0.0) a += 2.0 * ARGOS_PI;
         while(a >= 2.0 * ARGOS_PI) a -= 2.0 * ARGOS_PI;
      }

      std::sort(angles.begin(), angles.end());

      Real ideal = 2.0 * ARGOS_PI / static_cast<Real>(angles.size());
      Real acc = 0.0;

      for(size_t i = 0; i < angles.size(); ++i) {
         Real a1 = angles[i];
         Real a2 = angles[(i + 1) % angles.size()];

         Real gap = 0.0;

         if(i == angles.size() - 1) {
            gap = (a2 + 2.0 * ARGOS_PI) - a1;
         } else {
            gap = a2 - a1;
         }

         acc += std::fabs((gap - ideal) / std::max(ideal, 1e-6));
      }

      return std::exp(-acc / static_cast<Real>(angles.size()));
   }

   Real SphericalCoverageScore(const CVector3& center,
                               const std::vector<CVector3>& decoys) const {
      if(decoys.empty() || shell_dirs.empty()) {
         return 0.0;
      }

      Real err_sum = 0.0;
      int count = 0;

      for(size_t j = 0; j < decoys.size(); ++j) {
         CVector3 u = NormalizeSafe3D(decoys[j] - center);
         CVector3 b = shell_dirs[j % shell_dirs.size()];
         CVector3 e = u - b;
         err_sum += e.Length() * e.Length();
         count++;
      }

      if(count == 0) {
         return 0.0;
      }

      return std::exp(-err_sum / static_cast<Real>(count));
   }

   Real MeanShellError() const {
      CVector3 core = CoreCentroid();
      Real sum = 0.0;
      int count = 0;

      for(const auto& a : agents) {
         if(a.influential) continue;
         CVector3 slot = SlotTarget(a, core);
         sum += (a.pos - slot).Length();
         count++;
      }

      if(count == 0) return 0.0;
      return sum / static_cast<Real>(count);
   }

   int UpperDecoyCount() const {
      CVector3 core = CoreCentroid();
      int count = 0;
      for(const auto& a : agents) {
         if(!a.influential && a.pos.GetZ() > core.GetZ() + vertical_band_min) {
            count++;
         }
      }
      return count;
   }

   int LowerDecoyCount() const {
      CVector3 core = CoreCentroid();
      int count = 0;
      for(const auto& a : agents) {
         if(!a.influential && a.pos.GetZ() < core.GetZ() - vertical_band_min) {
            count++;
         }
      }
      return count;
   }

   Real VerticalProtectionScore() const {
      int ndec = CountDec();
      if(ndec <= 0) return 0.0;

      int target_upper = std::max(1, ndec / 4);
      int target_lower = std::max(1, ndec / 4);
      Real upper_ratio = static_cast<Real>(UpperDecoyCount()) / static_cast<Real>(target_upper);
      Real lower_ratio = static_cast<Real>(LowerDecoyCount()) / static_cast<Real>(target_lower);

      return Clamp01(std::min(upper_ratio, lower_ratio));
   }

   Real GlobalSphericalCoverageScore() const {
      std::vector<CVector3> decoys;
      CVector3 core = CoreCentroid();
      for(const auto& a : agents) {
         if(!a.influential) decoys.push_back(a.pos);
      }
      return SphericalCoverageScore(core, decoys);
   }

   std::vector<Real> ProtectionDiagnosticVector() const {
      std::vector<CVector3> infls;
      std::vector<CVector3> decoys;

      for(const auto& a : agents) {
         if(a.influential) infls.push_back(a.pos);
         else decoys.push_back(a.pos);
      }

      std::vector<Real> P;
      Real vertical_score = VerticalProtectionScore();

      /*
         Step-2 3D influential protection:
         Band membership is fully 3D and angular coverage is replaced by
         spherical shell coverage. This adds upper/lower protection.
      */
      for(const auto& x : infls) {
         Real sumB = 0.0;

         for(const auto& y : decoys) {
            Real d = (y - x).Length();
            sumB += BandMembership(d);
         }

         Real meanB = decoys.empty() ? 0.0 : sumB / static_cast<Real>(decoys.size());
         Real gamma3d = SphericalCoverageScore(x, decoys);

         P.push_back(0.65 * meanB + 0.25 * gamma3d + 0.10 * vertical_score);
      }

      /*
         Decoy diagnostic protection remains symmetric but now uses 3D distances.
         The angular term is kept planar for diagnostics only; the main influential
         protection score above is the 3D shell score.
      */
      std::vector<CVector3> allpts;

      for(const auto& x : infls) allpts.push_back(x);
      for(const auto& y : decoys) allpts.push_back(y);

      for(size_t j = 0; j < decoys.size(); ++j) {
         CVector3 y = decoys[j];

         Real sumB = 0.0;
         int count = 0;
         std::vector<Real> angles;

         for(size_t q = 0; q < allpts.size(); ++q) {
            CVector3 r3 = allpts[q] - y;
            Real d = r3.Length();

            if(d < 1e-8) {
               continue;
            }

            sumB += BandMembership(d);
            CVector3 r = XYOnly(r3);
            angles.push_back(std::atan2(r.GetY(), r.GetX()));
            count++;
         }

         Real meanB = (count == 0) ? 0.0 : sumB / static_cast<Real>(count);
         Real gamma = AngularUniformity(angles);

         P.push_back(0.8 * meanB + 0.2 * gamma);
      }

      return P;
   }

   Real ProtectionInfluentialMean() const {
      std::vector<Real> P = ProtectionDiagnosticVector();
      int m = CountInfl();

      if(m <= 0 || P.empty()) {
         return 0.0;
      }

      Real sum = 0.0;
      int count = std::min<int>(m, static_cast<int>(P.size()));

      for(int i = 0; i < count; ++i) {
         sum += P[i];
      }

      return sum / static_cast<Real>(count);
   }

   std::vector<Real> StructuralScores() const {
      std::vector<CVector3> pts;

      for(const auto& a : agents) {
         pts.push_back(a.pos);
      }

      return StructuralScoresFromPoints(pts);
   }

   std::vector<Real> MinMaxNormalize(const std::vector<Real>& x) const {
      std::vector<Real> y(x.size(), 0.0);

      if(x.empty()) {
         return y;
      }

      Real mn = *std::min_element(x.begin(), x.end());
      Real mx = *std::max_element(x.begin(), x.end());

      Real den = mx - mn;

      if(den < 1e-9) {
         return y;
      }

      for(size_t i = 0; i < x.size(); ++i) {
         y[i] = (x[i] - mn) / den;
      }

      return y;
   }

   Real Percentile(std::vector<Real> v, Real p) const {
      if(v.empty()) {
         return 0.0;
      }

      std::sort(v.begin(), v.end());

      Real idx = p * static_cast<Real>(v.size() - 1);
      int i0 = static_cast<int>(std::floor(idx));
      int i1 = static_cast<int>(std::ceil(idx));

      if(i0 < 0) i0 = 0;
      if(i1 < 0) i1 = 0;
      if(i0 >= static_cast<int>(v.size())) i0 = static_cast<int>(v.size()) - 1;
      if(i1 >= static_cast<int>(v.size())) i1 = static_cast<int>(v.size()) - 1;

      Real t = idx - static_cast<Real>(i0);

      return v[i0] * (1.0 - t) + v[i1] * t;
   }

   Real Mean(const std::vector<Real>& v) const {
      if(v.empty()) {
         return 0.0;
      }

      Real s = 0.0;
      for(Real x : v) {
         s += x;
      }

      return s / static_cast<Real>(v.size());
   }

   int NeighborCountCurrent(const CVector3& p,
                            const std::string& self_id) const {
      int count = 0;

      for(const auto& b : agents) {
         if(b.id == self_id) {
            continue;
         }

         if((p - b.pos).Length() <= graph_radius) {
            count++;
         }
      }

      return count;
   }

   int MinNeighborCountCurrent() const {
      if(agents.empty()) {
         return 0;
      }

      int best = 1000000;

      for(const auto& a : agents) {
         int c = NeighborCountCurrent(a.pos, a.id);
         if(c < best) {
            best = c;
         }
      }

      if(best == 1000000) {
         return 0;
      }

      return best;
   }

   CVector3 ConnectivityRescueForce(const CVector3& p,
                                    const std::string& self_id,
                                    const CVector3& slot,
                                    const CVector3& core,
                                    int n_neighbors) const {
      std::vector<std::pair<Real, CVector3>> near_agents;

      for(const auto& b : agents) {
         if(b.id == self_id) {
            continue;
         }

         Real d = (p - b.pos).Length();
         near_agents.push_back(std::make_pair(d, b.pos));
      }

      if(near_agents.empty()) {
         return Limit3D(core - p, k_conn_rescue);
      }

      std::sort(near_agents.begin(), near_agents.end(),
         [](const std::pair<Real, CVector3>& a,
            const std::pair<Real, CVector3>& b) {
            return a.first < b.first;
         });

      CVector3 neighbor_centroid(0,0,0);
      int used = 0;

      /*
         Use slightly more nearby agents than k_conn so that repair creates
         redundant replacement links, not only one fragile bridge.
      */
      int max_use = std::min<int>(static_cast<int>(k_conn + 2),
                                  static_cast<int>(near_agents.size()));

      for(int i = 0; i < max_use; ++i) {
         neighbor_centroid += near_agents[i].second;
         used++;
      }

      if(used > 0) {
         neighbor_centroid /= static_cast<Real>(used);
      } else {
         neighbor_centroid = core;
      }

      neighbor_centroid = ClampZ(neighbor_centroid);

      /*
         Step 3.2 urgent local-k repair:
         - if a decoy is almost isolated, nearest-neighbor centroid dominates
         - if it is only slightly weak, slot/core still dominate
         This keeps the repair non-rigid and replacement-link based.
      */
      Real w_slot = 0.55;
      Real w_core = 0.20;
      Real w_near = 0.25;
      Real step_max = k_conn_rescue;

      if(n_neighbors <= 1) {
         w_slot = 0.25;
         w_core = 0.10;
         w_near = 0.65;
         step_max = 0.240;
      }
      else if(n_neighbors <= static_cast<int>(k_conn)) {
         w_slot = 0.38;
         w_core = 0.15;
         w_near = 0.47;
         step_max = 0.190;
      }
      else if(n_neighbors <= static_cast<int>(k_conn + 1)) {
         w_slot = 0.48;
         w_core = 0.18;
         w_near = 0.34;
         step_max = 0.150;
      }

      CVector3 desired =
         slot              * w_slot +
         core              * w_core +
         neighbor_centroid * w_near;

      desired = ClampZ(desired);

      return Limit3D(desired - p, step_max);
   }

   std::vector<std::vector<int>> ConnectedComponents() const {
      const int n = static_cast<int>(agents.size());
      std::vector<std::vector<int>> comps;
      std::vector<int> visited(n, 0);

      for(int start_idx = 0; start_idx < n; ++start_idx) {
         if(visited[start_idx]) {
            continue;
         }

         std::vector<int> comp;
         std::vector<int> stack;
         stack.push_back(start_idx);
         visited[start_idx] = 1;

         while(!stack.empty()) {
            int u = stack.back();
            stack.pop_back();
            comp.push_back(u);

            for(int v = 0; v < n; ++v) {
               if(u == v || visited[v]) {
                  continue;
               }

               Real d = (agents[u].pos - agents[v].pos).Length();
               if(d <= graph_radius) {
                  visited[v] = 1;
                  stack.push_back(v);
               }
            }
         }

         comps.push_back(comp);
      }

      return comps;
   }

   int NumConnectedComponents() const {
      return static_cast<int>(ConnectedComponents().size());
   }

   Real LargestComponentFraction() const {
      const int n = static_cast<int>(agents.size());
      if(n <= 0) {
         return 0.0;
      }

      auto comps = ConnectedComponents();
      int best = 0;

      for(const auto& c : comps) {
         best = std::max(best, static_cast<int>(c.size()));
      }

      return static_cast<Real>(best) / static_cast<Real>(n);
   }

   std::vector<int> DegreeVectorCurrent() const {
      const int n = static_cast<int>(agents.size());
      std::vector<int> degree(n, 0);

      for(int i = 0; i < n; ++i) {
         for(int j = i + 1; j < n; ++j) {
            Real d = (agents[i].pos - agents[j].pos).Length();
            if(d <= graph_radius) {
               degree[i]++;
               degree[j]++;
            }
         }
      }

      return degree;
   }

   void LocalKConnectivityRepair() {
      /*
         Step 3.2:
         This repair works even when the whole graph is still one component.
         It prevents locally fragile agents from reaching degree 1 or 0.
         The identity of the links is non-rigid: the agent is pulled toward
         whichever nearby non-neighbor can restore redundant links fastest.
      */
      std::vector<int> degree = DegreeVectorCurrent();
      const int n = static_cast<int>(agents.size());

      for(int i = 0; i < n; ++i) {
         if(degree[i] > static_cast<int>(k_conn + 1)) {
            continue;
         }

         Real best_d = std::numeric_limits<Real>::max();
         int best_j = -1;

         /*
            Prefer the closest agent that is not already a neighbor.
            This creates a new replacement link instead of only shortening
            existing links.
         */
         for(int j = 0; j < n; ++j) {
            if(i == j) {
               continue;
            }

            Real d = (agents[i].pos - agents[j].pos).Length();

            if(d <= graph_radius * 0.92) {
               continue;
            }

            if(d < best_d) {
               best_d = d;
               best_j = j;
            }
         }

         /*
            If every nearby agent is already inside the graph radius, fall back
            to the nearest agent. This can still strengthen weak local geometry.
         */
         if(best_j < 0) {
            for(int j = 0; j < n; ++j) {
               if(i == j) {
                  continue;
               }

               Real d = (agents[i].pos - agents[j].pos).Length();
               if(d < best_d) {
                  best_d = d;
                  best_j = j;
               }
            }
         }

         if(best_j < 0) {
            continue;
         }

         Real pull_i_mag = 0.070;
         Real pull_j_mag = 0.025;

         if(degree[i] <= 1) {
            pull_i_mag = 0.170;
            pull_j_mag = 0.060;
         }
         else if(degree[i] <= static_cast<int>(k_conn)) {
            pull_i_mag = 0.120;
            pull_j_mag = 0.045;
         }

         CVector3 pi = agents[i].pos;
         CVector3 pj = agents[best_j].pos;

         CVector3 pull_i = Limit3D(pj - pi, pull_i_mag);
         CVector3 pull_j = Limit3D(pi - pj, pull_j_mag);

         agents[i].pos = ProjectOutside(agents[i].pos + pull_i);
         agents[best_j].pos = ProjectOutside(agents[best_j].pos + pull_j);

         if(agents[i].influential) {
            agents[i].pos = SetFlightZ(agents[i].pos);
         } else {
            agents[i].pos = Clamp(agents[i].pos);
         }

         if(agents[best_j].influential) {
            agents[best_j].pos = SetFlightZ(agents[best_j].pos);
         } else {
            agents[best_j].pos = Clamp(agents[best_j].pos);
         }

         agents[i].vel *= 0.20;
         agents[best_j].vel *= 0.70;

         last_bridge_pairs++;
         last_bridge_distance = std::max(last_bridge_distance, best_d);
      }
   }

   void ComponentBridgeRepair() {
      auto comps = ConnectedComponents();

      last_bridge_distance = 0.0;
      last_bridge_pairs = 0;

      if(comps.size() <= 1) {
         return;
      }

      /*
         Choose the largest component as the anchor.
         Every smaller component is connected back to it through the closest pair.
         This is non-rigid: the pair can change every timestep.
      */
      int largest_idx = 0;
      for(int c = 1; c < static_cast<int>(comps.size()); ++c) {
         if(comps[c].size() > comps[largest_idx].size()) {
            largest_idx = c;
         }
      }

      const auto& anchor = comps[largest_idx];

      for(int c = 0; c < static_cast<int>(comps.size()); ++c) {
         if(c == largest_idx) {
            continue;
         }

         Real best_d = std::numeric_limits<Real>::max();
         int best_i = -1;
         int best_j = -1;

         for(int i : comps[c]) {
            for(int j : anchor) {
               Real d = (agents[i].pos - agents[j].pos).Length();
               if(d < best_d) {
                  best_d = d;
                  best_i = i;
                  best_j = j;
               }
            }
         }

         if(best_i < 0 || best_j < 0) {
            continue;
         }

         CVector3 pi = agents[best_i].pos;
         CVector3 pj = agents[best_j].pos;

         CVector3 pull_i = Limit3D(pj - pi, k_component_bridge);
         CVector3 pull_j = Limit3D(pi - pj, k_component_bridge * 0.65);

         agents[best_i].pos = ProjectOutside(agents[best_i].pos + pull_i);
         agents[best_j].pos = ProjectOutside(agents[best_j].pos + pull_j);

         if(agents[best_i].influential) {
            agents[best_i].pos = SetFlightZ(agents[best_i].pos);
         } else {
            agents[best_i].pos = Clamp(agents[best_i].pos);
         }

         if(agents[best_j].influential) {
            agents[best_j].pos = SetFlightZ(agents[best_j].pos);
         } else {
            agents[best_j].pos = Clamp(agents[best_j].pos);
         }

         agents[best_i].vel *= 0.35;
         agents[best_j].vel *= 0.55;

         last_bridge_pairs++;
         last_bridge_distance = std::max(last_bridge_distance, best_d);
      }
   }

   Real Lambda2FastProxy() const {
      std::vector<CVector3> pts;

      for(const auto& a : agents) {
         pts.push_back(a.pos);
      }

      const int n = static_cast<int>(pts.size());
      if(n < 2) {
         return 0.0;
      }

      std::vector<std::vector<int>> A(n, std::vector<int>(n, 0));

      for(int i = 0; i < n; ++i) {
         for(int j = i + 1; j < n; ++j) {
            Real d = (pts[i] - pts[j]).Length();

            if(d <= graph_radius) {
               A[i][j] = 1;
               A[j][i] = 1;
            }
         }
      }

      std::vector<int> visited(n, 0);
      std::vector<int> stack;

      stack.push_back(0);
      visited[0] = 1;

      while(!stack.empty()) {
         int u = stack.back();
         stack.pop_back();

         for(int v = 0; v < n; ++v) {
            if(A[u][v] && !visited[v]) {
               visited[v] = 1;
               stack.push_back(v);
            }
         }
      }

      for(int i = 0; i < n; ++i) {
         if(!visited[i]) {
            return 0.0;
         }
      }

      Real deg_sum = 0.0;

      for(int i = 0; i < n; ++i) {
         Real deg = 0.0;
         for(int j = 0; j < n; ++j) {
            deg += A[i][j];
         }
         deg_sum += deg;
      }

      /*
         Not exact lambda2. Fast live connectivity proxy.
         0 = disconnected. Positive = connected.
      */
      return deg_sum / static_cast<Real>(n);
   }

   Real AvgBandError() const {
      Real sum = 0.0;
      int count = 0;

      for(const auto& a : agents) {
         if(a.influential) continue;

         Real best = std::numeric_limits<Real>::max();

         for(const auto& b : agents) {
            if(!b.influential) continue;
            best = std::min(best, (a.pos - b.pos).Length());
         }

         Real err = 0.0;

         if(best < rmin) {
            err = rmin - best;
         } else if(best > rmax) {
            err = best - rmax;
         }

         sum += err;
         count++;
      }

      if(count == 0) return 0.0;
      return sum / count;
   }

   Real MinDecoyDistance() const {
      Real best = std::numeric_limits<Real>::max();

      for(size_t i = 0; i < agents.size(); ++i) {
         if(agents[i].influential) continue;

         for(size_t j = i + 1; j < agents.size(); ++j) {
            if(agents[j].influential) continue;
            best = std::min(best, (agents[i].pos - agents[j].pos).Length());
         }
      }

      if(best == std::numeric_limits<Real>::max()) return 0.0;
      return best;
   }

   Real MinObstacleDecoyDistance() const {
      Real best = std::numeric_limits<Real>::max();

      for(const auto& a : agents) {
         if(a.influential) continue;

         for(const auto& o : obstacles) {
            CVector3 p_star, n_hat;
            best = std::min(best, ClosestPointObstacleDistance(a.pos, o, p_star, n_hat));
         }
      }

      if(best == std::numeric_limits<Real>::max()) return 0.0;
      return best;
   }

   Real Clamp01(Real x) const {
      if(x < 0.0) return 0.0;
      if(x > 1.0) return 1.0;
      return x;
   }

   Real GoalProgressScore() const {
      CVector3 core = CoreCentroid();
      Real dist_goal = XYOnly(goal - core).Length();

      return Clamp01(1.0 - dist_goal / std::max(initial_goal_dist, 1e-6));
   }

   Real CriticalityScore(Real combined_margin_mean,
                         Real lambda2,
                         Real min_od,
                         Real min_dd,
                         Real max_infl_dist) const {
      Real margin_score = Clamp01(0.5 + 0.5 * combined_margin_mean);
      Real conn_score = Clamp01(lambda2 / 20.0);
      Real obs_score = Clamp01(min_od / 0.80);
      Real sep_score = Clamp01(min_dd / 0.22);
      Real infl_core_score = Clamp01(1.0 - max_infl_dist / 1.25);
      Real goal_score = GoalProgressScore();

      return Clamp01(
         0.30 * margin_score +
         0.18 * conn_score +
         0.18 * obs_score +
         0.14 * sep_score +
         0.10 * infl_core_score +
         0.10 * goal_score
      );
   }

   void ApplyMovement() {
      CQuaternion q;
      q.FromEulerAngles(CRadians(0), CRadians(0), CRadians(0));

      for(auto& a : agents) {
         if(a.influential) {
            a.pos = SetFlightZ(a.pos);
         } else {
            a.pos = Clamp(a.pos);
         }
         a.entity->GetEmbodiedEntity().MoveTo(a.pos, q, false);
      }
   }

   void ApplyObstacleVisuals() {
      CQuaternion q;
      q.FromEulerAngles(CRadians(0), CRadians(0), CRadians(0));

      for(auto& o : obstacles) {
         if(o.dynamic && o.visual != nullptr) {
            CVector3 p = ObsCenter(o);
            o.visual->GetEmbodiedEntity().MoveTo(p, q, false);
         }
      }
   }

   void PostStep() override {
      step++;

      if(!initialized) {
         InitDefinedFormation();
      }

      UpdateGoalTracking();

      UpdateInfluentials();
      UpdateDecoys();

      /*
         Step-3 connectivity repair:
         If the graph splits, pull closest components together.
      */
      ComponentBridgeRepair();

      /*
         Step-3.2 local k-connectivity repair:
         Even when the graph has one component, reinforce agents with
         too few local neighbors.
      */
      LocalKConnectivityRepair();

      if(MinNeighborCountCurrent() <= static_cast<int>(k_conn + 1) ||
         NumConnectedComponents() > 1) {
         weak_connectivity_steps++;
      }

      ApplyMovement();
      ApplyObstacleVisuals();

      if(csv.is_open() && step % csv_log_period == 0) {
         WriteLogRow(false);
      }

      if(agent_csv.is_open() && step % csv_log_period == 0) {
         WriteAgentPositions();
      }

      if(step % terminal_log_period == 0) {
         WriteLogRow(true);
      }
   }

   void WriteAgentPositions() {
      if(!agent_csv.is_open()) {
         return;
      }

      for(const auto& a : agents) {
         agent_csv << step << ","
                   << a.id << ","
                   << (a.influential ? "infl" : "dec") << ","
                   << a.pos.GetX() << ","
                   << a.pos.GetY() << ","
                   << a.pos.GetZ() << "\n";
      }
   }

   void WriteLogRow(bool terminal) {
      CVector3 core = CoreCentroid();

      Real dist_goal = XYOnly(goal - core).Length();
      Real max_infl_dist = MaxInfluentialDistanceFromCore();

      Real band_err = AvgBandError();
      Real min_dd   = MinDecoyDistance();
      Real min_od   = MinObstacleDecoyDistance();
      Real lambda2  = Lambda2FastProxy();
      int min_neighbors = MinNeighborCountCurrent();
      int num_components = NumConnectedComponents();
      Real largest_component_fraction = LargestComponentFraction();

      std::vector<Real> P = ProtectionDiagnosticVector();
      std::vector<Real> S = StructuralScores();

      int m = CountInfl();
      int n = CountDec();

      std::vector<Real> P_infl, P_dec;
      std::vector<Real> S_infl, S_dec;
      std::vector<Real> J_infl, J_dec;

      Real lambdaP = fit_prot;
      Real lambdaS = fit_struct;

      for(int i = 0; i < m + n; ++i) {
         Real p = (i < static_cast<int>(P.size())) ? P[i] : 0.0;
         Real s = (i < static_cast<int>(S.size())) ? S[i] : 0.0;
         Real j = (lambdaP * p + lambdaS * s) / std::max(lambdaP + lambdaS, 1e-6);

         if(i < m) {
            P_infl.push_back(p);
            S_infl.push_back(s);
            J_infl.push_back(j);
         } else {
            P_dec.push_back(p);
            S_dec.push_back(s);
            J_dec.push_back(j);
         }
      }

      Real prot_infl_mean = Mean(P_infl);
      Real prot_dec_mean  = Mean(P_dec);
      Real struct_infl_mean = Mean(S_infl);
      Real struct_dec_mean  = Mean(S_dec);
      Real combined_infl_mean = Mean(J_infl);
      Real combined_dec_mean  = Mean(J_dec);

      Real margin_prot_mean = prot_infl_mean - prot_dec_mean;
      Real margin_struct_mean = struct_infl_mean - struct_dec_mean;
      Real margin_combined_mean = combined_infl_mean - combined_dec_mean;

      Real margin_prot_robust = Percentile(P_infl, 0.25) - Percentile(P_dec, 0.75);
      Real margin_struct_robust = Percentile(S_infl, 0.25) - Percentile(S_dec, 0.75);
      Real margin_combined_robust = Percentile(J_infl, 0.25) - Percentile(J_dec, 0.75);

      Real margin_prot_worst =
         (P_infl.empty() || P_dec.empty()) ? 0.0 :
         *std::min_element(P_infl.begin(), P_infl.end()) -
         *std::max_element(P_dec.begin(), P_dec.end());

      Real margin_struct_worst =
         (S_infl.empty() || S_dec.empty()) ? 0.0 :
         *std::min_element(S_infl.begin(), S_infl.end()) -
         *std::max_element(S_dec.begin(), S_dec.end());

      Real margin_combined_worst =
         (J_infl.empty() || J_dec.empty()) ? 0.0 :
         *std::min_element(J_infl.begin(), J_infl.end()) -
         *std::max_element(J_dec.begin(), J_dec.end());

      Real formation_health = FormationHealthFactor();
      Real goal_prog = GoalProgressScore();
      Real crit = CriticalityScore(margin_combined_mean, lambda2, min_od, min_dd, max_infl_dist);
      Real mean_shell_error = MeanShellError();
      Real spherical_coverage = GlobalSphericalCoverageScore();
      Real vertical_protection = VerticalProtectionScore();
      int upper_decoys = UpperDecoyCount();
      int lower_decoys = LowerDecoyCount();

      if(!terminal) {
         csv << step << ","
             << core.GetX() << ","
             << core.GetY() << ","
             << core.GetZ() << ","
             << dist_goal << ","
             << max_infl_dist << ","
             << band_err << ","
             << min_dd << ","
             << min_od << ","
             << lambda2 << ","
             << min_neighbors << ","
             << weak_connectivity_steps << ","
             << num_components << ","
             << largest_component_fraction << ","
             << last_bridge_pairs << ","
             << last_bridge_distance << ","
             << prot_infl_mean << ","
             << prot_dec_mean << ","
             << struct_infl_mean << ","
             << struct_dec_mean << ","
             << combined_infl_mean << ","
             << combined_dec_mean << ","
             << margin_prot_mean << ","
             << margin_prot_robust << ","
             << margin_prot_worst << ","
             << margin_struct_mean << ","
             << margin_struct_robust << ","
             << margin_struct_worst << ","
             << margin_combined_mean << ","
             << margin_combined_robust << ","
             << margin_combined_worst << ","
             << formation_health << ","
             << goal_prog << ","
             << crit << ","
             << mean_shell_error << ","
             << spherical_coverage << ","
             << vertical_protection << ","
             << upper_decoys << ","
             << lower_decoys << ","
             << (mission_complete ? 1 : 0)
             << "\n";
      } else {
         LOG << "[t=" << step << "]"
             << " core=(" << core.GetX() << "," << core.GetY() << "," << core.GetZ() << ")"
             << " dist_goal=" << dist_goal
             << " max_infl_core_dist=" << max_infl_dist
             << " min_dd=" << min_dd
             << " min_od=" << min_od
             << " lambda2_proxy=" << lambda2
             << " min_neighbors=" << min_neighbors
             << " weak_steps=" << weak_connectivity_steps
             << " components=" << num_components
             << " largest_comp=" << largest_component_fraction
             << " bridge_pairs=" << last_bridge_pairs
             << " bridge_dist=" << last_bridge_distance
             << " Pinfl=" << prot_infl_mean
             << " Pdec=" << prot_dec_mean
             << " Sinfl=" << struct_infl_mean
             << " Sdec=" << struct_dec_mean
             << " Jmargin=" << margin_combined_mean
             << " health=" << formation_health
             << " crit=" << crit
             << " shell_err=" << mean_shell_error
             << " sph_cov=" << spherical_coverage
             << " vert=" << vertical_protection
             << " upper=" << upper_decoys
             << " lower=" << lower_decoys
             << " mission_complete=" << (mission_complete ? 1 : 0)
             << std::endl;
      }
   }
};

REGISTER_LOOP_FUNCTIONS(CBTP_LoopFunctions, "loop");
