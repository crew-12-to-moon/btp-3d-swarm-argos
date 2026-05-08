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

using namespace argos;

struct Chromosome {
   Real s      = 0.045;
   Real wcent  = 0.70;
   Real wsep   = 2.20;
   Real wband  = 2.80;
   Real wobs   = 1.60;
   Real fitness = -1e9;
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
   std::vector<CVector3> waypoints;

   UInt32 step = 0;
   bool initialized = false;
   UInt32 current_wp = 0;

   std::ofstream csv;

   /*
      30-swarm setup:
      4 influentials + 26 decoys = 30 total
      Influential fraction = 13.33%
   */
   Real flight_z = 1.50;

   CVector3 start = CVector3(-7.000, -7.000, 1.500);
   CVector3 goal  = CVector3(7.000, 7.000, 1.500);

   Real goal_accept_radius = 1.35;
   Real wp_accept_radius   = 0.95;
   bool mission_complete   = false;

   /*
      Protection band.
      Ring radius is large enough for 26 decoys without over-compression.
   */
   Real rmin  = 1.000;
   Real rmax  = 2.100;
   Real rring = 1.550;
   Real tau   = 0.60;

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
   Real graph_radius = 2.050;

   /*
      Stuck detection
   */
   Real last_dist_to_wp = 1e9;
   UInt32 stuck_counter = 0;

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

   UInt32 csv_log_period = 25;
   UInt32 terminal_log_period = 100;

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
   Real k_conn_rescue = 0.120;

   void Init(TConfigurationNode&) override {
      LOG << "BTP ARGoS ANTI-STALL initialized: 30 swarm, protected decoy ring, core-lock with minimum progress" << std::endl;

      std::srand(7);

      LoadAgents();
      InitObstacles();
      LinkObstacleVisuals();
      InitWaypoints();
      InitDecoyPopulations();
      InitDefinedFormation();
      OpenCSV();
   }

   void Reset() override {
      agents.clear();
      obstacles.clear();
      waypoints.clear();

      step = 0;
      initialized = false;
      current_wp = 0;
      last_dist_to_wp = 1e9;
      stuck_counter = 0;
      initial_goal_dist = 1.0;
      mission_complete = false;

      if(csv.is_open()) {
         csv.close();
      }

      std::srand(7);

      LoadAgents();
      InitObstacles();
      LinkObstacleVisuals();
      InitWaypoints();
      InitDecoyPopulations();
      InitDefinedFormation();
      OpenCSV();
   }

   void Destroy() override {
      if(csv.is_open()) {
         csv.close();
      }
   }

   void OpenCSV() {
      csv.open("btp_argos_metrics.csv");

      csv << "step,"
          << "wp_index,"
          << "core_x,"
          << "core_y,"
          << "core_z,"
          << "dist_wp,"
          << "dist_goal,"
          << "max_infl_core_dist,"
          << "avg_band_error,"
          << "min_decoy_decoy,"
          << "min_obstacle_decoy,"
          << "lambda2_proxy,"
          << "min_neighbor_count,"
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
          << "mission_complete"
          << "\n";
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
      /* Generated obstacle mode: none */
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

   void InitWaypoints() {
      waypoints.clear();
      /* Generated waypoint mode: default */
      waypoints.push_back(CVector3(-7.000, -7.000, 1.500));
      waypoints.push_back(CVector3(-6.800, -3.000, 1.500));
      waypoints.push_back(CVector3(-4.000, -1.200, 1.500));
      waypoints.push_back(CVector3(-5.300, 2.600, 1.500));
      waypoints.push_back(CVector3(-1.000, 4.500, 1.500));
      waypoints.push_back(CVector3(2.500, 3.200, 1.500));
      waypoints.push_back(CVector3(5.000, 5.500, 1.500));
      waypoints.push_back(CVector3(7.000, 7.000, 1.500));
      current_wp = 1;
      initial_goal_dist = XYOnly(goal - start).Length();
      if(initial_goal_dist < 1e-6) { initial_goal_dist = 1.0; }
      LOG << "Loaded " << waypoints.size() << " waypoints | initial_goal_dist=" << initial_goal_dist << std::endl;
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
      p.SetZ(flight_z);
      return p;
   }

   CVector3 ObsCenter(const BoxObs& o) const {
      if(!o.dynamic) {
         return o.center;
      }

      Real s = std::sin(o.freq * static_cast<Real>(step) + o.phase);
      return o.center + o.amp * s;
   }

   Real SDFBox(const CVector3& p, const BoxObs& o) const {
      CVector3 c = ObsCenter(o);

      Real qx = std::fabs(p.GetX() - c.GetX()) - o.half.GetX();
      Real qy = std::fabs(p.GetY() - c.GetY()) - o.half.GetY();
      Real qz = std::fabs(p.GetZ() - c.GetZ()) - o.half.GetZ();

      Real ox = std::max(qx, 0.0);
      Real oy = std::max(qy, 0.0);
      Real oz = std::max(qz, 0.0);

      Real outside = std::sqrt(ox*ox + oy*oy + oz*oz);
      Real inside = std::min(std::max(qx, std::max(qy, qz)), 0.0);

      return outside + inside;
   }

   CVector3 GradSDFBoxXY(const CVector3& p, const BoxObs& o) const {
      Real e = 0.005;

      CVector3 ex(e,0,0);
      CVector3 ey(0,e,0);

      Real gx = (SDFBox(p + ex, o) - SDFBox(p - ex, o)) / (2.0 * e);
      Real gy = (SDFBox(p + ey, o) - SDFBox(p - ey, o)) / (2.0 * e);

      return NormalizeSafeXY(CVector3(gx, gy, 0.0));
   }

   CVector3 ObstacleForceXY(const CVector3& p, Real gain) const {
      CVector3 f(0,0,0);

      for(const auto& o : obstacles) {
         Real phi = SDFBox(p, o) - obs_safe;

         if(phi < obs_influence) {
            CVector3 n = GradSDFBoxXY(p, o);

            Real mag = 0.0;
            if(phi <= 0.0) {
               mag = 2.5;
            } else {
               mag = (1.0 / std::max(phi, 1e-4)) - (1.0 / obs_influence);
            }

            f += n * (gain * mag);
         }
      }

      f.SetZ(0.0);
      return f;
   }

   CVector3 ProjectOutside(CVector3 p) const {
      p.SetZ(flight_z);

      for(const auto& o : obstacles) {
         Real phi = SDFBox(p, o) - obs_safe;

         if(phi < 0.0) {
            CVector3 n = GradSDFBoxXY(p, o);
            p += n * (-phi + 0.03);
            p.SetZ(flight_z);
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
      if(waypoints.empty()) {
         return goal;
      }

      size_t idx = std::min(static_cast<size_t>(current_wp), waypoints.size() - 1);
      return waypoints[idx];
   }

   Real FormationHealthFactor() const {
      Real pinfl = ProtectionInfluentialMean();
      int min_neighbors = MinNeighborCountCurrent();
      Real min_dd = MinDecoyDistance();
      Real max_infl_dist = MaxInfluentialDistanceFromCore();

      Real health = 1.0;

      /*
         Do not collapse health too aggressively.
         Earlier core-lock version could deadlock/stall because the core slowed too much.
      */
      if(pinfl < 0.45) {
         health *= 0.65;
      } else if(pinfl < 0.55) {
         health *= 0.80;
      }

      if(min_neighbors < static_cast<int>(k_conn)) {
         health *= 0.75;
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
         Critical anti-stall fix:
         Never let formation health collapse to near zero.
         The swarm must keep moving enough to escape bad obstacle geometry.
      */
      return std::max<Real>(0.35, Clamp01(health));
   }

   void UpdateWaypoint() {
      if(waypoints.empty()) {
         return;
      }

      if(mission_complete) {
         return;
      }

      CVector3 core = CoreCentroid();
      CVector3 target = CurrentTarget();

      Real dist = XYOnly(target - core).Length();
      bool final_wp = (current_wp >= waypoints.size() - 1);
      Real accept_radius = final_wp ? goal_accept_radius : wp_accept_radius;
      Real max_infl_dist = MaxInfluentialDistanceFromCore();

      /*
         Final mission completion requires:
         core at goal + protection + connectivity + safety + all influentials with core.
      */
      if(final_wp && dist < accept_radius) {
         if(IsMissionSafeToComplete()) {
            mission_complete = true;

            LOG << "MISSION COMPLETE: core reached goal and swarm quality is valid. "
                << "dist_goal=" << dist
                << " goal_accept_radius=" << goal_accept_radius
                << std::endl;
         } else if(step % terminal_log_period == 0) {
            LOG << "Core inside goal radius, but mission not complete: swarm quality invalid."
                << std::endl;
         }

         return;
      }

      /*
         Intermediate waypoint switching now requires:
         - centroid near waypoint
         - decoy protection healthy
         - no isolated influential
      */
      if(!final_wp &&
         dist < accept_radius &&
         FormationHealthFactor() > 0.30 &&
         max_infl_dist < 1.20) {
         current_wp++;
         last_dist_to_wp = 1e9;
         stuck_counter = 0;

         LOG << "Switching to waypoint " << current_wp << std::endl;
         return;
      }

      if(dist > last_dist_to_wp - 0.004) {
         stuck_counter++;
      } else {
         stuck_counter = 0;
      }

      last_dist_to_wp = dist;

      /*
         Do not force waypoint switching if formation health is poor.
         First let the protection ring and influential core recover.
      */
      if(stuck_counter > 350 && !final_wp) {
         current_wp++;
         stuck_counter = 0;
         last_dist_to_wp = 1e9;

         LOG << "Stuck detected. Forcing waypoint " << current_wp << std::endl;
      }
   }

   bool IsMissionSafeToComplete() const {
      Real lambda2 = Lambda2FastProxy();
      Real min_dd = MinDecoyDistance();
      Real min_od = MinObstacleDecoyDistance();
      Real pinfl  = ProtectionInfluentialMean();
      int min_neighbors = MinNeighborCountCurrent();
      Real max_infl_dist = MaxInfluentialDistanceFromCore();

      return
         lambda2 > complete_min_lambda2_proxy &&
         min_neighbors >= static_cast<int>(k_conn) &&
         min_dd  > complete_min_decoy_dist &&
         min_od  > complete_min_obs_dist &&
         pinfl   > complete_min_protection &&
         max_infl_dist < complete_max_infl_core_dist;
   }

   void InitDefinedFormation() {
      /* Generated init=clean_ring, seed=1 */
      std::vector<CVector3> infl_pos = {
         CVector3(-7.000, -7.000, 1.500),
         CVector3(-6.700, -7.000, 1.500),
         CVector3(-7.300, -7.000, 1.500),
         CVector3(-7.000, -6.700, 1.500),
      };
      std::vector<CVector3> dec_pos = {
         CVector3(-5.450, -7.000, 1.500),
         CVector3(-5.495, -6.629, 1.500),
         CVector3(-5.628, -6.280, 1.500),
         CVector3(-5.840, -5.972, 1.500),
         CVector3(-6.119, -5.724, 1.500),
         CVector3(-6.450, -5.551, 1.500),
         CVector3(-6.813, -5.461, 1.500),
         CVector3(-7.187, -5.461, 1.500),
         CVector3(-7.550, -5.551, 1.500),
         CVector3(-7.881, -5.724, 1.500),
         CVector3(-8.160, -5.972, 1.500),
         CVector3(-8.372, -6.280, 1.500),
         CVector3(-8.505, -6.629, 1.500),
         CVector3(-8.550, -7.000, 1.500),
         CVector3(-8.505, -7.371, 1.500),
         CVector3(-8.372, -7.720, 1.500),
         CVector3(-8.160, -8.028, 1.500),
         CVector3(-7.881, -8.276, 1.500),
         CVector3(-7.550, -8.449, 1.500),
         CVector3(-7.187, -8.539, 1.500),
         CVector3(-6.813, -8.539, 1.500),
         CVector3(-6.450, -8.449, 1.500),
         CVector3(-6.119, -8.276, 1.500),
         CVector3(-5.840, -8.028, 1.500),
         CVector3(-5.628, -7.720, 1.500),
         CVector3(-5.495, -7.371, 1.500),
      };
      int i = 0;
      for(auto& a : agents) {
         if(!a.influential) continue;
         if(i < static_cast<int>(infl_pos.size())) a.pos = SetFlightZ(infl_pos[i]);
         a.vel = CVector3(0,0,0);
         i++;
      }
      int j = 0;
      for(auto& a : agents) {
         if(a.influential) continue;
         if(j < static_cast<int>(dec_pos.size())) a.pos = SetFlightZ(dec_pos[j]);
         a.vel = CVector3(0,0,0);
         a.slot = j;
         j++;
      }
      ApplyMovement();
      ApplyObstacleVisuals();
      initialized = true;
      LOG << "Generated formation initialized: influentials=" << CountInfl() << " decoys=" << CountDec() << std::endl;
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

         CVector3 d = XYOnly(y - decoys[k]);
         Real L = d.Length();

         if(L < dec_sep_radius && L > 1e-6) {
            f += NormalizeSafeXY(d) * ((dec_sep_radius - L) / dec_sep_radius);
         }
      }

      return f;
   }

   CVector3 BandForceNearestInfluentialFromPositions(const CVector3& y,
                                                     const std::vector<CVector3>& infls) const {
      Real best = std::numeric_limits<Real>::max();
      CVector3 nearest(0,0,flight_z);

      for(const auto& x : infls) {
         Real d = XYOnly(y - x).Length();

         if(d < best) {
            best = d;
            nearest = x;
         }
      }

      CVector3 radial = XYOnly(y - nearest);
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
         CVector3 f_obs = ObstacleForceXY(a.pos, k_obs_infl);

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

      Real th0 = 2.0 * ARGOS_PI * static_cast<Real>(dec_idx) / ndec;

      /*
         Almost no rotation.
         Ring should translate with the core, not swirl around it.
      */
      Real rot = 0.00005 * static_cast<Real>(step);
      Real th = th0 + rot;

      return SetFlightZ(core + CVector3(
         rring * std::cos(th),
         rring * std::sin(th),
         0.0
      ));
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
      CVector3 f_obs  = ObstacleForceXY(y, c.wobs);

      /*
         Strong assigned-slot force.
         This is the dominant protection-preserving force.
      */
      CVector3 f_slot = XYOnly(slot_target - y) * k_slot_bias;

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

      return LimitXY(v + f, c.s);
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
         Real d = XYOnly(y - x).Length();
         best = std::max(best, BandMembership(d));
      }

      return best;
   }

   Real CandidateObstaclePenalty(const CVector3& y) const {
      Real pen = 0.0;

      for(const auto& o : obstacles) {
         Real phi = SDFBox(y, o) - obs_safe;

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
            if(XYOnly(pts[i] - pts[j]).Length() <= graph_radius) {
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

         if(XYOnly(pts[self_idx] - pts[i]).Length() <= radius) {
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
      Real d = XYOnly(y - slot).Length();
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

   Real EvaluateChromosome(int dec_idx,
                           const CVector3& y0,
                           const CVector3& v0,
                           const Chromosome& c,
                           const std::vector<CVector3>& infls0,
                           const std::vector<CVector3>& decoys0) const {
      std::vector<CVector3> infls = infls0;
      std::vector<CVector3> decoys = decoys0;

      CVector3 y = y0;
      CVector3 v = v0;

      Real fitness = 0.0;

      CVector3 target = CurrentTarget();
      CVector3 core0 = MeanPosition(infls);
      CVector3 core_target_dir = NormalizeSafeXY(target - core0);

      for(UInt32 h = 0; h < horizon; ++h) {
         CVector3 core = MeanPosition(infls);
         CVector3 dir = NormalizeSafeXY(target - core);

         /*
            Forecast influential motion.
         */
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

         Real prot_reward = CandidateProtectionReward(y, infls);
         Real struct_reward = MeanInfluentialStructuralScore(infls, decoys);
         Real obs_pen = CandidateObstaclePenalty(y);
         Real move_pen = v.Length() * v.Length();
         Real conn_reward = CandidateConnectivityReward(infls, decoys);
         Real conn_break_pen = CandidateConnectivityBreakPenalty(infls, decoys, dec_idx);
         Real slot_reward = CandidateSlotReward(y, slot_target);

         fitness +=
            fit_prot   * prot_reward +
            fit_struct * struct_reward +
            fit_conn   * conn_reward +
            fit_slot   * slot_reward -
            fit_conn_break * conn_break_pen -
            fit_obs    * obs_pen -
            fit_move   * move_pen;
      }

      return fitness / static_cast<Real>(std::max<UInt32>(1, horizon));
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
            c.fitness = EvaluateChromosome(dec_idx, a.pos, a.vel, c, infls, decoys);
         }

         std::sort(a.population.begin(), a.population.end(),
            [](const Chromosome& p, const Chromosome& q) {
               return p.fitness > q.fitness;
            });

         if(!a.population.empty()) {
            a.active = a.population[0];
         }

         std::vector<Chromosome> next;

         if(a.population.size() >= 1) {
            next.push_back(a.population[0]);
         }

         while(next.size() < population_size) {
            size_t pool = std::min<size_t>(2, a.population.size());
            const Chromosome& p1 = a.population[std::rand() % pool];
            const Chromosome& p2 = a.population[std::rand() % pool];

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
         CVector3 to_slot = XYOnly(slot - a.pos);
         Real slot_dist = to_slot.Length();

         if(slot_dist > 0.95) {
            CVector3 rescue = LimitXY(to_slot * 0.40, 0.080);
            a.pos = ProjectOutside(a.pos + rescue);
            a.vel *= 0.25;
         }

         /*
            Core-envelope rescue:
            Prevent disconnected trailing subgroups.
         */
         Real dcore = XYOnly(a.pos - core).Length();

         if(dcore > rmax + 0.65) {
            CVector3 pull = LimitXY(slot - a.pos, 0.090);
            a.pos = ProjectOutside(a.pos + pull);
            a.vel *= 0.20;
         }

         /*
            Adaptive replacement-link rescue:
            If this decoy has fewer than k_conn links, reconnect it in a
            protection-aware way. Slot/core dominate, not nearest clump.
         */
         int n_neighbors = NeighborCountCurrent(a.pos, a.id);
         if(n_neighbors < static_cast<int>(k_conn)) {
            CVector3 rescue = ConnectivityRescueForce(a.pos, a.id, slot, core);
            a.pos = ProjectOutside(a.pos + rescue);
            a.vel *= 0.35;
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

   std::vector<Real> ProtectionDiagnosticVector() const {
      std::vector<CVector3> infls;
      std::vector<CVector3> decoys;

      for(const auto& a : agents) {
         if(a.influential) infls.push_back(a.pos);
         else decoys.push_back(a.pos);
      }

      std::vector<Real> P;

      /*
         Influential protection.
      */
      for(const auto& x : infls) {
         Real sumB = 0.0;
         std::vector<Real> angles;

         for(const auto& y : decoys) {
            CVector3 r = XYOnly(y - x);
            Real d = r.Length();

            sumB += BandMembership(d);
            angles.push_back(std::atan2(r.GetY(), r.GetX()));
         }

         Real meanB = decoys.empty() ? 0.0 : sumB / static_cast<Real>(decoys.size());
         Real gamma = AngularUniformity(angles);

         P.push_back(0.8 * meanB + 0.2 * gamma);
      }

      /*
         Decoy diagnostic protection.
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
            CVector3 r = XYOnly(allpts[q] - y);
            Real d = r.Length();

            if(d < 1e-8) {
               continue;
            }

            sumB += BandMembership(d);
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

         if(XYOnly(p - b.pos).Length() <= graph_radius) {
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
                                    const CVector3& core) const {
      std::vector<std::pair<Real, CVector3>> near_agents;

      for(const auto& b : agents) {
         if(b.id == self_id) {
            continue;
         }

         Real d = XYOnly(p - b.pos).Length();
         near_agents.push_back(std::make_pair(d, b.pos));
      }

      if(near_agents.empty()) {
         return LimitXY(core - p, k_conn_rescue);
      }

      std::sort(near_agents.begin(), near_agents.end(),
         [](const std::pair<Real, CVector3>& a,
            const std::pair<Real, CVector3>& b) {
            return a.first < b.first;
         });

      CVector3 neighbor_centroid(0,0,0);
      int used = 0;

      int max_use = std::min<int>(static_cast<int>(k_conn), static_cast<int>(near_agents.size()));

      for(int i = 0; i < max_use; ++i) {
         neighbor_centroid += near_agents[i].second;
         used++;
      }

      if(used > 0) {
         neighbor_centroid /= static_cast<Real>(used);
      } else {
         neighbor_centroid = core;
      }

      neighbor_centroid.SetZ(flight_z);

      /*
         Protection-aware replacement-link rescue:
         slot and core dominate; nearest neighbors are only secondary.
      */
      CVector3 desired =
         slot              * 0.60 +
         core              * 0.25 +
         neighbor_centroid * 0.15;

      desired.SetZ(flight_z);

      return LimitXY(desired - p, k_conn_rescue);
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
            Real d = XYOnly(pts[i] - pts[j]).Length();

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
            best = std::min(best, XYOnly(a.pos - b.pos).Length());
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
            best = std::min(best, XYOnly(agents[i].pos - agents[j].pos).Length());
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
            best = std::min(best, SDFBox(a.pos, o));
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
         a.pos = SetFlightZ(a.pos);
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

      UpdateWaypoint();

      UpdateInfluentials();
      UpdateDecoys();

      ApplyMovement();
      ApplyObstacleVisuals();

      if(csv.is_open() && step % csv_log_period == 0) {
         WriteLogRow(false);
      }

      if(step % terminal_log_period == 0) {
         WriteLogRow(true);
      }
   }

   void WriteLogRow(bool terminal) {
      CVector3 core = CoreCentroid();
      CVector3 target = CurrentTarget();

      Real dist_wp   = XYOnly(target - core).Length();
      Real dist_goal = XYOnly(goal - core).Length();
      Real max_infl_dist = MaxInfluentialDistanceFromCore();

      Real band_err = AvgBandError();
      Real min_dd   = MinDecoyDistance();
      Real min_od   = MinObstacleDecoyDistance();
      Real lambda2  = Lambda2FastProxy();
      int min_neighbors = MinNeighborCountCurrent();

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

      if(!terminal) {
         csv << step << ","
             << current_wp << ","
             << core.GetX() << ","
             << core.GetY() << ","
             << core.GetZ() << ","
             << dist_wp << ","
             << dist_goal << ","
             << max_infl_dist << ","
             << band_err << ","
             << min_dd << ","
             << min_od << ","
             << lambda2 << ","
             << min_neighbors << ","
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
             << (mission_complete ? 1 : 0)
             << "\n";
      } else {
         LOG << "[t=" << step << "]"
             << " wp=" << current_wp
             << " core=(" << core.GetX() << "," << core.GetY() << "," << core.GetZ() << ")"
             << " dist_goal=" << dist_goal
             << " max_infl_core_dist=" << max_infl_dist
             << " min_dd=" << min_dd
             << " min_od=" << min_od
             << " lambda2_proxy=" << lambda2
             << " min_neighbors=" << min_neighbors
             << " Pinfl=" << prot_infl_mean
             << " Pdec=" << prot_dec_mean
             << " Sinfl=" << struct_infl_mean
             << " Sdec=" << struct_dec_mean
             << " Jmargin=" << margin_combined_mean
             << " health=" << formation_health
             << " crit=" << crit
             << " mission_complete=" << (mission_complete ? 1 : 0)
             << std::endl;
      }
   }
};

REGISTER_LOOP_FUNCTIONS(CBTP_LoopFunctions, "loop");
