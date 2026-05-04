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

using namespace argos;

struct Agent {
   std::string id;
   CPrototypeEntity* entity = nullptr;
   CVector3 pos;
   CVector3 vel;
   bool influential = false;
   int slot = -1;
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
      Larger mission setup
   */
   Real flight_z = 1.50;

   CVector3 start = CVector3(-8.0, -8.0, 1.50);
   CVector3 goal  = CVector3( 8.0,  8.0, 1.50);

   /*
      Protection band and ring
   */
   Real rmin  = 0.75;
   Real rmax  = 1.65;
   Real rring = 1.25;
   Real tau   = 0.55;

   /*
      Step sizes per ARGoS tick
   */
   Real infl_step_max = 0.030;
   Real dec_step_max  = 0.050;

   /*
      Influential core gains
   */
   Real k_goal_core     = 0.060;
   Real k_core_cohesion = 0.012;
   Real k_infl_sep      = 0.014;
   Real k_obs_infl      = 0.050;

   /*
      Decoy gains
   */
   Real k_shape   = 0.090;
   Real k_band    = 0.060;
   Real k_dec_sep = 0.050;
   Real k_follow  = 0.030;
   Real k_tangent = 0.0015;
   Real k_obs_dec = 0.055;

   Real infl_sep_radius = 0.30;
   Real dec_sep_radius  = 0.28;

   /*
      SDF obstacle settings
   */
   Real obs_safe      = 0.35;
   Real obs_influence = 1.35;

   /*
      Larger arena clamp
   */
   Real xmin = -10.6;
   Real xmax =  10.6;
   Real ymin = -10.6;
   Real ymax =  10.6;

   /*
      Connectivity proxy radius
   */
   Real graph_radius = 1.90;

   /*
      Stuck detection
   */
   Real last_dist_to_wp = 1e9;
   UInt32 stuck_counter = 0;

   /*
      Criticality normalization
   */
   Real initial_goal_dist = 1.0;

   /*
      Mission completion logic
      If the influential core enters goal_accept_radius, mission is complete.
   */
   Real goal_accept_radius = 1.25;
   Real wp_accept_radius   = 0.85;
   bool mission_complete   = false;

   void Init(TConfigurationNode&) override {
      LOG << "BTP long-path 3D sim initialized: large arena + dense obstacle field + waypoint navigation" << std::endl;

      LoadAgents();
      InitObstacles();
      LinkObstacleVisuals();
      InitWaypoints();
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

      LoadAgents();
      InitObstacles();
      LinkObstacleVisuals();
      InitWaypoints();
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
          << "avg_band_error,"
          << "min_decoy_decoy,"
          << "min_obstacle_decoy,"
          << "lambda2_proxy,"
          << "protection_score,"
          << "connectivity_score,"
          << "obstacle_safety_score,"
          << "decoy_spacing_score,"
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

      /*
         Ground buildings.
         These are positioned to create an urban obstacle field,
         but the waypoint corridor is kept wide enough to avoid choking.
      */
      obstacles.push_back({"bld_0",  CVector3(-6.2, -5.2, 1.1), CVector3(0.6, 0.7, 1.1), false});
      obstacles.push_back({"bld_1",  CVector3(-3.8, -6.4, 1.0), CVector3(0.55, 0.5, 1.0), false});
      obstacles.push_back({"bld_2",  CVector3(-1.3, -4.8, 1.4), CVector3(0.7, 0.5, 1.4), false});
      obstacles.push_back({"bld_3",  CVector3( 1.8, -6.0, 1.2), CVector3(0.6, 0.65, 1.2), false});

      obstacles.push_back({"bld_4",  CVector3(-7.0, -1.5, 1.3), CVector3(0.55, 0.7, 1.3), false});
      obstacles.push_back({"bld_5",  CVector3(-4.4, -0.6, 1.6), CVector3(0.65, 0.55, 1.6), false});
      obstacles.push_back({"bld_6",  CVector3(-1.4, -1.8, 1.2), CVector3(0.5, 0.8, 1.2), false});
      obstacles.push_back({"bld_7",  CVector3( 2.4, -1.2, 1.5), CVector3(0.75, 0.55, 1.5), false});

      obstacles.push_back({"bld_8",  CVector3(-5.8,  2.9, 1.2), CVector3(0.7, 0.55, 1.2), false});
      obstacles.push_back({"bld_9",  CVector3(-2.8,  3.4, 1.7), CVector3(0.55, 0.8, 1.7), false});
      obstacles.push_back({"bld_10", CVector3( 0.7,  2.4, 1.3), CVector3(0.6, 0.65, 1.3), false});
      obstacles.push_back({"bld_11", CVector3( 4.0,  2.8, 1.5), CVector3(0.7, 0.6, 1.5), false});

      obstacles.push_back({"bld_12", CVector3( 2.0,  6.5, 1.2), CVector3(0.6, 0.55, 1.2), false});
      obstacles.push_back({"bld_13", CVector3( 6.0,  5.6, 1.4), CVector3(0.65, 0.65, 1.4), false});

      /*
         Small dynamic aerial obstacles.
         Sizes are intentionally small to avoid blocking corridors fully.
      */
      obstacles.push_back({"air_0", CVector3(-5.0, -2.8, 2.2), CVector3(0.30, 0.30, 0.25), true,
                           CVector3(1.2, 0.0, 0.25), 0.020, 0.0});

      obstacles.push_back({"air_1", CVector3(-2.3,  0.9, 2.4), CVector3(0.28, 0.28, 0.24), true,
                           CVector3(0.0, 1.1, 0.20), 0.024, 1.1});

      obstacles.push_back({"air_2", CVector3( 0.5, -3.0, 2.1), CVector3(0.25, 0.25, 0.22), true,
                           CVector3(1.0, 0.0, 0.18), 0.018, 2.2});

      obstacles.push_back({"air_3", CVector3( 2.5,  1.2, 2.5), CVector3(0.30, 0.30, 0.25), true,
                           CVector3(0.0, 1.3, 0.25), 0.021, 0.7});

      obstacles.push_back({"air_4", CVector3( 5.2,  3.5, 2.3), CVector3(0.26, 0.26, 0.22), true,
                           CVector3(1.0, 0.0, 0.20), 0.019, 1.8});

      obstacles.push_back({"air_5", CVector3( 0.0,  6.0, 2.2), CVector3(0.25, 0.25, 0.22), true,
                           CVector3(0.0, 1.1, 0.20), 0.017, 2.7});
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

      /*
         Long route across the obstacle environment.
         The route intentionally threads between building clusters,
         but avoids narrow throat gaps.
      */
      waypoints.push_back(CVector3(-8.0, -8.0, flight_z));
      waypoints.push_back(CVector3(-8.2, -4.2, flight_z));
      waypoints.push_back(CVector3(-6.7, -2.6, flight_z));
      waypoints.push_back(CVector3(-6.5,  0.8, flight_z));
      waypoints.push_back(CVector3(-4.5,  2.0, flight_z));
      waypoints.push_back(CVector3(-3.2,  5.4, flight_z));
      waypoints.push_back(CVector3(-0.2,  6.8, flight_z));
      waypoints.push_back(CVector3( 2.8,  5.2, flight_z));
      waypoints.push_back(CVector3( 5.0,  6.9, flight_z));
      waypoints.push_back(CVector3( 8.0,  8.0, flight_z));

      current_wp = 1;

      initial_goal_dist = XYOnly(goal - start).Length();
      if(initial_goal_dist < 1e-6) {
         initial_goal_dist = 1.0;
      mission_complete = false;
      }

      LOG << "Loaded " << waypoints.size()
          << " waypoints | initial_goal_dist=" << initial_goal_dist
          << std::endl;
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

   CVector3 CurrentTarget() const {
      if(waypoints.empty()) {
         return goal;
      }

      UInt32 idx = std::min(current_wp, static_cast<UInt32>(waypoints.size() - 1));
      return waypoints[idx];
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

      /*
         If this is the final waypoint and the swarm core enters the goal radius,
         mark mission complete.
      */
      if(final_wp && dist < accept_radius) {
         mission_complete = true;

         LOG << "MISSION COMPLETE: core entered goal radius. "
             << "dist_goal=" << dist
             << " goal_accept_radius=" << goal_accept_radius
             << std::endl;

         return;
      }

      /*
         Intermediate waypoint switch.
         Larger acceptance radius prevents orbiting/chattering around waypoint.
      */
      if(!final_wp && dist < accept_radius) {
         current_wp++;
         last_dist_to_wp = 1e9;
         stuck_counter = 0;

         LOG << "Switching to waypoint " << current_wp << std::endl;
         return;
      }

      /*
         Stuck detector.
         If distance does not improve for too long, force the next waypoint.
         Do not skip the final goal completion condition.
      */
      if(dist > last_dist_to_wp - 0.004) {
         stuck_counter++;
      } else {
         stuck_counter = 0;
      }

      last_dist_to_wp = dist;

      if(stuck_counter > 450 && !final_wp) {
         current_wp++;
         stuck_counter = 0;
         last_dist_to_wp = 1e9;

         LOG << "Stuck detected. Forcing waypoint " << current_wp << std::endl;
      }
   }

   void InitDefinedFormation() {
      std::vector<CVector3> core_offsets = {
         CVector3( 0.00,  0.00, 0.00),
         CVector3( 0.25,  0.00, 0.00),
         CVector3(-0.25,  0.00, 0.00),
         CVector3( 0.00,  0.25, 0.00),
         CVector3( 0.00, -0.25, 0.00)
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

         Real th = 2.0 * ARGOS_PI * static_cast<Real>(j) / std::max(1, ndec);

         a.pos = SetFlightZ(start + CVector3(
            rring * std::cos(th),
            rring * std::sin(th),
            0.0
         ));

         a.vel = CVector3(0,0,0);
         a.slot = j;
         j++;
      }

      ApplyMovement();
      ApplyObstacleVisuals();

      initialized = true;

      LOG << "Defined ring formation initialized at z=" << flight_z << std::endl;
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

   CVector3 DecoySeparation(const Agent& a) const {
      CVector3 f(0,0,0);

      for(const auto& b : agents) {
         if(b.influential || a.id == b.id) continue;

         CVector3 d = XYOnly(a.pos - b.pos);
         Real L = d.Length();

         if(L < dec_sep_radius && L > 1e-6) {
            f += NormalizeSafeXY(d) * ((dec_sep_radius - L) / dec_sep_radius);
         }
      }

      return f;
   }

   CVector3 BandForceNearestInfluential(const Agent& a) const {
      Real best = std::numeric_limits<Real>::max();
      CVector3 nearest(0,0,flight_z);

      for(const auto& b : agents) {
         if(!b.influential) continue;

         Real d = XYOnly(a.pos - b.pos).Length();
         if(d < best) {
            best = d;
            nearest = b.pos;
         }
      }

      CVector3 radial = XYOnly(a.pos - nearest);
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

   CVector3 SlotTarget(const Agent& a, const CVector3& core) const {
      int ndec = CountDec();
      if(ndec <= 0 || a.slot < 0) {
         return core;
      }

      Real th0 = 2.0 * ARGOS_PI * static_cast<Real>(a.slot) / ndec;

      /*
         Almost no rotation. This avoids orbiting and crowding at the goal.
      */
      Real rot = 0.00025 * static_cast<Real>(step);
      Real th = th0 + rot;

      return SetFlightZ(core + CVector3(
         rring * std::cos(th),
         rring * std::sin(th),
         0.0
      ));
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

      for(auto& a : agents) {
         if(!a.influential) continue;

         CVector3 f_goal = target_dir * (k_goal_core * slow);
         CVector3 f_sep = InfluentialSeparation(a) * k_infl_sep;
         CVector3 f_cohesion = XYOnly(core - a.pos) * k_core_cohesion;
         CVector3 f_obs = ObstacleForceXY(a.pos, k_obs_infl);

         CVector3 f = f_goal + f_sep + f_cohesion + f_obs;

         a.vel = LimitXY(f, infl_step_max);
         a.pos = ProjectOutside(a.pos + a.vel);
      }
   }

   void UpdateDecoys() {
      CVector3 core = CoreCentroid();
      CVector3 target = CurrentTarget();
      CVector3 core_target_dir = NormalizeSafeXY(target - core);

      for(auto& a : agents) {
         if(a.influential) continue;

         CVector3 slot = SlotTarget(a, core);

         CVector3 f_shape = XYOnly(slot - a.pos) * k_shape;
         CVector3 f_band = BandForceNearestInfluential(a) * k_band;
         CVector3 f_sep = DecoySeparation(a) * k_dec_sep;
         CVector3 f_follow = core_target_dir * k_follow;
         CVector3 f_obs = ObstacleForceXY(a.pos, k_obs_dec);

         CVector3 radial = NormalizeSafeXY(a.pos - core);
         CVector3 tangent(-radial.GetY(), radial.GetX(), 0.0);
         CVector3 f_tangent = tangent * k_tangent;

         CVector3 f = f_shape + f_band + f_sep + f_follow + f_obs + f_tangent;

         a.vel = LimitXY(f, dec_step_max);
         a.pos = ProjectOutside(a.pos + a.vel);
      }
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

   Real Lambda2Proxy() const {
      const int n = static_cast<int>(agents.size());

      if(n < 2) return 0.0;

      std::vector<std::vector<int>> A(n, std::vector<int>(n, 0));

      for(int i = 0; i < n; ++i) {
         for(int j = i + 1; j < n; ++j) {
            Real d = XYOnly(agents[i].pos - agents[j].pos).Length();

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
         if(!visited[i]) return 0.0;
      }

      Real deg_sum = 0.0;

      for(int i = 0; i < n; ++i) {
         Real deg = 0.0;

         for(int j = 0; j < n; ++j) {
            deg += A[i][j];
         }

         deg_sum += deg;
      }

      return deg_sum / static_cast<Real>(n);
   }

   Real Clamp01(Real x) const {
      if(x < 0.0) return 0.0;
      if(x > 1.0) return 1.0;
      return x;
   }

   Real ProtectionScore() const {
      return std::exp(-AvgBandError() / std::max(tau, 1e-6));
   }

   Real ConnectivityScore() const {
      return Clamp01(Lambda2Proxy() / 25.0);
   }

   Real ObstacleSafetyScore() const {
      return Clamp01(MinObstacleDecoyDistance() / 0.80);
   }

   Real DecoySpacingScore() const {
      return Clamp01(MinDecoyDistance() / 0.22);
   }

   Real GoalProgressScore() const {
      CVector3 core = CoreCentroid();
      Real dist_goal = XYOnly(goal - core).Length();

      return Clamp01(1.0 - dist_goal / std::max(initial_goal_dist, 1e-6));
   }

   Real CriticalityScore() const {
      Real p_prot = ProtectionScore();
      Real p_conn = ConnectivityScore();
      Real p_obs  = ObstacleSafetyScore();
      Real p_sep  = DecoySpacingScore();
      Real p_goal = GoalProgressScore();

      Real score =
         0.35 * p_prot +
         0.20 * p_conn +
         0.20 * p_obs  +
         0.15 * p_sep  +
         0.10 * p_goal;

      return Clamp01(score);
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

      if(csv.is_open() && step % 5 == 0) {
         CVector3 core = CoreCentroid();
         CVector3 target = CurrentTarget();

         Real dist_wp   = XYOnly(target - core).Length();
         Real dist_goal = XYOnly(goal - core).Length();

         Real band_err = AvgBandError();
         Real min_dd   = MinDecoyDistance();
         Real min_od   = MinObstacleDecoyDistance();
         Real lambda2  = Lambda2Proxy();

         Real p_prot = ProtectionScore();
         Real p_conn = ConnectivityScore();
         Real p_obs  = ObstacleSafetyScore();
         Real p_sep  = DecoySpacingScore();
         Real p_goal = GoalProgressScore();
         Real crit   = CriticalityScore();

         csv << step << ","
             << current_wp << ","
             << core.GetX() << ","
             << core.GetY() << ","
             << core.GetZ() << ","
             << dist_wp << ","
             << dist_goal << ","
             << band_err << ","
             << min_dd << ","
             << min_od << ","
             << lambda2 << ","
             << p_prot << ","
             << p_conn << ","
             << p_obs << ","
             << p_sep << ","
             << p_goal << ","
             << crit << ","
             << (mission_complete ? 1 : 0)
             << "\n";
      }

      if(step % 50 == 0) {
         CVector3 core = CoreCentroid();
         CVector3 target = CurrentTarget();

         Real dist_wp   = XYOnly(target - core).Length();
         Real dist_goal = XYOnly(goal - core).Length();

         Real band_err = AvgBandError();
         Real min_dd   = MinDecoyDistance();
         Real min_od   = MinObstacleDecoyDistance();
         Real lambda2  = Lambda2Proxy();

         Real p_prot = ProtectionScore();
         Real p_conn = ConnectivityScore();
         Real p_obs  = ObstacleSafetyScore();
         Real p_sep  = DecoySpacingScore();
         Real p_goal = GoalProgressScore();
         Real crit   = CriticalityScore();

         LOG << "[t=" << step << "]"
             << " wp=" << current_wp
             << " core=(" << core.GetX() << "," << core.GetY() << "," << core.GetZ() << ")"
             << " dist_wp=" << dist_wp
             << " dist_goal=" << dist_goal
             << " band_err=" << band_err
             << " min_dd=" << min_dd
             << " min_od=" << min_od
             << " lambda2_proxy=" << lambda2
             << " prot=" << p_prot
             << " conn=" << p_conn
             << " obs_safe=" << p_obs
             << " sep_safe=" << p_sep
             << " goal_prog=" << p_goal
             << " criticality=" << crit
             << " mission_complete=" << (mission_complete ? 1 : 0)
             << std::endl;
      }
   }
};

REGISTER_LOOP_FUNCTIONS(CBTP_LoopFunctions, "loop");
