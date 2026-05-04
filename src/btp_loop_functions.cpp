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
   CVector3 center;
   CVector3 half;
   bool dynamic = false;
   CVector3 amp;
   Real freq = 0.0;
   Real phase = 0.0;
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
      Mission setup
   */
   Real flight_z = 1.15;

   CVector3 start = CVector3(-3.0, -3.0, 1.15);
   CVector3 goal  = CVector3( 3.0,  3.0, 1.15);

   /*
      Protection band
      Larger ring radius avoids decoy-decoy crowding.
   */
   Real rmin  = 0.65;
   Real rmax  = 1.45;
   Real rring = 1.12;
   Real tau   = 0.45;

   /*
      Step sizes are position increments per ARGoS tick.
      Keep these small enough to avoid bouncing.
   */
   Real infl_step_max = 0.026;
   Real dec_step_max  = 0.042;

   /*
      Influential core gains
   */
   Real k_goal_core     = 0.050;
   Real k_core_cohesion = 0.012;
   Real k_infl_sep      = 0.014;
   Real k_obs_infl      = 0.045;

   /*
      Decoy gains
   */
   Real k_shape   = 0.095;
   Real k_band    = 0.065;
   Real k_dec_sep = 0.035;
   Real k_follow  = 0.025;
   Real k_tangent = 0.002;
   Real k_obs_dec = 0.050;

   Real infl_sep_radius = 0.25;
   Real dec_sep_radius  = 0.22;

   /*
      Obstacle SDF parameters
   */
   Real obs_safe      = 0.28;
   Real obs_influence = 1.00;

   /*
      Arena clamp
   */
   Real xmin = -4.8;
   Real xmax =  4.8;
   Real ymin = -4.8;
   Real ymax =  4.8;
   Real zmin =  0.70;
   Real zmax =  2.20;

   /*
      Connectivity proxy graph radius
   */
   Real graph_radius = 1.65;

   /*
      Stuck detection
   */
   Real last_dist_to_wp = 1e9;
   UInt32 stuck_counter = 0;

   void Init(TConfigurationNode&) override {
      LOG << "BTP 3D sim initialized: waypoint navigation + fixed altitude + ring protection" << std::endl;

      LoadAgents();
      InitObstacles();
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

      if(csv.is_open()) csv.close();

      LoadAgents();
      InitObstacles();
      InitWaypoints();
      InitDefinedFormation();
      OpenCSV();
   }

   void Destroy() override {
      if(csv.is_open()) csv.close();
   }

   void OpenCSV() {
      csv.open("btp_argos_metrics.csv");
      csv << "step,wp_index,core_x,core_y,core_z,dist_wp,dist_goal,avg_band_error,min_decoy_decoy,min_obstacle_decoy,lambda2_proxy\n";
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
         These match the visual buildings in test.argos.
      */
      obstacles.push_back({CVector3(-1.7, -0.6, 0.8), CVector3(0.45, 0.80, 0.80), false});
      obstacles.push_back({CVector3( 1.4, -0.3, 0.9), CVector3(0.50, 0.65, 0.90), false});
      obstacles.push_back({CVector3(-0.8,  1.6, 0.9), CVector3(0.70, 0.45, 0.90), false});
      obstacles.push_back({CVector3( 2.2,  1.8, 0.8), CVector3(0.45, 0.55, 0.80), false});

      /*
         Internal moving aerial obstacles.
         They are not visualized yet, but the controller avoids them.
      */
      obstacles.push_back({CVector3(0.2, 0.8, 1.6), CVector3(0.25, 0.25, 0.25), true,
                           CVector3(0.80, 0.00, 0.20), 0.035, 0.0});

      obstacles.push_back({CVector3(1.0, -1.4, 1.5), CVector3(0.30, 0.20, 0.25), true,
                           CVector3(0.00, 0.80, 0.15), 0.025, 1.2});
   }

   void InitWaypoints() {
      waypoints.clear();

      /*
         Safe route around the building cluster.
         This prevents the classic potential-field local minimum.
      */
      waypoints.push_back(CVector3(-3.0, -3.0, flight_z));
      waypoints.push_back(CVector3(-3.3, -0.6, flight_z));
      waypoints.push_back(CVector3(-2.5,  2.8, flight_z));
      waypoints.push_back(CVector3( 0.6,  3.4, flight_z));
      waypoints.push_back(CVector3( 3.0,  3.0, flight_z));

      current_wp = 1;
      LOG << "Loaded " << waypoints.size() << " waypoints" << std::endl;
   }

   int CountInfl() const {
      int c = 0;
      for(const auto& a : agents) if(a.influential) c++;
      return c;
   }

   int CountDec() const {
      int c = 0;
      for(const auto& a : agents) if(!a.influential) c++;
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

   CVector3 NormalizeSafe(const CVector3& v) const {
      Real L = v.Length();
      if(L < 1e-9) return CVector3(0,0,0);
      return v / L;
   }

   CVector3 NormalizeSafeXY(const CVector3& v) const {
      CVector3 u = v;
      u.SetZ(0.0);
      Real L = u.Length();
      if(L < 1e-9) return CVector3(0,0,0);
      return u / L;
   }

   CVector3 Limit(const CVector3& v, Real max_mag) const {
      Real L = v.Length();
      if(L > max_mag && L > 1e-9) return (v / L) * max_mag;
      return v;
   }

   CVector3 LimitXY(CVector3 v, Real max_mag) const {
      v.SetZ(0.0);
      Real L = v.Length();
      if(L > max_mag && L > 1e-9) return (v / L) * max_mag;
      return v;
   }

   CVector3 Clamp(CVector3 p) const {
      p.SetX(std::max(xmin, std::min(xmax, p.GetX())));
      p.SetY(std::max(ymin, std::min(ymax, p.GetY())));
      p.SetZ(flight_z);
      return p;
   }

   CVector3 ObsCenter(const BoxObs& o) const {
      if(!o.dynamic) return o.center;

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
               mag = 2.0;
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
            p += n * (-phi + 0.02);
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

      if(n > 0) c /= n;
      c.SetZ(flight_z);
      return c;
   }

   CVector3 CurrentTarget() const {
      if(waypoints.empty()) return goal;
      UInt32 idx = std::min(current_wp, static_cast<UInt32>(waypoints.size() - 1));
      return waypoints[idx];
   }

   void UpdateWaypoint() {
      if(waypoints.empty()) return;

      CVector3 core = CoreCentroid();
      CVector3 target = CurrentTarget();

      Real dist = (target - core).Length();

      /*
         Move to next waypoint when close.
      */
      if(dist < 0.45 && current_wp < waypoints.size() - 1) {
         current_wp++;
         last_dist_to_wp = 1e9;
         stuck_counter = 0;

         LOG << "Switching to waypoint " << current_wp << std::endl;
         return;
      }

      /*
         Stuck escape:
         if distance does not improve for too long, force next waypoint.
      */
      if(dist > last_dist_to_wp - 0.003) {
         stuck_counter++;
      } else {
         stuck_counter = 0;
      }

      last_dist_to_wp = dist;

      if(stuck_counter > 350 && current_wp < waypoints.size() - 1) {
         current_wp++;
         stuck_counter = 0;
         last_dist_to_wp = 1e9;

         LOG << "Stuck detected. Forcing waypoint " << current_wp << std::endl;
      }
   }

   void InitDefinedFormation() {
      std::vector<CVector3> core_offsets = {
         CVector3( 0.00,  0.00, 0.00),
         CVector3( 0.20,  0.00, 0.00),
         CVector3(-0.20,  0.00, 0.00),
         CVector3( 0.00,  0.20, 0.00),
         CVector3( 0.00, -0.20, 0.00)
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
      initialized = true;

      LOG << "Defined ring formation initialized at fixed altitude z=" << flight_z << std::endl;
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

      if(d < 1e-6) return CVector3(0.05,0,0);

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
      if(ndec <= 0 || a.slot < 0) return core;

      Real th0 = 2.0 * ARGOS_PI * static_cast<Real>(a.slot) / ndec;

      /*
         Very slow rotation only; too much rotation causes bouncing.
      */
      Real rot = 0.0006 * static_cast<Real>(step);
      Real th = th0 + rot;

      return SetFlightZ(core + CVector3(
         rring * std::cos(th),
         rring * std::sin(th),
         0.0
      ));
   }

   void UpdateInfluentials() {
      CVector3 core = CoreCentroid();
      CVector3 target = CurrentTarget();

      CVector3 to_target = XYOnly(target - core);
      CVector3 target_dir = NormalizeSafeXY(to_target);

      Real dist = to_target.Length();
      Real slow = std::min(1.0, dist / 1.0);

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

         if(best < rmin) err = rmin - best;
         else if(best > rmax) err = best - rmax;

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
         for(int j = 0; j < n; ++j) deg += A[i][j];
         deg_sum += deg;
      }

      return deg_sum / static_cast<Real>(n);
   }

   void ApplyMovement() {
      CQuaternion q;
      q.FromEulerAngles(CRadians(0), CRadians(0), CRadians(0));

      for(auto& a : agents) {
         a.pos = SetFlightZ(a.pos);
         a.entity->GetEmbodiedEntity().MoveTo(a.pos, q, false);
      }
   }

   void PostStep() override {
      step++;

      if(!initialized) InitDefinedFormation();

      UpdateWaypoint();

      UpdateInfluentials();
      UpdateDecoys();
      ApplyMovement();

      if(csv.is_open() && step % 5 == 0) {
         CVector3 core = CoreCentroid();
         CVector3 target = CurrentTarget();

         csv << step << ","
             << current_wp << ","
             << core.GetX() << ","
             << core.GetY() << ","
             << core.GetZ() << ","
             << XYOnly(target - core).Length() << ","
             << XYOnly(goal - core).Length() << ","
             << AvgBandError() << ","
             << MinDecoyDistance() << ","
             << MinObstacleDecoyDistance() << ","
             << Lambda2Proxy()
             << "\n";
      }

      if(step % 50 == 0) {
         CVector3 core = CoreCentroid();
         CVector3 target = CurrentTarget();

         LOG << "[t=" << step << "]"
             << " wp=" << current_wp
             << " core=(" << core.GetX() << "," << core.GetY() << "," << core.GetZ() << ")"
             << " dist_wp=" << XYOnly(target - core).Length()
             << " dist_goal=" << XYOnly(goal - core).Length()
             << " band_err=" << AvgBandError()
             << " min_dd=" << MinDecoyDistance()
             << " min_od=" << MinObstacleDecoyDistance()
             << " lambda2_proxy=" << Lambda2Proxy()
             << std::endl;
      }
   }
};

REGISTER_LOOP_FUNCTIONS(CBTP_LoopFunctions, "loop");
