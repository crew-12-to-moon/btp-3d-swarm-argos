#include <argos3/core/control_interface/ci_controller.h>

using namespace argos;

class CInfluentialController : public CCI_Controller {
public:
   void Init(TConfigurationNode&) override {}
   void ControlStep() override {}
   void Reset() override {}
   void Destroy() override {}
};

REGISTER_CONTROLLER(CInfluentialController, "infl");
