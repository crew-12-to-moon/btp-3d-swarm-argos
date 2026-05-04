#include <argos3/core/control_interface/ci_controller.h>

using namespace argos;

class CDecoyController : public CCI_Controller {
public:
   void Init(TConfigurationNode&) override {}
   void ControlStep() override {}
   void Reset() override {}
   void Destroy() override {}
};

REGISTER_CONTROLLER(CDecoyController, "dec");
