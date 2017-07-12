#include <iostream>
#include <boost/array.hpp>
#include <boost/numeric/odeint.hpp>

using namespace boost::numeric;

/**
 * @see https://www.ebi.ac.uk/biomodels-main/BIOMD0000000010
 */
struct MapkSystem {

  using StateType = boost::array<double, 8>;

  static constexpr StateType InitialState = {
      90.0,  // Mos
      10.0,  // Mos-P
      280.0, // Mek1
      10.0,  // Mek1-P
      10.0,  // Mek1-PP
      280.0, // Erk2
      10.0,  // Erk2-P
      10.0   // Erk2-PP
  };

  static constexpr double V1 = 2.5;
  static constexpr double Ki = 9.0;
  static constexpr double n = 1.0;
  static constexpr double K1 = 10.0;
  static constexpr double V2 = 0.25;
  static constexpr double KK2 = 8.0;
  static constexpr double k3 = 0.025;
  static constexpr double KK3 = 15.0;
  static constexpr double k4 = 0.025;
  static constexpr double KK4 = 15.0;
  static constexpr double V5 = 0.75;
  static constexpr double KK5 = 15.0;
  static constexpr double V6 = 0.75;
  static constexpr double KK6 = 15.0;
  static constexpr double k7 = 0.025;
  static constexpr double KK7 = 15.0;
  static constexpr double k8 = 0.025;
  static constexpr double KK8 = 15.0;
  static constexpr double V9 = 0.5;
  static constexpr double KK9 = 15.0;
  static constexpr double V10 = 0.5;
  static constexpr double KK10 = 15.0;

  void operator()(const StateType &x, StateType &dxdt, double t) {
    // d[Mos]/dt =
    //     - uVol*V1*Mos/((1+(Erk2-PP/Ki)^n)*(K1+Mos))
    //     + uVol*V2*Mos-P/(KK2+Mos-P)
    dxdt[0] =
        - V1 * x[0] / ((1.0 + pow(x[7] / Ki, n)) * (K1 + x[0]))
        + V2 * x[1] / (KK2 + x[1]);

    // d[Mos-P]/dt =
    //     uVol*V1*Mos/((1+(Erk2-PP/Ki)^n)*(K1+Mos))
    //     - uVol*V2*Mos-P/(KK2+Mos-P)
    dxdt[1] =
        V1 * x[0] / ((1.0 + pow(x[7] / Ki, n)) * (K1 + x[0]))
        - V2 * x[1] / (KK2 + x[1]);

    // d[Mek1]/dt =
    //     - uVol*k3*Mos-P*Mek1/(KK3+Mek1)
    //     + uVol*V6*Mek1-P/(KK6+Mek1-P)
    dxdt[2] =
        - k3 * x[1] * x[2] / (KK3 + x[2])
        + V6 * x[3] / (KK6 + x[3]);

    // d[Mek1-P]/dt =
    //     uVol*k3*Mos-P*Mek1/(KK3+Mek1)
    //     - uVol*k4*Mos-P*Mek1-P/(KK4+Mek1-P)
    //     + uVol*V5*Mek1-PP/(KK5+Mek1-PP)
    //     - uVol*V6*Mek1-P/(KK6+Mek1-P)
    dxdt[3] =
        k3 * x[1] * x[2] / (KK3 + x[2])
        - k4 * x[1] * x[3] / (KK4 + x[3])
        + V5 * x[4] / (KK5 + x[4])
        - V6 * x[3] / (KK6 + x[3]);

    // d[Mek1-PP]/dt =
    //     uVol*k4*Mos-P*Mek1-P/(KK4+Mek1-P)
    //     - uVol*V5*Mek1-PP/(KK5+Mek1-PP)
    dxdt[4] =
        k4 * x[1] * x[3] / (KK4 + x[3])
        - V5 * x[4] / (KK5 + x[4]);

    // d[Erk2]/dt =
    //     - uVol*k7*Mek1-PP*Erk2/(KK7+Erk2)
    //     + uVol*V10*Erk2-P/(KK10+Erk2-P)
    dxdt[5] =
        - k7 * x[4] * x[5] / (KK7 + x[5])
        + V10 * x[6] / (KK10 + x[6]);

    // d[Erk2-P]/dt =
    //     uVol*k7*Mek1-PP*Erk2/(KK7+Erk2)
    //     - uVol*k8*Mek1-PP*Erk2-P/(KK8+Erk2-P)
    //     + uVol*V9*Erk2-PP/(KK9+Erk2-PP)
    //     - uVol*V10*Erk2-P/(KK10+Erk2-P)
    dxdt[6] =
        k7 * x[4] * x[5] / (KK7 + x[5])
        - k8 * x[4] * x[6] / (KK8 + x[6])
        + V9 * x[7] / (KK9 + x[7])
        - V10 * x[6] / (KK10 + x[6]);

    // d[Erk2-PP]/dt =
    //     uVol*k8*Mek1-PP*Erk2-P/(KK8+Erk2-P)
    //     - uVol*V9*Erk2-PP/(KK9+Erk2-PP)
    dxdt[7] =
        k8 * x[4] * x[6] / (KK8 + x[6])
        - V9 * x[7] / (KK9 + x[7]);
  }

};

struct StdoutObserver {

  void operator()(const MapkSystem::StateType &x , const double t) {
    std::cout << t;

    auto size = x.size();
    for (auto i = 0; i < size; i++) {
      std::cout << " " << x[i];
    }

    std::cout << std::endl;
  }

};

void simulateWithRK4() {
  const double start = 0.0;
  const double duration = 1000.0;
  const double interval = 0.1;

  MapkSystem system;
  StdoutObserver observer;
  odeint::runge_kutta4<MapkSystem::StateType> stepper;

  auto state = MapkSystem::InitialState;

  odeint::integrate_const(stepper, system, state, start, duration, interval, std::ref(observer));
}

void simulateWithRKDopri() {
  const double start = 0.0;
  const double duration = 1000.0;
  const double interval = 0.1;

  MapkSystem system;
  StdoutObserver observer;
  odeint::runge_kutta_dopri5<MapkSystem::StateType> stepper;
  auto controlledStepper = odeint::make_controlled(0.0001, 0.0001, stepper);

  auto state = MapkSystem::InitialState;

  odeint::integrate_adaptive(controlledStepper, system, state, start, duration, interval, std::ref(observer));
}

int main(int argc, char **argv) {
  auto startTime = clock();

  //simulateWithRK4();
  simulateWithRKDopri();

  auto endTime = clock();
  std::cerr << "clock: " << (endTime - startTime) << std::endl;
}