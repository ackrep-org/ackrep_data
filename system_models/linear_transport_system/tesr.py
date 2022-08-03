import pyinduct as pi
import numpy as np
from ipydex import IPS
import scipy
import matplotlib.pyplot as plt
import pyqtgraph as pg

sys_name = "transport_system"


# --- modelling ---
v = 4
l = 5
T = 5
spat_bounds = (0, l)
spat_domain = pi.Domain(bounds=spat_bounds, num=51)
temp_domain = pi.Domain(bounds=(0, T), num=100)

init_x = pi.Function(lambda z: 0, domain=spat_bounds)

init_funcs = pi.LagrangeFirstOrder.cure_interval(spat_domain)
func_label = "init_funcs"
pi.register_base(func_label, init_funcs)
# inpuc function
u = pi.SimulationInputSum(
    [
        pi.SignalGenerator("square", np.array(temp_domain), frequency=0.1, scale=1, offset=1, phase_shift=1),
        pi.SignalGenerator("square", np.array(temp_domain), frequency=0.2, scale=2, offset=2, phase_shift=2),
        pi.SignalGenerator("square", np.array(temp_domain), frequency=0.3, scale=3, offset=3, phase_shift=3),
        pi.SignalGenerator("square", np.array(temp_domain), frequency=0.4, scale=4, offset=4, phase_shift=4),
        pi.SignalGenerator("square", np.array(temp_domain), frequency=0.5, scale=5, offset=5, phase_shift=5),
    ]
)

x = pi.FieldVariable(func_label)
phi = pi.TestFunction(func_label)

weak_form = pi.WeakFormulation(
    [
        pi.IntegralTerm(pi.Product(x.derive(temp_order=1), phi), spat_bounds),
        pi.IntegralTerm(pi.Product(x, phi.derive(1)), spat_bounds, scale=-v),
        pi.ScalarTerm(pi.Product(x(l), phi(l)), scale=v),
        pi.ScalarTerm(pi.Product(pi.Input(u), phi(0)), scale=-v),
    ],
    name=sys_name,
)

ics = pi.sanitize_input(init_x, pi.core.Function)
initial_states = {weak_form.name: ics}
spatial_domains = {weak_form.name: spat_domain}
derivative_orders = {weak_form.name: (0, 0)}

weak_forms = pi.sanitize_input([weak_form], pi.WeakFormulation)
print("simulate systems: {}".format([f.name for f in weak_forms]))

print(">>> parse weak formulations")
canonical_equations = pi.parse_weak_formulations(weak_forms)

print(">>> create state space system")
state_space_form = pi.create_state_space(canonical_equations)

# --- simulation ---

print(">>> derive initial conditions")
q0 = pi.core.project_on_bases(initial_states, canonical_equations)

print(">>> perform time step integration")
sim_domain, q = pi.simulate_state_space(state_space_form, q0, temp_domain, settings=None)

print(">>> perform postprocessing")
eval_data = pi.get_sim_results(sim_domain, spatial_domains, q, state_space_form, derivative_orders=derivative_orders)

evald_x = pi.evaluate_approximation(func_label, q, sim_domain, spat_domain, name="x(z,t)")

pi.tear_down(labels=(func_label,))


# pyqtgraph visualization
win0 = pi.surface_plot(evald_x, zlabel=evald_x.name)

# IPS()

# pyqtgraph visualization
win1 = pg.plot(
    np.array(eval_data[0].input_data[0]).flatten(),
    u.get_results(eval_data[0].input_data[0]).flatten(),
    labels=dict(left="u(t)", bottom="t"),
    pen="b",
)

plt.plot(np.array(eval_data[0].input_data[0]).flatten(), u.get_results(eval_data[0].input_data[0]).flatten())
plt.show()
# IPS()

win1.showGrid(x=False, y=True, alpha=0.5)

import pyqtgraph.exporters as exp

exporter = exp.ImageExporter(win1.plotItem)
# exporter.export("input.png")


# vis.save_2d_pg_plot(win0, 'transport_system')
# win2 = pi.PgAnimatedPlot(eval_data,
#                             title=eval_data[0].name,
#                             save_pics=True,
#                             create_video=True,
#                             labels=dict(left='x(z,t)', bottom='z'))
# pi.show()
