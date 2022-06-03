# ToDo: Document this script.
from matplotlib.ticker import FormatStrFormatter

import pterasoftware as ps
from matplotlib import pyplot as plt

num_airplanes = 5

speed = 1.0
alpha = 5.0
x_spacing = 0.5
y_spacing = 0.5
root_to_mid_span = 0.2275
root_chord = 0.1094
mid_to_tip_span = 0.350 - 0.2275
tip_chord = 0.0219
flapping_amplitude = 15.0

period = x_spacing / speed
root_to_mid_chord = root_chord
mid_to_tip_chord = (root_chord + tip_chord) / 2

# Leave alpha zero here as the wing twist is used later to set alpha.
this_operating_point = ps.operating_point.OperatingPoint(
    velocity=speed,
    alpha=0,
)
this_operating_point_movement = ps.movement.OperatingPointMovement(
    base_operating_point=this_operating_point,
)
del this_operating_point

these_airplane_movements = []
row = None
position = None
offset_sign = None
for airplane_id in range(num_airplanes):
    if airplane_id == 0:
        row = 1
        position = ""
        offset_sign = 1
    elif airplane_id % 2 != 0:
        row += 1
        position = "Right "
        offset_sign = 1
    else:
        position = "Left "
        offset_sign = -1

    this_name = "Airplane " + position + str(row)

    offset = row - 1

    this_airplane = ps.geometry.Airplane(
        name=this_name,
        x_ref=offset * x_spacing,
        y_ref=offset_sign * offset * y_spacing,
        wings=[
            ps.geometry.Wing(
                name="Main Wing",
                symmetric=True,
                chordwise_spacing="uniform",
                x_le=offset * x_spacing,
                y_le=offset_sign * offset * y_spacing,
                wing_cross_sections=[
                    ps.geometry.WingCrossSection(
                        twist=alpha,
                        chord=root_chord,
                        airfoil=ps.geometry.Airfoil(name="naca0012"),
                        spanwise_spacing="uniform",
                    ),
                    ps.geometry.WingCrossSection(
                        twist=alpha,
                        y_le=root_to_mid_span,
                        chord=root_chord,
                        airfoil=ps.geometry.Airfoil(name="naca0012"),
                        spanwise_spacing="uniform",
                    ),
                    ps.geometry.WingCrossSection(
                        twist=alpha,
                        y_le=root_to_mid_span + mid_to_tip_span,
                        chord=tip_chord,
                        airfoil=ps.geometry.Airfoil(name="naca0012"),
                    ),
                ],
            ),
        ],
    )

    this_airplane_movement = ps.movement.AirplaneMovement(
        base_airplane=this_airplane,
        wing_movements=[
            ps.movement.WingMovement(
                base_wing=this_airplane.wings[0],
                wing_cross_sections_movements=[
                    ps.movement.WingCrossSectionMovement(
                        base_wing_cross_section=this_airplane.wings[
                            0
                        ].wing_cross_sections[0],
                    ),
                    ps.movement.WingCrossSectionMovement(
                        base_wing_cross_section=this_airplane.wings[
                            0
                        ].wing_cross_sections[1],
                        sweeping_amplitude=flapping_amplitude,
                        sweeping_period=period,
                        sweeping_spacing="sine",
                    ),
                    ps.movement.WingCrossSectionMovement(
                        base_wing_cross_section=this_airplane.wings[
                            0
                        ].wing_cross_sections[2],
                        sweeping_amplitude=flapping_amplitude,
                        sweeping_period=period,
                        sweeping_spacing="sine",
                    ),
                ],
            )
        ],
    )

    these_airplane_movements.append(this_airplane_movement)

    del this_airplane
    del this_airplane_movement

this_movement = ps.movement.Movement(
    airplane_movements=these_airplane_movements,
    operating_point_movement=this_operating_point_movement,
    num_steps=None,
    delta_time=None,
)

del these_airplane_movements

this_problem = ps.problems.UnsteadyProblem(
    movement=this_movement,
    only_final_results=True,
)

del this_movement

# 5% Convergence:
#   1 Airplane:
#       wake:               prescribed
#       cycles:             2
#       panel aspect ratio: 4 (2 and 1 spanwise panels)
#       chordwise panels:   3
#   3 Airplanes:
#       wake:               free
#       cycles:             2
#       panel aspect ratio: 4 (2 and 1 spanwise panels)
#       chordwise panels:   3
#   5 Airplanes:
#       wake:               free
#       cycles:             2
#       panel aspect ratio: 4 (2 and 1 spanwise panels)
#       chordwise panels:   3
# 1% Convergence:
#   1 Airplane:
#       wake:               free
#       cycles:             2
#       panel aspect ratio: 1 (6 and 6 spanwise panels)
#       chordwise panels:   3
#   3 Airplanes:
#       wake:               free
#       cycles:             3
#       panel aspect ratio: 1 (10 and 9 spanwise panels)
#       chordwise panels:   5
#   5 Airplanes:
#       wake:               free
#       cycles:             3
#       panel aspect ratio: 1 (10 and 9 spanwise panels)
#       chordwise panels:   5
converged_parameters = ps.convergence.analyze_unsteady_convergence(
    ref_problem=this_problem,
    coefficient_mask=[True, False, True, False, False, False],
    prescribed_wake=True,
    free_wake=True,
    num_cycles_bounds=(3, 4),
    panel_aspect_ratio_bounds=(2, 1),
    num_chordwise_panels_bounds=(5, 6),
    convergence_criteria=1.0,
)

(
    converged_wake,
    converged_length,
    converged_ar,
    converged_chord,
    wake_list,
    length_list,
    ar_list,
    chord_list,
    iter_times,
    coefficients,
) = converged_parameters

single_flap = len(length_list) == 1
single_ar = len(ar_list) == 1
single_chord = len(chord_list) == 1

converged_wake_id = wake_list.index(converged_wake)
converged_length_id = length_list.index(converged_length)
converged_ar_id = ar_list.index(converged_ar)
converged_chord_id = chord_list.index(converged_chord)

if not single_flap:
    force_figure, force_axes = plt.subplots()
    moment_figure, moment_axes = plt.subplots()

    row = None
    for airplane_id in range(num_airplanes):
        if airplane_id == 0:
            row = 1
        elif airplane_id % 2 == 0:
            row += 1
        else:
            continue

        this_force = coefficients[
            converged_wake_id,
            : converged_length_id + 2,
            converged_ar_id,
            converged_chord_id,
            airplane_id,
            :3,
        ]
        this_moment = coefficients[
            converged_wake_id,
            : converged_length_id + 2,
            converged_ar_id,
            converged_chord_id,
            airplane_id,
            3:,
        ]

        force_axes.plot(
            length_list[: converged_length_id + 2],
            this_force,
            label="Row " + str(row),
            marker="o",
            linestyle="--",
        )
        moment_axes.plot(
            length_list[: converged_length_id + 2],
            this_moment,
            label="Row " + str(row),
            marker="o",
            linestyle="--",
        )

    force_axes.set_xlabel("Number of Flap Cycles")
    moment_axes.set_xlabel("Number of Flap Cycles")

    force_axes.set_ylabel("Final Cycle-Averaged Force Coefficient")
    moment_axes.set_ylabel("Final Cycle-Averaged Moment Coefficient")

    force_axes.set_title("Number of Flap Cycles\nForce Convergence")
    moment_axes.set_title("Number of Flap Cycles\nMoment Convergence")

    force_axes.yaxis.set_major_formatter(FormatStrFormatter("%.4f"))
    moment_axes.yaxis.set_major_formatter(FormatStrFormatter("%.4f"))

    force_axes.legend(loc="lower left")
    moment_axes.legend(loc="lower left")

    force_figure.show()
    moment_figure.show()

if not single_ar:
    force_figure, force_axes = plt.subplots()
    moment_figure, moment_axes = plt.subplots()

    row = None
    for airplane_id in range(num_airplanes):
        if airplane_id == 0:
            row = 1
        elif airplane_id % 2 == 0:
            row += 1
        else:
            continue

        this_force = coefficients[
            converged_wake_id,
            converged_length_id,
            : converged_ar_id + 2,
            converged_chord_id,
            airplane_id,
            0:2,
        ]
        this_moment = coefficients[
            converged_wake_id,
            converged_length_id,
            : converged_ar_id + 2,
            converged_chord_id,
            airplane_id,
            3:5,
        ]

        force_axes.plot(
            ar_list[: converged_ar_id + 2],
            this_force,
            label="Row " + str(row),
            marker="o",
            linestyle="--",
        )
        moment_axes.plot(
            ar_list[: converged_ar_id + 2],
            this_moment,
            label="Row " + str(row),
            marker="o",
            linestyle="--",
        )

    min_ar = min(ar_list[: converged_ar_id + 2])
    max_ar = max(ar_list[: converged_ar_id + 2])
    ar_range = max_ar - min_ar
    ar_pad = ar_range * 0.25
    x_min = max_ar + ar_pad
    x_max = min_ar - ar_pad
    force_axes.set_xlim(x_min, x_max)
    moment_axes.set_xlim(x_min, x_max)

    force_axes.set_xlabel("Panel Aspect Ratio")
    moment_axes.set_xlabel("Panel Aspect Ratio")

    force_axes.set_ylabel("Final Cycle-Averaged Force Coefficient")
    moment_axes.set_ylabel("Final Cycle-Averaged Moment Coefficient")

    force_axes.set_title("Panel Aspect Ratio\nForce Convergence")
    moment_axes.set_title("Panel Aspect Ratio\nMoment Convergence")

    force_axes.yaxis.set_major_formatter(FormatStrFormatter("%.4f"))
    moment_axes.yaxis.set_major_formatter(FormatStrFormatter("%.4f"))

    force_axes.legend(loc="lower left")
    moment_axes.legend(loc="lower left")

    force_figure.show()
    moment_figure.show()

if not single_chord:
    force_figure, force_axes = plt.subplots()
    moment_figure, moment_axes = plt.subplots()

    row = None
    for airplane_id in range(num_airplanes):
        if airplane_id == 0:
            row = 1
        elif airplane_id % 2 == 0:
            row += 1
        else:
            continue

        this_force = coefficients[
            converged_wake_id,
            converged_length_id,
            converged_ar_id,
            : converged_chord_id + 2,
            airplane_id,
            0:2,
        ]
        this_moment = coefficients[
            converged_wake_id,
            converged_length_id,
            converged_ar_id,
            : converged_chord_id + 2,
            airplane_id,
            3:5,
        ]

        force_axes.plot(
            chord_list[: converged_chord_id + 2],
            this_force,
            label="Row " + str(row),
            marker="o",
            linestyle="--",
        )
        moment_axes.plot(
            chord_list[: converged_chord_id + 2],
            this_moment,
            label="Row " + str(row),
            marker="o",
            linestyle="--",
        )

        force_axes.set_xlabel("Number of Chordwise Panels")
        moment_axes.set_xlabel("Number of Chordwise Panels")

        force_axes.set_ylabel("Final Cycle-Averaged Force Coefficient")
        moment_axes.set_ylabel("Final Cycle-Averaged Moment Coefficient")

        force_axes.set_title("Number of Chordwise Panels\nForce Convergence")
        moment_axes.set_title("Number of Chordwise Panels\nMoment Convergence")

        force_axes.yaxis.set_major_formatter(FormatStrFormatter("%.4f"))
        moment_axes.yaxis.set_major_formatter(FormatStrFormatter("%.4f"))

        force_axes.legend(loc="lower left")
        moment_axes.legend(loc="lower left")

        force_figure.show()
        moment_figure.show()
