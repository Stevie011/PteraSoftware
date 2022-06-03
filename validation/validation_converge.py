# ToDo: Update this script's docstring and documentation.
"""This script runs a validation case of Ptera Software’s UVLM.

I first emulate the geometry and kinematics of a flapping robotic test stand from
"Experimental and Analytical Pressure Characterization of a Rigid Flapping Wing for
Ornithopter Development" by Derrick Yeo, Ella M. Atkins, and Wei Shyy. Then,
I run the UVLM simulation of an experiment from this paper. Finally, I compare the
simulated results to the published experimental results.

WebPlotDigitizer, by Ankit Rohatgi, was used to extract data from Yeo et al., 2011.

More information can be found in my accompanying report: "Validating an Open-Source
UVLM Solver for Analyzing Flapping Wing Flight: An Experimental Approach." """

import time
import logging
import numpy as np
import pterasoftware as ps

validation_logger = logging.getLogger("validation")
validation_logger.setLevel(logging.INFO)
logging.basicConfig()

# Set the given flapping frequency in Hertz.
validation_flapping_frequency = 3.3

# Set the given characteristics of the wing in meters.
half_span = 0.213
chord = 0.072

# Set the given forward flight velocity in meters per second.
validation_velocity = 2.9

# Set the given angle of attack in degrees. Note: If you analyze a different
# operating point where this is not zero, you need to modify the code to rotate the
# experimental lift into the wind axes.
validation_alpha = 0

# This wing planform has a rounded tip so the outermost wing cross section needs to
# be inset some amount. This value is in meters.
tip_inset = 0.005

# Import the extracted coordinates from the paper’s diagram of the planform. The
# resulting array is of the form [spanwise coordinate, chordwise coordinate],
# and is ordered from the leading edge root, to the tip, to the trailing edge root.
# The origin is the trailing edge root point. The positive spanwise axis extends from
# root to tip and the positive chordwise axis from trailing edge to leading edge. The
# coordinates are in millimeters.
planform_coords = np.genfromtxt("extracted_planform_coordinates.csv", delimiter=",")

# Convert the coordinates to meters.
planform_coords = planform_coords / 1000

# Set the origin to the leading edge root point.
planform_coords = planform_coords - np.array([0, chord])

# Switch the sign of the chordwise coordinates.
planform_coords = planform_coords * np.array([1, -1])

# Swap the axes to the form [chordwise coordinate, spanwise coordinate]. The
# coordinates are now in the geometry frame projected on the XY plane.
planform_coords[:, [0, 1]] = planform_coords[:, [1, 0]]

# Find the index of the point where the planform x-coordinate equals the half span.
tip_index = np.where(planform_coords[:, 1] == half_span)[0][0]

# Using the tip index, split the coordinates into two arrays of leading and trailing
# edge coordinates.
leading_coords = planform_coords[:tip_index, :]
trailing_coords = np.flip(planform_coords[tip_index:, :], axis=0)


# ToDo: Document this function.
def normalized_validation_geometry_sweep_function(this_time):
    """This function takes in the time during a flap cycle and returns the flap angle
    in degrees. It uses a normalized flapping frequency of 1 Hertz, and is based on a
    fourth-order Fourier series. The coefficients were calculated by the authors of
    Yeo et al., 2011.

    :param this_time: float or 1D array of floats
        This is a single time or an array of time values at which to calculate the
        flap angle. The units are seconds.
    :return flap_angle: float or 1D array of floats
        This is a single flap angle or an array of flap angle values at the inputted
        time value or values. The units are degrees.
    """

    # Set the Fourier series coefficients.
    a_0 = 0.0354
    a_1 = 4.10e-5
    b_1 = 0.3793
    a_2 = -0.0322
    b_2 = -1.95e-6
    a_3 = -8.90e-7
    b_3 = -0.0035
    a_4 = 0.00046
    b_4 = -3.60e-6

    # Calculate and return the flap angle(s).
    flap_angle = (
        -(
            a_0
            + a_1 * np.cos(1 * this_time)
            + b_1 * np.sin(1 * this_time)
            + a_2 * np.cos(2 * this_time)
            + b_2 * np.sin(2 * this_time)
            + a_3 * np.cos(3 * this_time)
            + b_3 * np.sin(3 * this_time)
            + a_4 * np.cos(4 * this_time)
            + b_4 * np.sin(4 * this_time)
        )
        / 0.0174533
    )
    return flap_angle


# ToDo: Document this function.
def analyze_convergence(
    coefficient_mask,
    prescribed_wake=True,
    free_wake=True,
    num_cycles_bounds=(1, 4),
    panel_aspect_ratio_bounds=(4, 1),
    num_chordwise_panels_bounds=(3, 12),
    convergence_criteria=5.0,
):
    """This function finds the converged parameters of an unsteady problem.

    Convergence is found by varying if the solver's wake state (prescribed or free),
    the final length of the problem's wake (in number of chord lengths for static
    geometry and number of flap cycles for variable geometry), the airplanes' panel
    aspect ratios, and the airplanes' numbers of chordwise panels. These values are
    iterated over via four nested loops. The outermost loop is the wake state. The
    next loop is the wake length. The loop after that is the panel aspect ratios,
    and the final loop is the number of chordwise panels.

    With each new combination of these values, the problem is solved, and its
    resultant force and moment coefficients are stored. The force coefficients are
    combined by taking the vector norm. This is repeated for the moment coefficients.
    Then, absolute percent change (APE) of the resultant force coefficient is found
    between this interation, and the iterations with the incrementally coarser meshes
    in all four parameters (wake state, wake length, panel aspect ratio,
    and number of chordwise panels). The process is repeated for to find the
    resultant moment coefficient APE.

    The maximums of the resultant force coefficient APEs and resultant moment
    coefficient APEs are found. This leaves us with four maximum APEs, one for each
    parameter. If any of the parameter's APE is below the iteration has found a
    converged solution for that parameter.

    If an iteration's four APEs are all below the converged criteria, then the solver
    will exit the loops and return the converged parameters. However, the converged
    parameters are actually the values incrementally coarser than the final values (
    because the incrementally coarser values were found to be within the convergence
    criteria percent difference from the final values).

    There are two edge cases to this function. The first occurs when the user
    indicates that they only want check a single value for any of the four parameters
    (i.e. panel_aspect_ratio_bounds=(2, 2) or (prescribed_wake=True and
    free_wake=False)). Then, this parameter will not be iterated over,
    and convergence will only be checked for the other parameters.

    The second edge case happens if the panel aspect ratio has not converged at a
    value of 1 or if the wake state hasn't converged once it is equal to a free wake.
    These conditions are the gold standards for panel aspect ratio and wake state,
    so the solver will return 1 for the converged value of panel aspect ratio and a
    free wake for the converged wake state. In the code below, this state is referred
    to as a "saturated" panel aspect ratio or wake state.

    :param prescribed_wake: bool, optional
        This parameter determines if a prescribed wake type should be analyzed. If
        this parameter is false, then the free_wake parameter must be set to True.
        The default value is True.
    :param free_wake: bool, optional
        This parameter determines if a free wake type should be analyzed.  If this
        parameter is false, then the prescribed_wake parameter must be set to
        True. The default value is True.
    :param num_cycles_bounds: tuple, optional
        This parameter determines the range of wake lengths, measured in number of
        cycles to simulate. If the problem has static geometry, it will be ignored,
        and the num_chords_bounds parameter will control the wake lengths instead.
        Reasonable values range from 1 to 10, depending strongly on the Strouhal
        number. The first value must be less than or equal to the second value. The
        default value is (1, 4).
    :param panel_aspect_ratio_bounds: tuple, optional
        This parameter determines the range of panel aspect ratios, from largest to
        smallest. For a given wing section, this value dictates the average panel
        body-frame-y length divided by the average body-frame-x width. Historically,
        these values range between 5 and 1. Values above 5 can be uses for a coarser
        mesh, but the minimum value should not be less than 1. The first value must
        be greater than or equal to the second value. The default value is ( , 1).
    :param num_chordwise_panels_bounds: tuple, optional
        This parameter determines the range of each wing section's number of
        chordwise panels from smallest to largest. The first value must be less than
        or equal to the second value. The default value is (3, 12).
    :param convergence_criteria: float, optional
        This parameter determines at what point the function continues the problem
        converged. Specifically, it is the absolute percent change in the resultant
        force coefficient or moment coefficient (whichever is higher). Therefore,
        it is in units of percent. Refer to the description above for more details on
        how it affects the solver. In short, set this value to 5.0 for a lenient
        convergence, and 1.0 for a strict convergence. The default value is 5.0.
    :return: list
        This function returns a list of four ints. In order, they are the converged
        wake state, the converged wake length, the converged of panel aspect ratio
        and the converged number of chordwise panels. If the function could not find
        a set of converged parameters, it returns values of None for all items in the
        list.
    """
    validation_logger.info("Beginning convergence analysis.")

    wake_list = []
    if prescribed_wake:
        wake_list.append(True)
    if free_wake:
        wake_list.append(False)

    wake_lengths_list = list(range(num_cycles_bounds[0], num_cycles_bounds[1] + 1))

    panel_aspect_ratios_list = list(
        range(panel_aspect_ratio_bounds[0], panel_aspect_ratio_bounds[1] - 1, -1)
    )
    num_chordwise_panels_list = list(
        range(num_chordwise_panels_bounds[0], num_chordwise_panels_bounds[1] + 1)
    )

    # Initialize some empty arrays to hold attributes regarding each iteration. Going
    # forward, an "iteration" refers to a problem containing one of the combinations
    # of the wake state, wake length, panel aspect ratio, and number of chordwise
    # panels parameters.
    iter_times = np.zeros(
        (
            len(wake_list),
            len(wake_lengths_list),
            len(panel_aspect_ratios_list),
            len(num_chordwise_panels_list),
        )
    )
    coefficients = np.zeros(
        (
            len(wake_list),
            len(wake_lengths_list),
            len(panel_aspect_ratios_list),
            len(num_chordwise_panels_list),
            1,
            6,
        )
    )

    iteration = 0
    num_iterations = (
        len(wake_list)
        * len(wake_lengths_list)
        * len(panel_aspect_ratios_list)
        * len(num_chordwise_panels_list)
    )

    # Begin iterating through the first loop of wake states.
    for wake_id, wake in enumerate(wake_list):
        if wake:
            validation_logger.info("Wake type: prescribed")
        else:
            validation_logger.info("Wake type: free")

        # Begin iterating through the second loop of wake lengths.
        for length_id, wake_length in enumerate(wake_lengths_list):
            validation_logger.info("\tCycles: " + str(wake_length))

            # Begin iterating through the third loop of panel aspect ratios.
            for ar_id, panel_aspect_ratio in enumerate(panel_aspect_ratios_list):
                validation_logger.info(
                    "\t\tPanel aspect ratio: " + str(panel_aspect_ratio)
                )

                # Begin iterating through the fourth and innermost loop of number of
                # chordwise panels.
                for chord_id, num_chordwise_panels in enumerate(
                    num_chordwise_panels_list
                ):
                    validation_logger.info(
                        "\t\t\tChordwise panels: " + str(num_chordwise_panels)
                    )

                    iteration += 1
                    validation_logger.info(
                        "\t\t\t\tIteration Number: "
                        + str(iteration)
                        + "/"
                        + str(num_iterations)
                    )

                    these_ids = (wake_id, length_id, ar_id, chord_id)

                    # As we can't directly specify the panel aspect
                    # ratio, calculate the number of spanwise panels that
                    # corresponds to the desired panel aspect ratio.
                    span = half_span * 2
                    wing_area = 0.02452052069501779
                    standard_mean_chord = wing_area / span

                    this_num_spanwise_sections = round(
                        (span * num_chordwise_panels)
                        / (standard_mean_chord * panel_aspect_ratio)
                        / 2
                    )

                    # Calculate the spanwise difference between the wing cross sections.
                    spanwise_step = (half_span - tip_inset) / this_num_spanwise_sections

                    # Define four arrays to hold the coordinates of the front and
                    # back points of each
                    # section’s left and right wing cross sections.
                    front_left_vertices = np.zeros((this_num_spanwise_sections, 2))
                    front_right_vertices = np.zeros((this_num_spanwise_sections, 2))
                    back_left_vertices = np.zeros((this_num_spanwise_sections, 2))
                    back_right_vertices = np.zeros((this_num_spanwise_sections, 2))

                    # Iterate through the locations of the future sections to
                    # populate the wing cross
                    # section coordinates.
                    for spanwise_loc in range(this_num_spanwise_sections):
                        # Find the y-coordinates of the vertices.
                        front_left_vertices[spanwise_loc, 1] = (
                            spanwise_loc * spanwise_step
                        )
                        back_left_vertices[spanwise_loc, 1] = (
                            spanwise_loc * spanwise_step
                        )
                        front_right_vertices[spanwise_loc, 1] = (
                            spanwise_loc + 1
                        ) * spanwise_step
                        back_right_vertices[spanwise_loc, 1] = (
                            spanwise_loc + 1
                        ) * spanwise_step

                        # Interpolate between the leading edge coordinates to find
                        # the x-coordinate of
                        # the front left vertex.
                        front_left_vertices[spanwise_loc, 0] = np.interp(
                            spanwise_loc * spanwise_step,
                            leading_coords[:, 1],
                            leading_coords[:, 0],
                        )

                        # Interpolate between the trailing edge coordinates to find
                        # the x-coordinate of
                        # the back left vertex.
                        back_left_vertices[spanwise_loc, 0] = np.interp(
                            spanwise_loc * spanwise_step,
                            trailing_coords[:, 1],
                            trailing_coords[:, 0],
                        )

                        # Interpolate between the leading edge coordinates to find
                        # the x-coordinate of
                        # the front right vertex.
                        front_right_vertices[spanwise_loc, 0] = np.interp(
                            (spanwise_loc + 1) * spanwise_step,
                            leading_coords[:, 1],
                            leading_coords[:, 0],
                        )

                        # Interpolate between the trailing edge coordinates to find
                        # the x-coordinate of
                        # the back right vertex.
                        back_right_vertices[spanwise_loc, 0] = np.interp(
                            (spanwise_loc + 1) * spanwise_step,
                            trailing_coords[:, 1],
                            trailing_coords[:, 0],
                        )

                    # Define an empty list to hold the wing cross sections.
                    this_airplane_wing_cross_sections = []

                    # Iterate through the wing cross section vertex arrays to create
                    # the wing cross
                    # section objects.
                    for i in range(this_num_spanwise_sections):

                        # Get the left wing cross section’s vertices at this position.
                        this_front_left_vertex = front_left_vertices[i, :]
                        this_back_left_vertex = back_left_vertices[i, :]

                        # Get this wing cross section’s leading and trailing edge
                        # x-coordinates.
                        this_x_le = this_front_left_vertex[0]
                        this_x_te = this_back_left_vertex[0]

                        # Get this wing cross section’s leading edge y-coordinate.
                        this_y_le = this_front_left_vertex[1]

                        # Calculate this wing cross section’s chord.
                        this_chord = this_x_te - this_x_le

                        # Define this wing cross section object.
                        this_wing_cross_section = ps.geometry.WingCrossSection(
                            x_le=this_x_le,
                            y_le=this_y_le,
                            chord=this_chord,
                            airfoil=ps.geometry.Airfoil(
                                name="naca0000",
                            ),
                            num_spanwise_panels=1,
                        )

                        # Append this wing cross section to the list of wing cross
                        # sections.
                        this_airplane_wing_cross_sections.append(
                            this_wing_cross_section
                        )

                        # Check if this the last section.
                        if i == this_num_spanwise_sections - 1:
                            # If so, get the right wing cross section vertices at
                            # this position.
                            this_front_right_vertex = front_right_vertices[i, :]
                            this_back_right_vertex = back_right_vertices[i, :]

                            # Get this wing cross section’s leading and trailing edge
                            # x-coordinates.
                            this_x_le = this_front_right_vertex[0]
                            this_x_te = this_back_right_vertex[0]

                            # Get this wing cross section’s leading edge y-coordinate.
                            this_y_le = this_front_right_vertex[1]

                            # Calculate this wing cross section’s chord.
                            this_chord = this_x_te - this_x_le

                            # Define this wing cross section object.
                            this_wing_cross_section = ps.geometry.WingCrossSection(
                                x_le=this_x_le,
                                y_le=this_y_le,
                                chord=this_chord,
                                airfoil=ps.geometry.Airfoil(
                                    name="naca0000",
                                ),
                                num_spanwise_panels=1,
                            )

                            # Append this wing cross section to the list of wing
                            # cross sections.
                            this_airplane_wing_cross_sections.append(
                                this_wing_cross_section
                            )

                    # Define the validation airplane object.
                    this_airplane = ps.geometry.Airplane(
                        name="Validation Airplane",
                        wings=[
                            ps.geometry.Wing(
                                symmetric=True,
                                wing_cross_sections=this_airplane_wing_cross_sections,
                                chordwise_spacing="uniform",
                                num_chordwise_panels=num_chordwise_panels,
                            ),
                        ],
                    )

                    # Delete the extraneous pointer.
                    del this_airplane_wing_cross_sections

                    # Initialize an empty list to hold each wing cross section
                    # movement object.
                    these_wing_cross_section_movements = []

                    # Define the first wing cross section movement, which is stationary.
                    first_wing_cross_section_movement = (
                        ps.movement.WingCrossSectionMovement(
                            base_wing_cross_section=this_airplane.wings[
                                0
                            ].wing_cross_sections[0],
                        )
                    )

                    # Append the first wing cross section movement object to the list.
                    these_wing_cross_section_movements.append(
                        first_wing_cross_section_movement
                    )

                    # Delete the extraneous pointer.
                    del first_wing_cross_section_movement

                    # Iterate through each of the wing cross sections.
                    for j in range(1, this_num_spanwise_sections + 1):
                        # Define the wing cross section movement for this wing cross
                        # section. The amplitude and period are both set to one
                        # because the true amplitude and period are already accounted
                        # for in the custom sweep function. Append this wing cross
                        # section movement to the list of wing cross section movements.
                        these_wing_cross_section_movements.append(
                            ps.movement.WingCrossSectionMovement(
                                base_wing_cross_section=this_airplane.wings[
                                    0
                                ].wing_cross_sections[j],
                                sweeping_amplitude=1,
                                sweeping_period=1 / validation_flapping_frequency,
                                sweeping_spacing="custom",
                                custom_sweep_function=normalized_validation_geometry_sweep_function,
                            )
                        )

                    # Define the wing movement object that contains the wing cross
                    # section movements.
                    this_main_wing_movement = ps.movement.WingMovement(
                        base_wing=this_airplane.wings[0],
                        wing_cross_sections_movements=these_wing_cross_section_movements,
                    )

                    # Delete the extraneous pointer.
                    del these_wing_cross_section_movements

                    # Define the airplane movement that contains the wing movement.
                    this_airplane_movement = ps.movement.AirplaneMovement(
                        base_airplane=this_airplane,
                        wing_movements=[
                            this_main_wing_movement,
                        ],
                    )

                    # Delete the extraneous pointers.
                    del this_airplane
                    del this_main_wing_movement

                    # Define an operating point corresponding to the conditions of
                    # the validation study.
                    this_operating_point = ps.operating_point.OperatingPoint(
                        alpha=validation_alpha,
                        velocity=validation_velocity,
                    )

                    # Define an operating point movement that contains the operating
                    # point.
                    this_operating_point_movement = ps.movement.OperatingPointMovement(
                        base_operating_point=this_operating_point,
                    )

                    # Delete the extraneous pointer.
                    del this_operating_point

                    these_airplane_movements = [this_airplane_movement]

                    del this_airplane_movement

                    # Define the overall movement.
                    this_movement = ps.movement.Movement(
                        airplane_movements=these_airplane_movements,
                        operating_point_movement=this_operating_point_movement,
                        num_cycles=wake_length,
                    )

                    # Delete the extraneous pointers.
                    del this_operating_point_movement

                    # Define the validation problem.
                    this_problem = ps.problems.UnsteadyProblem(
                        movement=this_movement,
                    )

                    # Delete the extraneous pointer.
                    del this_movement

                    # Create and run this iteration's solver and time how long it
                    # takes to execute.
                    this_solver = ps.unsteady_ring_vortex_lattice_method.UnsteadyRingVortexLatticeMethodSolver(
                        unsteady_problem=this_problem
                    )
                    iter_start = time.time()
                    this_solver.run(
                        logging_level="Warning",
                        prescribed_wake=wake,
                        calculate_streamlines=False,
                    )
                    iter_stop = time.time()
                    this_iter_time = iter_stop - iter_start

                    # Create and fill arrays with each of this iteration's airplane's
                    # resultant force and moment coefficients.
                    these_coefficients = np.zeros((len(these_airplane_movements), 6))
                    for airplane_id, airplane in enumerate(these_airplane_movements):
                        these_force_coefficients = (
                            this_problem.final_rms_force_coefficients[airplane_id]
                        )
                        these_moment_coefficients = (
                            this_problem.final_rms_moment_coefficients[airplane_id]
                        )

                        these_coefficients[airplane_id] = np.hstack(
                            [these_force_coefficients, these_moment_coefficients]
                        )

                    # Populate the arrays that store information of all the
                    # iterations with the data from this iteration.
                    coefficients[these_ids] = these_coefficients
                    iter_times[these_ids] = this_iter_time

                    validation_logger.info(
                        "\t\t\t\tIteration Time: "
                        + str(round(this_iter_time, 3))
                        + " s"
                    )

                    max_wake_apc = get_max_apc(
                        0, these_ids, coefficients, coefficient_mask
                    )
                    max_length_apc = get_max_apc(
                        1, these_ids, coefficients, coefficient_mask
                    )
                    max_ar_apc = get_max_apc(
                        2, these_ids, coefficients, coefficient_mask
                    )
                    max_chord_apc = get_max_apc(
                        3, these_ids, coefficients, coefficient_mask
                    )

                    if not np.isnan(max_wake_apc):
                        validation_logger.info(
                            "\t\t\t\tMaximum coefficient change from wake type: "
                            + str(round(max_wake_apc, 2))
                            + "%"
                        )

                    if not np.isnan(max_length_apc):
                        validation_logger.info(
                            "\t\t\t\tMaximum coefficient change from wake length: "
                            + str(round(max_length_apc, 2))
                            + "%"
                        )

                    if not np.isnan(max_ar_apc):
                        validation_logger.info(
                            "\t\t\t\tMaximum coefficient change from panel aspect "
                            "ratio: " + str(round(max_ar_apc, 2)) + "%"
                        )

                    if not np.isnan(max_chord_apc):
                        validation_logger.info(
                            "\t\t\t\tMaximum coefficient change from chordwise "
                            "panels: " + str(round(max_chord_apc, 2)) + "%"
                        )

                    # Consider the panel aspect ratio value to be saturated if it is
                    # equal to 1. This is because a panel aspect ratio of 1 is
                    # considered the maximum degree of fineness. Consider the wake
                    # state to be saturated if it False (which corresponds to a free
                    # wake), as this is considered to be the most accurate wake state.
                    wake_saturated = not wake
                    ar_saturated = panel_aspect_ratio == 1

                    # Check if the user only specified one value for any of the four
                    # convergence parameters.
                    single_wake = len(wake_list) == 1
                    single_length = len(wake_lengths_list) == 1
                    single_ar = len(panel_aspect_ratios_list) == 1
                    single_chord = len(num_chordwise_panels_list) == 1

                    # Check if the iteration calculated that it is converged with
                    # respect to any of the four convergence parameters.
                    wake_converged = max_wake_apc < convergence_criteria
                    length_converged = max_length_apc < convergence_criteria
                    ar_converged = max_ar_apc < convergence_criteria
                    chord_converged = max_chord_apc < convergence_criteria

                    # Consider each convergence parameter to have passed it is
                    # converged, single, or saturated.
                    wake_passed = wake_converged or single_wake or wake_saturated
                    length_passed = length_converged or single_length
                    ar_passed = ar_converged or single_ar or ar_saturated
                    chord_passed = chord_converged or single_chord

                    # If all four convergence parameters have passed, then the solver
                    # has found a converged or semi-converged value and will return
                    # the converged parameters.
                    if wake_passed and length_passed and ar_passed and chord_passed:
                        if single_wake:
                            converged_wake_id = wake_id
                        else:
                            # We've tested both prescribed and free wakes.
                            if wake_converged:
                                # There isn't a big difference between the prescribed
                                # wake and free wake, so the prescribed wake is
                                # converged.
                                converged_wake_id = wake_id - 1
                            else:
                                # There is a big different difference between the
                                # prescribed wake and free wake, so the free wake is
                                # converged.
                                converged_wake_id = wake_id

                        if single_length:
                            converged_length_id = length_id
                        else:
                            converged_length_id = length_id - 1

                        if single_ar:
                            converged_ar_id = ar_id
                        else:
                            # We've tested more than one panel aspect ratio.
                            if ar_converged:
                                # There is no big difference between this panel aspect
                                # ratio and the last (coarser) panel aspect ratio.
                                # Therefore, the last (coarser) panel aspect ratio is
                                # converged.
                                converged_ar_id = ar_id - 1
                            else:
                                # There is a big difference between this panel aspect
                                # ratio and the last (coarser) panel aspect ratio.
                                # However, the panel aspect ratio is one, so it's
                                # saturated. Therefore, this panel aspect ratio is
                                # converged.
                                converged_ar_id = ar_id

                        if single_chord:
                            converged_chord_id = chord_id
                        else:
                            converged_chord_id = chord_id - 1

                        converged_wake = wake_list[converged_wake_id]
                        converged_wake_length = wake_lengths_list[converged_length_id]
                        converged_chordwise_panels = num_chordwise_panels_list[
                            converged_chord_id
                        ]
                        converged_aspect_ratio = panel_aspect_ratios_list[
                            converged_ar_id
                        ]
                        converged_iter_time = iter_times[
                            converged_wake_id,
                            converged_length_id,
                            converged_ar_id,
                            converged_chord_id,
                        ]

                        if single_wake or single_length or single_ar or single_chord:
                            validation_logger.info(
                                "The analysis found a semi-converged mesh:"
                            )
                            if single_wake:
                                validation_logger.warning(
                                    "Wake type convergence not checked."
                                )
                            if single_length:
                                validation_logger.warning(
                                    "Wake length convergence not checked."
                                )
                            if single_ar:
                                validation_logger.warning(
                                    "Panel aspect ratio convergence not checked."
                                )
                            if single_chord:
                                validation_logger.warning(
                                    "Chordwise panels convergence not checked."
                                )
                        else:
                            validation_logger.info(
                                "The analysis found a converged mesh:"
                            )

                        if converged_wake:
                            validation_logger.info("\tWake type: prescribed")
                        else:
                            validation_logger.info("\tWake type: free")

                        validation_logger.info(
                            "\tCycles: " + str(converged_wake_length)
                        )
                        validation_logger.info(
                            "\tPanel aspect ratio: " + str(converged_aspect_ratio)
                        )
                        validation_logger.info(
                            "\tChordwise panels: " + str(converged_chordwise_panels)
                        )
                        validation_logger.info(
                            "\tIteration time: "
                            + str(round(converged_iter_time, 3))
                            + " s"
                        )

                        return [
                            converged_wake,
                            converged_wake_length,
                            converged_aspect_ratio,
                            converged_chordwise_panels,
                            wake_list,
                            wake_lengths_list,
                            panel_aspect_ratios_list,
                            num_chordwise_panels_list,
                            iter_times,
                            coefficients,
                        ]

    # If all iterations have been checked and none of them resulted in all
    # convergence parameters passing, then indicate that no converged solution was
    # found and return values of None for the converged parameters.
    validation_logger.info("The analysis did not find a converged mesh.")
    return [None, None, None, None]


def get_max_apc(param_index, these_ids, coefficients, coefficient_mask):
    max_pc = np.nan

    param_id = these_ids[param_index]

    if param_id > 0:

        last_ids = list(these_ids)
        last_ids[param_index] = param_id - 1
        last_ids = tuple(last_ids)

        these_coefficients = coefficients[these_ids]
        last_coefficients = coefficients[last_ids]

        these_coefficients = these_coefficients[:, coefficient_mask]
        last_coefficients = last_coefficients[:, coefficient_mask]

        apcs = absolute_percent_change(these_coefficients, last_coefficients)

        max_pc = np.max(apcs)

    return max_pc


# ToDo: Document this function.
def absolute_percent_change(new, old):
    return 100 * np.abs((new - old) / old)


# 5% Convergence:
#   wake:               prescribed
#   cycles:             2
#   panel aspect ratio: 4 (3 spanwise sections)
#   chordwise panels:   3
# 1% Convergence:
#   wake:               free
#   cycles:             2
#   panel aspect ratio: 4 (4 spanwise sections)
#   chordwise panels:   4
converged_parameters = analyze_convergence(
    coefficient_mask=[False, False, True, False, False, False],
    prescribed_wake=True,
    free_wake=True,
    num_cycles_bounds=(2, 4),
    panel_aspect_ratio_bounds=(4, 2),
    num_chordwise_panels_bounds=(3, 6),
    convergence_criteria=1.0,
)
