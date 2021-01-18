from .base import *

def collect_ida_curves(results_filename, gm_metadata, ida_folder):
    gm_ids = gm_metadata.index
    n_gms = len(gm_ids)

    ida_segments = np.zeros((n_gms, 300, 2))
    ida_segments[:, :, 0] = 0.2

    for i, gm_id in zip(range(n_gms), gm_ids):
        key = ida_folder + gm_id + '/ida_curve'
        ida_curve = pd.read_hdf(results_filename, key)

        ida_curve = ida_curve.loc[:, ['Sa_avg', 'Story Drift Ratio (max)']].to_numpy()
        collapse = ida_curve[-1, 0]
        ida_segments[i, :, 1] = collapse

        ida_segments[i, 0:len(ida_curve), 1] = ida_curve[:, 0]
        ida_segments[i, 0:len(ida_curve), 0] = ida_curve[:, 1]

    return ida_segments

def collect_peak_and_residual_drift_curves(results_filename, gm_metadata):
    gm_ids = gm_metadata.index
    n_gms = len(gm_ids)
    collapse_intensities = gm_metadata['Intact Collapse Sa_avg']

    key = '/mainshock_damage_results/peak_story_drift_max'
    peak_drift = pd.read_hdf(results_filename, key=key)
    peak_drift = peak_drift[np.sort(peak_drift.columns)].copy()

    key = '/mainshock_damage_results/residual_drift_max'
    residual_drift = pd.read_hdf(results_filename, key=key)
    residual_drift = residual_drift[np.sort(peak_drift.columns)].copy()

    stripes = peak_drift.columns
    n_stripes = len(stripes)

    for edp_type in ['peak', 'residual']:
        if edp_type == 'peak':
            edp = peak_drift
        elif edp_type == 'residual':
            edp = residual_drift

        segments = np.zeros((n_gms, n_stripes + 1, 2))
        segments[:, 0, :] = 0
        segments[:, 1:, 0] = edp.to_numpy()

        for stripe, i in zip(stripes, range(n_stripes)):
            segments[:, i + 1, 1] = collapse_intensities * stripe

        if edp_type == 'peak':
            peak_segments = segments
        elif edp_type == 'residual':
            residual_segments = segments

    return peak_segments, residual_segments


def plot_building_at_t(t, edp, columns, beams, plot_scale, color_name, line_type, ax):
#     ax.cla()

    [_, n_pts] = edp.shape
    edp = np.insert(edp, 0, np.zeros((1, n_pts)), axis=0)

    [n_columns, _, _] = columns.shape
    [n_beams, _, _] = beams.shape
    n_stories = int(n_columns - n_beams)
    n_bays = int(n_beams / n_stories)

    columns_t = columns.copy()
    beams_t = beams.copy()

    i_col = 0
    i_beam = 0
    for i_story in range(n_stories):
        for i_end in range(2):
            columns_t[i_col:i_col + n_bays + 2, i_end, 0] = columns[i_col:i_col + n_bays + 2, i_end, 0] + plot_scale * \
                                                                                                          edp[
                                                                                                              i_story + i_end, t]
        i_col = i_col + n_bays + 1

        beams_t[i_beam:i_beam + n_bays + 1, :, 0] = beams[i_beam:i_beam + n_bays + 1, :, 0] + plot_scale * edp[
            i_story + 1, t]
        i_beam = i_beam + n_bays

    column_collection = LineCollection(columns_t, color=color_name, linestyle=line_type)
    _ = ax.add_collection(column_collection)

    beam_collection = LineCollection(beams_t, color=color_name, linestyle=line_type)
    _ = ax.add_collection(beam_collection)

    _ = ax.axis('scaled')

    building_height = np.max(columns[:, :, 1])
    building_width = np.max(columns[:, :, 0])
    y_gap = 20
    x_gap = 500
    _ = ax.set_xlim(-x_gap, building_width + x_gap)
    _ = ax.set_ylim(0, building_height + y_gap)
    _ = ax.axis('off')
    # _ = ax.text(building_width / 2, -y_gap, 'Displacement scale: ' + str(plot_scale) + 'x', ha='center', va='top',
    #             fontsize=18)


def plot_hinges(t, edp, joints_x, joints_y, plot_scale, peak_joint_pos, peak_joint_neg, hinge_yield_rotation_positive,
                hinge_cap_rotation_positive, hinge_yield_rotation_negative, hinge_cap_rotation_negative,  ax):
    ## plot hinges takes the rotation of every hinge of a frame around each beam-column joint of coordinates (joint_x by joint_y)
    # in positive and negative direction (peak_joint_pos, peak_joint_neg) and plots the hinges that are in either of
    # the following groups: (yield, cap/3]; (cap/3, cap*2/3], (cap*2/3, cap], or greater than cap
    # INPUTS
    #       t                             = time index to display deformed shape and hinges state
    #       edp                           = 2D displacement array [story, time]
    #       joints_x                      = 2D x-coord of each joint [story, column]
    #       joints_y                      = 2D y-coord of each joint [story, column]
    #       plot_scale                    = scalar to amplify displacement
    #       peak_joint_pos                = 4D array with rotation demand per hinge in each joint [floor, column, hinge_loc, time]
    #                                       hinge_loc is 0: bottom of the joint
    #                                                    1: right
    #                                                    2: top
    #                                                    3: left
    #       peak_joint_neg                = same as before but with maximum negative demand
    #       hinge_yield_rotation_positive = 4D array with yield rotation capacity per hinge in each joint [floor, column, hinge_loc, time]
    #       hinge_cap_rotation_positive
    #       hinge_yield_rotation_negative
    #       hinge_cap_rotation_negative
    #       ax                            = axis to add the hinge plots

    # Retrieve basic info for loops
    n_stories, n_bays = joints_x.shape
    n_stories = n_stories - 1
    n_bays = n_bays - 1

    # Assemble vector with hinges in each state
    disp_t = edp[:, t]  # displacement for deformed shape
    disp_t = np.insert(disp_t, 0, 0, axis=0)  # add the hinge at column base# add zero displacement at base of the column
    dhinge = 4  # plotting delta from joint

    joints_x_yield_pos = np.empty((0, 1))
    joints_y_yield_pos = np.empty((0, 1))
    joints_x_yield_neg = np.empty((0, 1))
    joints_y_yield_neg = np.empty((0, 1))
    joints_x_yield_both = np.empty((0, 1))
    joints_y_yield_both = np.empty((0, 1))

    joints_x_capt = np.empty((0, 1))
    joints_y_capt = np.empty((0, 1))

    joints_x_cap2t = np.empty((0, 1))
    joints_y_cap2t = np.empty((0, 1))

    joints_x_cap = np.empty((0, 1))
    joints_y_cap = np.empty((0, 1))

    for floor_i in range(n_stories + 1):
        disp_curr = disp_t[floor_i] * plot_scale

        for col_i in range(n_bays + 1):

            for hinge_loc in range(4):

                # Read rotation demand of current hinge
                peakPos = peak_joint_pos[floor_i, col_i, hinge_loc, 0]
                peakNeg = -peak_joint_neg[floor_i, col_i, hinge_loc, 0]

                # Read rotation capacity of current hinge
                yieldPos = hinge_yield_rotation_positive[floor_i, col_i, hinge_loc, 0]
                yieldNeg = -hinge_yield_rotation_negative[floor_i, col_i, hinge_loc, 0]
                capPos = hinge_cap_rotation_positive[floor_i, col_i, hinge_loc, 0]
                capNeg = -hinge_cap_rotation_negative[floor_i, col_i, hinge_loc, 0]

                # Plotting position of current hinge
                if hinge_loc == 0:  # Bottom hinge
                    curr_x = joints_x[floor_i, col_i] + disp_curr
                    curr_y = joints_y[floor_i, col_i] - dhinge * plot_scale
                elif hinge_loc == 1:  # Right hinge
                    curr_x = joints_x[floor_i, col_i] + disp_curr + dhinge * plot_scale
                    curr_y = joints_y[floor_i, col_i]
                elif hinge_loc == 2:  # Top hinge
                    curr_x = joints_x[floor_i, col_i] + disp_curr
                    curr_y = joints_y[floor_i, col_i] + dhinge * plot_scale
                else:  # Left hinge
                    curr_x = joints_x[floor_i, col_i] + disp_curr - dhinge * plot_scale
                    curr_y = joints_y[floor_i, col_i]

                if yieldPos != 0:
                    if (peakPos > yieldPos and peakPos <= capPos / 3):
                        # Between yield and capping/3
                        joints_x_yield_pos = np.append(joints_x_yield_pos, curr_x)
                        joints_y_yield_pos = np.append(joints_y_yield_pos, curr_y)

                    if (peakNeg > yieldNeg and peakNeg <= capNeg / 3):
                        # Between yield and capping/3
                        joints_x_yield_neg = np.append(joints_x_yield_neg, curr_x)
                        joints_y_yield_neg = np.append(joints_y_yield_neg, curr_y)

                    if (peakPos > yieldPos and peakPos <= capPos / 3) and (peakNeg > yieldNeg and peakNeg <= capNeg / 3):
                        # Between yield and capping/3
                        joints_x_yield_both = np.append(joints_x_yield_both, curr_x)
                        joints_y_yield_both = np.append(joints_y_yield_both, curr_y)

                    elif (peakPos > capPos / 3 and peakPos <= capPos * 2 / 3) or (
                            peakNeg > capNeg / 3 and peakNeg <= capNeg * 2 / 3):
                        # Between capping/3 and capping*2/3
                        joints_x_capt = np.append(joints_x_capt, curr_x)
                        joints_y_capt = np.append(joints_y_capt, curr_y)

                    elif (peakPos > capPos * 2 / 3 and peakPos <= capPos) or (
                            peakNeg > capNeg * 2 / 3 and peakNeg <= capNeg):
                        # Between capping*2/3 and capping
                        #                         print('yieldPos='+str(yieldPos)+';capPos='+str(capPos))
                        #                         print('floor='+str(floor_i)+';column='+str(col_i)+';hinge='+str(hinge_loc))
                        joints_x_cap2t = np.append(joints_x_cap2t, curr_x)
                        joints_y_cap2t = np.append(joints_y_cap2t, curr_y)
                    elif (peakPos > capPos) or (peakNeg > capNeg):
                        # beyond capping point
                        #                         print('yieldPos='+str(yieldPos)+';capPos='+str(capPos))
                        #                         print('floor='+str(floor_i)+';column='+str(col_i)+';hinge='+str(hinge_loc))
                        joints_x_cap = np.append(joints_x_cap, curr_x)
                        joints_y_cap = np.append(joints_y_cap, curr_y)

    # Plot hinges with appropiate colors
    _ = ax.plot(joints_x_yield_pos, joints_y_yield_pos, 'o', color='pink')
    _ = ax.plot(joints_x_yield_neg, joints_y_yield_neg, 'o', color='hotpink')
    _ = ax.plot(joints_x_yield_both, joints_y_yield_both, 'o', color='m')
    _ = ax.plot(joints_x_capt, joints_y_capt, 'o', color='c')
    _ = ax.plot(joints_x_cap2t, joints_y_cap2t, 'o', color='r')
    _ = ax.plot(joints_x_cap, joints_y_cap, 'o', color='b')



def plot_hinges_prob(t, edp, joints_x, joints_y, plot_scale, peak_joint_pos, peak_joint_neg,
                     hinge_yield_rotation_positive,
                     hinge_cap_rotation_positive, hinge_yield_rotation_negative, hinge_cap_rotation_negative, ax):
    ## plot hinges takes the rotation of every hinge of a frame around each beam-column joint of coordinates (joint_x by joint_y)
    # in positive and negative direction (peak_joint_pos, peak_joint_neg) and plots the hinges that are in either of
    # the following groups: (yield, cap/3]; (cap/3, cap*2/3], (cap*2/3, cap], or greater than cap
    # INPUTS
    #       t                             = time index to display deformed shape and hinges state
    #       edp                           = 2D displacement array [story, time]
    #       joints_x                      = 2D x-coord of each joint [story, column]
    #       joints_y                      = 2D y-coord of each joint [story, column]
    #       plot_scale                    = scalar to amplify displacement
    #       peak_joint_pos                = 4D array with rotation demand per hinge in each joint [floor, column, hinge_loc, time]
    #                                       hinge_loc is 0: bottom of the joint
    #                                                    1: right
    #                                                    2: top
    #                                                    3: left
    #       peak_joint_neg                = same as before but with maximum negative demand
    #       hinge_yield_rotation_positive = 4D array with yield rotation capacity per hinge in each joint [floor, column, hinge_loc, time]
    #       hinge_cap_rotation_positive
    #       hinge_yield_rotation_negative
    #       hinge_cap_rotation_negative
    #       ax                            = axis to add the hinge plots

    # Retrieve basic info for loops
    n_stories, n_bays = joints_x.shape
    n_stories = n_stories - 1
    n_bays = n_bays - 1

    # Assemble vector with hinges in each state
    disp_t = edp[:, t]  # displacement for deformed shape
    disp_t = np.insert(disp_t, 0, 0,
                       axis=0)  # add the hinge at column base# add zero displacement at base of the column
    dhinge = 4  # plotting delta from joint

    joints_x_yield = np.empty((0, 1))
    joints_y_yield = np.empty((0, 1))

    joints_x_capt = np.empty((0, 1))
    joints_y_capt = np.empty((0, 1))

    joints_x_cap2t = np.empty((0, 1))
    joints_y_cap2t = np.empty((0, 1))

    joints_x_cap = np.empty((0, 1))
    joints_y_cap = np.empty((0, 1))

    for floor_i in range(n_stories + 1):
        disp_curr = disp_t[floor_i] * plot_scale

        for col_i in range(n_bays + 1):

            for hinge_loc in range(4):

                # Read rotation demand of current hinge
                peakPos = peak_joint_pos[floor_i, col_i, hinge_loc, 0]
                peakNeg = -peak_joint_neg[floor_i, col_i, hinge_loc, 0]

                # Read rotation capacity of current hinge
                yieldPos = hinge_yield_rotation_positive[floor_i, col_i, hinge_loc, 0]
                yieldNeg = -hinge_yield_rotation_negative[floor_i, col_i, hinge_loc, 0]
                capPos = hinge_cap_rotation_positive[floor_i, col_i, hinge_loc, 0]
                capNeg = -hinge_cap_rotation_negative[floor_i, col_i, hinge_loc, 0]

                # Plotting position of current hinge
                if hinge_loc == 0:  # Bottom hinge
                    curr_x = joints_x[floor_i, col_i] + disp_curr
                    curr_y = joints_y[floor_i, col_i] - dhinge * plot_scale
                elif hinge_loc == 1:  # Right hinge
                    curr_x = joints_x[floor_i, col_i] + disp_curr + dhinge * plot_scale
                    curr_y = joints_y[floor_i, col_i]
                elif hinge_loc == 2:  # Top hinge
                    curr_x = joints_x[floor_i, col_i] + disp_curr
                    curr_y = joints_y[floor_i, col_i] + dhinge * plot_scale
                else:  # Left hinge
                    curr_x = joints_x[floor_i, col_i] + disp_curr - dhinge * plot_scale
                    curr_y = joints_y[floor_i, col_i]

                # Define color for plot
                if yieldPos != 0:
                    # Compute expected di of current hinge
                    if hinge_loc == 1 or hinge_loc == 3:
                        isBeam = True
                        pos_fragilities = get_FEMAP58_fragility(isBeam, capPos)
                        neg_fragilities = get_FEMAP58_fragility(isBeam, capNeg)
                        di, _ = compute_di_prob(peakPos, peakNeg, pos_fragilities, neg_fragilities)
                    else:
                        isBeam = False
                        pos_fragilities = get_FEMAP58_fragility(isBeam, capPos)
                        neg_fragilities = get_FEMAP58_fragility(isBeam, capNeg)
                        di, _ = compute_di_prob(peakPos, peakNeg, pos_fragilities, neg_fragilities)

                    # Select color of current hinge based on di
                    if (di > 0.5) and (di <= 1.5):
                        # Damage state 1
                        joints_x_yield = np.append(joints_x_yield, curr_x)
                        joints_y_yield = np.append(joints_y_yield, curr_y)

                    elif (di > 1.5) and (di <= 2.5):
                        # Between capping/3 and capping*2/3
                        joints_x_capt = np.append(joints_x_capt, curr_x)
                        joints_y_capt = np.append(joints_y_capt, curr_y)

                    elif (di > 2.5):
                        # Between capping*2/3 and capping
                        #                         print('yieldPos='+str(yieldPos)+';capPos='+str(capPos))
                        #                         print('floor='+str(floor_i)+';column='+str(col_i)+';hinge='+str(hinge_loc))
                        joints_x_cap2t = np.append(joints_x_cap2t, curr_x)
                        joints_y_cap2t = np.append(joints_y_cap2t, curr_y)

    # Plot hinges with appropiate colors
    _ = ax.plot(joints_x_yield, joints_y_yield, 'o', color='m')
    _ = ax.plot(joints_x_capt, joints_y_capt, 'o', color='c')
    _ = ax.plot(joints_x_cap2t, joints_y_cap2t, 'o', color='r')
	
	
def plot_mainshock_damage_visual(displacement, periods, spectrum, acc, dt, n_pts, time_series, story_idx, story_displacement,
                                 story_drift, column_geometry, beam_geometry, joints_x, joints_y,peak_joint_pos,
                                 peak_joint_neg, hinge_yield_rotation_positive, hinge_cap_rotation_positive,
                                 hinge_yield_rotation_negative, hinge_cap_rotation_negative):

    ax = list()

    fig = plt.figure(figsize=(15, 10))
    ax.append(plt.subplot2grid((3, 3), (0, 0), rowspan=1, colspan=2))
    ax.append(plt.subplot2grid((3, 3), (1, 0), rowspan=1, colspan=2))
    ax.append(plt.subplot2grid((3, 3), (2, 0), rowspan=1, colspan=2))
    ax.append(plt.subplot2grid((3, 3), (0, 2), rowspan=2, colspan=1))
    ax.append(plt.subplot2grid((3, 3), (2, 2), rowspan=1, colspan=1))

    for i in [1, 2]:
        ax[0].get_shared_x_axes().join(ax[0], ax[i])

    color = 'tab:blue'

    current_ax = ax[4]
    _ = current_ax.plot(periods, spectrum, color=color)
    _ = current_ax.set_xlim(0, 5)
    _ = current_ax.set_ylabel('Spectral Acceleration [g]')
    _ = current_ax.set_xlabel('Period')

    current_ax = ax[2]
    _ = current_ax.plot(np.arange(n_pts) * dt, acc, color=color)
    _ = current_ax.set_xlim(left=0)
    ylim = 1.1 * np.max(np.abs(acc))
    _ = current_ax.set_ylim((-ylim, ylim))
    _ = current_ax.set_ylabel('Ground Acceleration [g]')
    _ = current_ax.set_xlabel('Seconds')

    edp_list = [story_displacement, story_drift]
    edp_name = ['Displacement [in]', 'Story Drift Ratio [%]']
    level_name = [str(story_idx + 1) + 'th Floor', str(story_idx) + 'th Story']

    for edp, i in zip(edp_list, range(len(edp_list))):
        current_ax = ax[i]
        _ = current_ax.plot(time_series, edp, color=color)
        ylim = 1.1 * np.max(np.abs(edp))
        _ = current_ax.set_ylim((-ylim, ylim))
        ylabel = level_name[i] + '\n' + edp_name[i]
        _ = current_ax.set_ylabel(ylabel)

    for i in range(5):
        if i != 3:
            _ = ax[i].grid('on')

    current_ax = ax[0]
    _ = current_ax.set_xlim((0, time_series[-1]))

    current_ax = ax[3]
    t = len(time_series) - 1
    plot_scale = 10
    plot_building_at_t(t, displacement, column_geometry, beam_geometry, plot_scale, current_ax)
    plot_hinges(t, displacement, joints_x, joints_y, plot_scale, peak_joint_pos, peak_joint_neg,
                hinge_yield_rotation_positive, hinge_cap_rotation_positive, hinge_yield_rotation_negative,
                hinge_cap_rotation_negative, current_ax)
    # plot_hinges_prob(t, displacement, joints_x, joints_y, plot_scale, peak_joint_pos, peak_joint_neg,
    #             hinge_yield_rotation_positive, hinge_cap_rotation_positive, hinge_yield_rotation_negative,
    #             hinge_cap_rotation_negative, current_ax)
    plt.tight_layout()
    plt.show()
