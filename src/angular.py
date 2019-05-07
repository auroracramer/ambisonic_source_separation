import math
import numpy as np

def rotate_bformat(bfsig, yaw, pitch, roll, order='xyz'):
    R_mat = euler_to_rotation_matrix(-yaw*np.pi/180.0,
                                     -pitch*np.pi/180.0,
                                     roll*np.pi/180,
                                     order=order);

    # augment with zero order
    Rbf = np.zeros((4,4));
    Rbf[0,:] = 1.0;
    Rbf[1:,1:] = R_mat;

    # apply to B-format signals
    try:
        return np.dot(Rbf, bfsig)
    except:
        import pdb
        pdb.set_trace()


def rotate_coord(coord, yaw, pitch, roll, order='xyz'):
    R_mat = euler_to_rotation_matrix(-yaw*np.pi/180.0,
                                     -pitch*np.pi/180.0,
                                     roll*np.pi/180,
                                     order=order);

    return np.dot(R_mat, coord)


def euler_to_rotation_matrix(alpha, beta, gamma, order='xyz'):
    """
    %   alpha:  first angle of rotation
    %   beta:   second angle of rotation
    %   gamma:  third angle of rotation
    %
    %   order:  definition of the axes of rotation, e.g. for the y-convention
    %           this should be 'zyz', for the x-convnention 'zxz', and for
    %           the yaw-pitch-roll convention 'zyx'
    """
    def Rx(theta):
        return np.array([
            [1.0, 0.0, 0.0],
            [0.0, np.cos(theta), np.sin(theta)],
            [0.0, -np.sin(theta), np.cos(theta)]])

    def Ry(theta):
        return np.array([
            [np.cos(theta), 0.0, -np.sin(theta)],
            [0.0, 1.0, 0.0],
            [np.sin(theta), 0.0, np.cos(theta)]
        ])
    # [cos(theta) sin(theta) 0; -sin(theta) cos(theta) 0; 0 0 1]
    def Rz(theta):
        return np.array([
            [np.cos(theta), np.sin(theta), 0.0],
            [-np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0]])

    R = np.eye(3)
    for idx, dim in reversed(list(enumerate(order))):
        if dim == 'x':
            R_func = Rx
        elif dim == 'y':
            R_func = Ry
        elif dim == 'z':
            R_func = Rz

        if idx == 0:
            theta = alpha
        elif idx == 1:
            theta = beta
        elif idx == 2:
            theta = gamma

        R = np.dot(R, R_func(theta))

    return R


def cartesian_to_spherical(dist_coord, rads=True):
    # Convert cartesian coordiantes to spherical coordinates
    x, y, z = dist_coord
    r = np.linalg.norm(dist_coord)
    azi = math.atan2(y, x)
    # NOTE: This differs from typical elevation/inclination, which is usually measured with 0 pointing up
    elv = math.asin(z/float(r))

    if elv > np.pi/2:
        elv = (np.pi - elv) % (np.pi/2)
        azi = (np.pi + azi) % (2 * np.pi) - np.pi
    elif elv < - np.pi/2:
        elv = -((np.pi + elv) % (np.pi/2))
        azi = (np.pi + azi) % (2 * np.pi) - np.pi


    if not rads:
        azi = azi * 180.0 / np.pi
        elv = elv * 180.0 / np.pi

    return np.array([r, azi, elv])


def steer_vector(azis, elvs):
    #azi_res is resolution of azimuth angle from -180 to 180
    #ele_res is resolution of elevation angle from -90 to 90
    #theta=np.arange(-180,180,azi_res)
    #phi=np.arange(-90,90,ele_res)

    #azis and eles are pairs of chosen directions
    D = np.stack([
        np.ones((azis.shape[0],)),
        np.sqrt(3) * np.cos(azis) * np.cos(elvs),
        np.sqrt(3) * np.sin(azis) * np.cos(elvs),
        np.sqrt(3) * np.sin(elvs)])

    return D


def create_sc_to_pos_dict(speaker_coord=None):
    if speaker_coord is None:
        speaker_coord = np.array([-2, 6, 0])

    azi_list = []
    elv_list = []
    pos_list = []

    for grid_x in np.arange(13, step=1):
        for grid_y in np.arange(13, step=1):
            # Note that we swap x and y here to be consistent with standard
            # spherical coordinate convention
            x = grid_y
            y = grid_x
            mic_coord = np.array([x, y, 0])
            for d_yaw in np.arange(-180, 180, 60):
                for d_pitch in np.arange(-90, 90 + 10, 45):
                    for d_roll in np.arange(-90, 90 + 10, 45):


                        rel_src_coord = mic_coord - speaker_coord
                        rot_src_coord = rotate_coord(rel_src_coord, d_yaw, d_pitch, d_roll, order='xyz')
                        rot_src_coord_spherical = cartesian_to_spherical(rot_src_coord, rads=False)
                        rot_azi, rot_elv = rot_src_coord_spherical[1:]

                        azi_list.append(rot_azi)
                        elv_list.append(rot_elv)
                        pos_list.append((x, y, d_yaw, d_pitch, d_roll))

    sc_to_pos_dict = {}
    for azi, elv, pos in zip(azi_list, elv_list, pos_list):
        if (azi, elv) not in sc_to_pos_dict:
            sc_to_pos_dict[(azi, elv)] = []
        sc_to_pos_dict[(azi, elv)].append(pos)

    return sc_to_pos_dict


def get_sc_list(sc_to_pos_dict):
    return sorted(sorted(list(sc_to_pos_dict.keys()), key=lambda x: x[1]), key=lambda x: x[0])


def beamformer(pair,steer_mat):
    #pair is the index of desired pair of azimuth/elevation, D=(m,n), inv(D)=(n,m)
    u=np.zeros(steer_mat.shape[1])#(1,n)
    u[pair]=1
    beamformer=np.linalg.pinv(steer_mat)*u[:,None]#output=(n,m) should compute it only once
    return beamformer
