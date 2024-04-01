import numpy as np


def mjuu_frame2quat(x, y, z):
    mat = np.array([x, y, z])

    quat = np.zeros(4)

    # q0 largest
    if np.trace(mat) > 0:
        quat[0] = 0.5 * np.sqrt(1 + np.trace(mat))
        quat[1] = 0.25 * (mat[1][2] - mat[2][1]) / quat[0]
        quat[2] = 0.25 * (mat[2][0] - mat[0][2]) / quat[0]
        quat[3] = 0.25 * (mat[0][1] - mat[1][0]) / quat[0]

    # q1 largest
    elif mat[0][0] > mat[1][1] and mat[0][0] > mat[2][2]:
        quat[1] = 0.5 * np.sqrt(1 + mat[0][0] - mat[1][1] - mat[2][2])
        quat[0] = 0.25 * (mat[1][2] - mat[2][1]) / quat[1]
        quat[2] = 0.25 * (mat[1][0] + mat[0][1]) / quat[1]
        quat[3] = 0.25 * (mat[2][0] + mat[0][2]) / quat[1]

    # q2 largest
    elif mat[1][1] > mat[2][2]:
        quat[2] = 0.5 * np.sqrt(1 - mat[0][0] + mat[1][1] - mat[2][2])
        quat[0] = 0.25 * (mat[2][0] - mat[0][2]) / quat[2]
        quat[1] = 0.25 * (mat[1][0] + mat[0][1]) / quat[2]
        quat[3] = 0.25 * (mat[2][1] + mat[1][2]) / quat[2]

    # q3 largest
    else:
        quat[3] = 0.5 * np.sqrt(1 - mat[0][0] - mat[1][1] + mat[2][2])
        quat[0] = 0.25 * (mat[0][1] - mat[1][0]) / quat[3]
        quat[1] = 0.25 * (mat[2][0] + mat[0][2]) / quat[3]
        quat[2] = 0.25 * (mat[2][1] + mat[1][2]) / quat[3]

    # normalize the quaternion
    quat /= np.linalg.norm(quat)

    return quat


def xyaxes2quat(xyaxes):
    x_axis = np.array(xyaxes[:3])
    y_axis = np.array(xyaxes[3:])
    z_axis = np.cross(x_axis, y_axis)
    return mjuu_frame2quat(x_axis, y_axis, z_axis)


# print(xyaxes2quat([-0.868, -0.496, -0.000, 0.192, -0.336, 0.922]))
# print(xyaxes2quat([0.612, -0.791, 0.000, 0.374, 0.290, 0.881]))
print(xyaxes2quat([-0.904, 0.427, 0.000, -0.266, -0.563, 0.783]))  # -0.904 0.427 0.000 -0.266 -0.563 0.783

# void mjuu_frame2quat(double* quat, const double* x, const double* y, const double* z) {
#   const double* mat[3] = {x, y, z};  // mat[c][r] indexing

#   // q0 largest
#   if (mat[0][0]+mat[1][1]+mat[2][2]>0) {
#     quat[0] = 0.5 * sqrt(1 + mat[0][0] + mat[1][1] + mat[2][2]);
#     quat[1] = 0.25 * (mat[1][2] - mat[2][1]) / quat[0];
#     quat[2] = 0.25 * (mat[2][0] - mat[0][2]) / quat[0];
#     quat[3] = 0.25 * (mat[0][1] - mat[1][0]) / quat[0];
#   }

#   // q1 largest
#   else if (mat[0][0]>mat[1][1] && mat[0][0]>mat[2][2]) {
#     quat[1] = 0.5 * sqrt(1 + mat[0][0] - mat[1][1] - mat[2][2]);
#     quat[0] = 0.25 * (mat[1][2] - mat[2][1]) / quat[1];
#     quat[2] = 0.25 * (mat[1][0] + mat[0][1]) / quat[1];
#     quat[3] = 0.25 * (mat[2][0] + mat[0][2]) / quat[1];
#   }

#   // q2 largest
#   else if (mat[1][1]>mat[2][2]) {
#     quat[2] = 0.5 * sqrt(1 - mat[0][0] + mat[1][1] - mat[2][2]);
#     quat[0] = 0.25 * (mat[2][0] - mat[0][2]) / quat[2];
#     quat[1] = 0.25 * (mat[1][0] + mat[0][1]) / quat[2];
#     quat[3] = 0.25 * (mat[2][1] + mat[1][2]) / quat[2];
#   }

#   // q3 largest
#   else {
#     quat[3] = 0.5 * sqrt(1 - mat[0][0] - mat[1][1] + mat[2][2]);
#     quat[0] = 0.25 * (mat[0][1] - mat[1][0]) / quat[3];
#     quat[1] = 0.25 * (mat[2][0] + mat[0][2]) / quat[3];
#     quat[2] = 0.25 * (mat[2][1] + mat[1][2]) / quat[3];
#   }

#   mjuu_normvec(quat, 4);
# }
