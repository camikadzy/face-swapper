import face_recognition
import numpy as np
import scipy.spatial as spatial
import cv2
import dlib
import time
import logging
from defenitions import ROOT_DIR

home = ROOT_DIR

PREDICTOR_PATH = f'{home}/ml_data/shape_predictor_68_face_landmarks.dat'
predictor = dlib.shape_predictor(PREDICTOR_PATH)


## 3D Transform
def bilinear_interpolate(img, coords):
    """ Interpolates over every image channel
    http://en.wikipedia.org/wiki/Bilinear_interpolation
    :param img: max 3 channel image
    :param coords: 2 x _m_ array. 1st row = xcoords, 2nd row = ycoords
    :returns: array of interpolated pixels with same shape as coords
    """
    int_coords = np.int32(coords)
    x0, y0 = int_coords
    dx, dy = coords - int_coords

    # 4 Neighour pixels
    q11 = img[y0, x0]
    q21 = img[y0, x0 + 1]
    q12 = img[y0 + 1, x0]
    q22 = img[y0 + 1, x0 + 1]

    btm = q21.T * dx + q11.T * (1 - dx)
    top = q22.T * dx + q12.T * (1 - dx)
    inter_pixel = top * dy + btm * (1 - dy)

    return inter_pixel.T

def grid_coordinates(points):
    """ x,y grid coordinates within the ROI of supplied points
    :param points: points to generate grid coordinates
    :returns: array of (x, y) coordinates
    """
    xmin = np.min(points[:, 0])
    xmax = np.max(points[:, 0]) + 1
    ymin = np.min(points[:, 1])
    ymax = np.max(points[:, 1]) + 1

    return np.asarray([(x, y) for y in range(ymin, ymax)
                       for x in range(xmin, xmax)], np.uint32)


def process_warp(src_img, result_img, tri_affines, dst_points, delaunay):
    """
    Warp each triangle from the src_image only within the
    ROI of the destination image (points in dst_points).
    """
    roi_coords = grid_coordinates(dst_points)
    # indices to vertices. -1 if pixel is not in any triangle
    roi_tri_indices = delaunay.find_simplex(roi_coords)

    for simplex_index in range(len(delaunay.simplices)):
        coords = roi_coords[roi_tri_indices == simplex_index]
        num_coords = len(coords)
        out_coords = np.dot(tri_affines[simplex_index],
                            np.vstack((coords.T, np.ones(num_coords))))
        x, y = coords.T
        result_img[y, x] = bilinear_interpolate(src_img, out_coords)

    return None


def triangular_affine_matrices(vertices, src_points, dst_points):
    """
    Calculate the affine transformation matrix for each
    triangle (x,y) vertex from dst_points to src_points
    :param vertices: array of triplet indices to corners of triangle
    :param src_points: array of [x, y] points to landmarks for source image
    :param dst_points: array of [x, y] points to landmarks for destination image
    :returns: 2 x 3 affine matrix transformation for a triangle
    """
    ones = [1, 1, 1]
    for tri_indices in vertices:
        src_tri = np.vstack((src_points[tri_indices, :].T, ones))
        dst_tri = np.vstack((dst_points[tri_indices, :].T, ones))
        mat = np.dot(src_tri, np.linalg.inv(dst_tri))[:2, :]
        yield mat


def warp_image_3d(src_img, src_points, dst_points, dst_shape, dtype=np.uint8):
    rows, cols = dst_shape[:2]
    result_img = np.zeros((rows, cols, 3), dtype=dtype)

    delaunay = spatial.Delaunay(dst_points)
    tri_affines = np.asarray(list(triangular_affine_matrices(
        delaunay.simplices, src_points, dst_points)))

    process_warp(src_img, result_img, tri_affines, dst_points, delaunay)

    return result_img


## 2D Transform
def transformation_from_points(points1, points2):
    points1 = points1.astype(np.float64)
    points2 = points2.astype(np.float64)

    c1 = np.mean(points1, axis=0)
    c2 = np.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2

    s1 = np.std(points1)
    s2 = np.std(points2)
    points1 /= s1
    points2 /= s2

    U, S, Vt = np.linalg.svd(np.dot(points1.T, points2))
    R = (np.dot(U, Vt)).T

    return np.vstack([np.hstack([s2 / s1 * R,
                                (c2.T - np.dot(s2 / s1 * R, c1.T))[:, np.newaxis]]),
                      np.array([[0., 0., 1.]])])


def warp_image_2d(im, M, dshape):
    output_im = np.zeros(dshape, dtype=im.dtype)
    cv2.warpAffine(im,
                   M[:2],
                   (dshape[1], dshape[0]),
                   dst=output_im,
                   borderMode=cv2.BORDER_TRANSPARENT,
                   flags=cv2.WARP_INVERSE_MAP)

    return output_im


## Generate Mask
def mask_from_points(size, points,erode_flag=1):
    radius = 10  # kernel size
    kernel = np.ones((radius, radius), np.uint8)

    mask = np.zeros(size, np.uint8)
    cv2.fillConvexPoly(mask, cv2.convexHull(points), 255)
    if erode_flag:
        mask = cv2.erode(mask, kernel,iterations=1)

    return mask


## Color Correction
def correct_colours(im1, im2, landmarks1):
    COLOUR_CORRECT_BLUR_FRAC = 0.75
    LEFT_EYE_POINTS = list(range(42, 48))
    RIGHT_EYE_POINTS = list(range(36, 42))

    blur_amount = COLOUR_CORRECT_BLUR_FRAC * np.linalg.norm(
                              np.mean(landmarks1[LEFT_EYE_POINTS], axis=0) -
                              np.mean(landmarks1[RIGHT_EYE_POINTS], axis=0))
    blur_amount = int(blur_amount)
    if blur_amount % 2 == 0:
        blur_amount += 1
    im1_blur = cv2.GaussianBlur(im1, (blur_amount, blur_amount), 0)
    im2_blur = cv2.GaussianBlur(im2, (blur_amount, blur_amount), 0)

    # Avoid divide-by-zero errors.
    im2_blur = im2_blur.astype(int)
    im2_blur += 128*(im2_blur <= 1)

    result = im2.astype(np.float64) * im1_blur.astype(np.float64) / im2_blur.astype(np.float64)
    result = np.clip(result, 0, 255).astype(np.uint8)

    return result


## Copy-and-paste
def apply_mask(img, mask):
    """ Apply mask to supplied image
    :param img: max 3 channel image
    :param mask: [0-255] values in mask
    :returns: new image with mask applied
    """
    masked_img=cv2.bitwise_and(img,img,mask=mask)

    return masked_img


## Alpha blending
def alpha_feathering(src_img, dest_img, img_mask, blur_radius=15):
    mask = cv2.blur(img_mask, (blur_radius, blur_radius))
    mask = mask / 255.0

    result_img = np.empty(src_img.shape, np.uint8)
    for i in range(3):
        result_img[..., i] = src_img[..., i] * mask + dest_img[..., i] * (1-mask)

    return result_img


def check_points(img,points):
    # Todo: I just consider one situation.
    if points[8,1]>img.shape[0]:
        logging.error("Jaw part out of image")
    else:
        return True
    return False


def face_swap(src_face, dst_face, src_points, dst_points, dst_shape, dst_img, args, end=48):
    h, w = dst_face.shape[:2]

    ## 3d warp
    warped_src_face = warp_image_3d(src_face, src_points[:end], dst_points[:end], (h, w))
    ## Mask for blending
    mask = mask_from_points((h, w), dst_points)
    mask_src = np.mean(warped_src_face, axis=2) > 0
    mask = np.asarray(mask * mask_src, dtype=np.uint8)
    ## Correct color
    if args == 'correct_color':
        warped_src_face = apply_mask(warped_src_face, mask)
        dst_face_masked = apply_mask(dst_face, mask)
        warped_src_face = correct_colours(dst_face_masked, warped_src_face, dst_points)
    ## 2d warp
    if args=='warp_2d':
        unwarped_src_face = warp_image_3d(warped_src_face, dst_points[:end], src_points[:end], src_face.shape[:2])
        warped_src_face = warp_image_2d(unwarped_src_face, transformation_from_points(dst_points, src_points),
                                        (h, w, 3))

        mask = mask_from_points((h, w), dst_points)
        mask_src = np.mean(warped_src_face, axis=2) > 0
        mask = np.asarray(mask * mask_src, dtype=np.uint8)

    ## Shrink the mask
    kernel = np.ones((10, 10), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    ##Poisson Blending
    r = cv2.boundingRect(mask)
    center = ((r[0] + int(r[2] / 2), r[1] + int(r[3] / 2)))
    output = cv2.seamlessClone(warped_src_face, dst_face, mask, center, cv2.NORMAL_CLONE)

    x, y, w, h = dst_shape
    dst_img_cp = dst_img.copy()
    dst_img_cp[y:y + h, x:x + w] = output

## Face detection
def face_detection(img, upsample_times=1):
    # Ask the detector to find the bounding boxes of each face. The 1 in the
    # second argument indicates that we should upsample the image 1 time. This
    # will make everything bigger and allow us to detect more faces.
    detector = dlib.get_frontal_face_detector()
    faces = detector(img, upsample_times)

    return faces


## Face and points detection
def face_points_detection(img, bbox: dlib.rectangle):
    # Get the landmarks/parts for the face in box d.
    shape = predictor(img, bbox)

    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    coords = np.asarray(list([p.x, p.y] for p in shape.parts()), dtype=np.int)

    # return the array of (x, y)-coordinates
    return coords


def select_face(im, cords, r=10):
    #
    face = dlib.rectangle(*cords)
    points = np.asarray(face_points_detection(im, face))
    im_w, im_h = im.shape[:2]
    left, top = np.min(points, 0)
    right, bottom = np.max(points, 0)
    x, y = max(0, left - r), max(0, top - r)
    w, h = min(right + r, im_h) - x, min(bottom + r, im_w) - y
    p = points - np.asarray([[x, y]])
    s = ((x, y, w, h))
    f = (im[y:y + h, x:x + w])
    return p, s, f


def get_smeared_faces(chat_id, pic_name):
    pic_dl = face_recognition.load_image_file(pic_name)
    pic = cv2.imread(pic_name)
    coords = []
    face_locations = face_recognition.face_locations(pic_dl)
    for ind, face in enumerate(face_locations):
        pat = pic[face[0]:face[2], face[3]:face[1]]
        cv2.imwrite(f"{home}/faces/{chat_id}_{str(ind)}.jpg", pat)
        coords.append((face[3], face[0], face[1], face[2]))

    try:
        return ind + 1, coords, pic, pic_dl
    except Exception:
        return -1, coords, pic, pic_dl


def swap(pic, pic_dl, target_coords, imagecv, imagecv_2, frames, patterns):
    src_points, src_shape, src_face = select_face(pic, target_coords)
    pic_dl = pic_dl[target_coords[1]:target_coords[3], target_coords[0]:target_coords[2]]

    if len(face_recognition.face_encodings(pic_dl)) > 0:
        dl_encoding = face_recognition.face_encodings(pic_dl)[0]

        for ind, pat in enumerate(patterns):
            if face_recognition.compare_faces(pat, dl_encoding)[0]:
                dst_points, dst_shape, dst_face = select_face(imagecv, frames[ind])
                imagecv = face_swap(src_face, dst_face, src_points, dst_points, dst_shape, imagecv, 'correct_color')
                imagecv_2 = face_swap(src_face, dst_face, src_points, dst_points, dst_shape, imagecv_2, 'warp_2d')

                # cv2.imshow(imagecv)
                break
        return imagecv,imagecv_2
    else:
        return np.array([]),np.array([])


def final_swap_and_clear(target_photo: str, users_with_photos: list) -> str:
    image = face_recognition.load_image_file(target_photo)
    imagecv = cv2.imread(target_photo)

    face_locations = face_recognition.face_locations(image)
    patterns = []
    frames = []
    for face in face_locations:
        pat = image[face[0]:face[2], face[3]:face[1]]
        frames.append((face[3], face[0], face[1], face[2]))
        face_rec = face_recognition.face_encodings(pat)
        if face_rec:
            patterns.append([face_rec[0]])

    niggas_gang = []
    for n, pic_name, coords, pic, pic_dl in users_with_photos:
        target_coords = coords[n]
        niggas_gang.append((pic, pic_dl, target_coords))

    imagecv_2 = np.copy(imagecv)

    for nigga in niggas_gang:
        im_swap,im_swap_2 = swap(*nigga, imagecv, imagecv_2, frames, patterns)

        if im_swap.any():
            imagecv = im_swap
        if im_swap_2.any():
            imagecv_2 = im_swap_2

    if imagecv.any():
        result_name = f"{home}/results/{str(int(time.time()))}.jpg"
    else:
        result_name = target_photo
    cv2.imwrite(result_name, imagecv)

    if imagecv_2.any():
        result2_name = f"{home}/results/{str(int(time.time()) + 1)}.jpg"
    else:
        result2_name = target_photo
    cv2.imwrite(result2_name, imagecv_2)

    return result_name, result2_name
